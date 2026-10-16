from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import torch

try:
    from methods.common.adapters import TrainingHistoryLogger
    from methods.common.config import build_run_name, copy_config_backup, ensure_absolute, find_latest_run_dir, infer_device, load_yaml, resolve_stage_config, safe_torch_load, save_yaml, set_seed, validate_and_normalize_config
    from methods.common.data import build_dataloader, build_dataset
    from methods.common.registry import get_baseline_adapter
    from protocol import VAL_FIXED_SEED, fixed_validation_draw, log_param_count, resume_best_val
except ImportError:
    from .model_baseline import (
        TrainingHistoryLogger,
        build_dataloader,
        build_dataset,
        build_run_name,
        copy_config_backup,
        ensure_absolute,
        find_latest_run_dir,
        get_baseline_adapter,
        infer_device,
        load_yaml,
        resolve_stage_config,
        safe_torch_load,
        save_yaml,
        set_seed,
        validate_and_normalize_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified trainer for deterministic baselines.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_baseline_Det.yaml",
        help="Path to the unified deterministic YAML config.",
    )
    parser.add_argument("--baseline-model", type=str, default=None, help="Override baseline_model from YAML.")
    parser.add_argument("--training-stage", type=int, default=None, help="Override training_stage from YAML.")
    parser.add_argument("--device", type=str, default=None, help="Explicit device, e.g. cuda:0 or cpu.")
    parser.add_argument("--reload", action="store_true", help="Resume from the latest unified run for this model/stage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ensure_absolute(args.config)
    cfg = load_yaml(config_path)

    if args.baseline_model is not None:
        cfg["baseline_model"] = args.baseline_model
    if args.training_stage is not None:
        cfg["training_stage"] = int(args.training_stage)
    if args.reload:
        cfg.setdefault("shared", {}).update({"reload": True})

    cfg = validate_and_normalize_config(cfg)
    stage_cfg = resolve_stage_config(cfg)

    set_seed(int(cfg["shared"]["seed"]))
    device = infer_device(args.device, cfg["shared"]["device_ids"])
    save_root = ensure_absolute(cfg["shared"]["paths"]["save_root"])
    config_backup_root = ensure_absolute(cfg["shared"]["paths"]["config_backup_root"])
    adapter = get_baseline_adapter(cfg["baseline_model"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = build_run_name(cfg, timestamp)
    run_dir = save_root / run_name

    start_epoch = 1
    best_val = float("inf")
    reload_ckpt = None

    if bool(cfg["shared"]["reload"]):
        latest_run_dir = find_latest_run_dir(save_root, cfg)
        if latest_run_dir is not None:
            preferred_checkpoint = latest_run_dir / "last.pt"
            fallback_checkpoint = latest_run_dir / "best.pt"
            if preferred_checkpoint.exists():
                reload_ckpt = safe_torch_load(preferred_checkpoint, map_location="cpu")
                run_dir = latest_run_dir
            elif fallback_checkpoint.exists():
                reload_ckpt = safe_torch_load(fallback_checkpoint, map_location="cpu")
                run_dir = latest_run_dir

        if reload_ckpt is not None:
            start_epoch = int(reload_ckpt.get("epoch", 0)) + 1
            best_val = resume_best_val(reload_ckpt, run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_root = run_dir / "Evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)

    copy_config_backup(config_path, config_backup_root, run_name)
    save_yaml(run_dir / "run_config.yaml", cfg)
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_path": str(config_path),
                "baseline_model": cfg["baseline_model"],
                "training_stage": int(cfg["training_stage"]),
                "device": str(device),
                "started_at": timestamp,
                "val_split": "tune",
                "val_fixed_seed": VAL_FIXED_SEED,
                "val_n_obs": list(cfg["shared"]["conditioning"]["vis_n_obs_list"]),
            },
            handle,
            indent=2,
        )

    stats_path = run_dir / "dataset_stats.pt"
    train_set = build_dataset(cfg, split="train", stats_path=stats_path)
    val_set = build_dataset(cfg, split="tune", stats_path=stats_path)
    batch_size = int(stage_cfg["training"]["batch_size"])
    num_workers = int(cfg["shared"]["data"]["num_workers"])

    train_loader = build_dataloader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = build_dataloader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    bundle = adapter.build_for_training(
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        train_set=train_set,
        val_set=val_set,
    )
    if reload_ckpt is not None:
        adapter.load_checkpoint(bundle, reload_ckpt)
    log_param_count(bundle.model, cfg["baseline_model"])

    logger = TrainingHistoryLogger(run_dir)

    print(f"Config:         {config_path}")
    print(f"Run directory:  {run_dir}")
    print(f"Baseline:       {cfg['baseline_model']}")
    print(f"Stage:          {cfg['training_stage']}")
    print(f"Device:         {device}")
    print(f"Train samples:  {len(train_set)}")
    print(f"Val samples:    {len(val_set)}")
    if reload_ckpt is not None:
        print(f"Resuming from epoch {start_epoch}")

    total_epochs = int(stage_cfg["training"]["epochs"])
    eval_every = int(stage_cfg["training"]["eval_every"])
    save_every = int(stage_cfg["training"]["save_every"])

    from methods.common.loss_plot import LossTracker

    _tracker = LossTracker(run_dir, name='det_baseline')

    # Optional wall-clock budget (hours).  Opt-in: unset/0 keeps the previous
    # behaviour of running the full epoch count.  Used for budget-matched
    # baseline comparisons; on expiry the loop breaks cleanly (real exit code 0)
    # after writing a `budget.pt` checkpoint.
    _budget_h = float(os.environ.get("BASELINE_MAX_HOURS", "0") or 0.0)
    _run_t0 = time.perf_counter()
    _steps_per_epoch = max(1, len(train_loader))
    # Archive ~N checkpoints across the tail of the budget so checkpoint noise
    # (metric spread across nearby checkpoints) is measurable after the fact.
    # Opt-in: inert unless BASELINE_MAX_HOURS is set.
    _archive_n = int(os.environ.get("BASELINE_ARCHIVE_N", "10") or 0)
    _archive_frac = float(os.environ.get("BASELINE_ARCHIVE_TAIL_FRAC", "0.05"))
    _archive_every_s = (_budget_h * 3600.0 * _archive_frac / _archive_n) if (_budget_h > 0 and _archive_n > 0) else 0.0
    _archive_start_s = _budget_h * 3600.0 * (1.0 - _archive_frac)
    _archive_last = 0.0
    _archive_i = 0
    # Optimizer steps are the I/O-independent unit of work.  Wall-clock alone
    # cannot be compared across runs when shared-filesystem contention varies
    # between them, so report both, plus the duty cycle (fraction of elapsed
    # wall-clock that was actual compute).
    _total_steps = 0
    _proc_t0 = _run_t0
    _cost = {
        "steps_per_epoch": _steps_per_epoch,
        "train_step_time_s_excl_val": None,
        "train_epoch_time_s_excl_val": None,
        "train_peak_gpu_mem_mb": None,
        "budget_hours": _budget_h,
        "budget_matched_epoch": None,
        "total_optimizer_steps": 0,
        "batch_size": batch_size,
        "samples_seen": 0,
        "epochs_completed": 0,
        "cumul_train_compute_s": 0.0,
        "wall_elapsed_s": 0.0,
        "duty_cycle_compute_over_wall": None,
    }

    def _refresh_cost(epoch_done: int, cumul_compute: float) -> None:
        _cost["total_optimizer_steps"] = int(_total_steps)
        _cost["samples_seen"] = int(_total_steps) * batch_size
        _cost["epochs_completed"] = int(epoch_done)
        _cost["cumul_train_compute_s"] = round(float(cumul_compute), 3)
        _w = time.perf_counter() - _proc_t0
        _cost["wall_elapsed_s"] = round(_w, 3)
        _cost["duty_cycle_compute_over_wall"] = round(cumul_compute / _w, 4) if _w > 0 else None
        with open(run_dir / "cost_train.json", "w", encoding="utf-8") as _h:
            json.dump(_cost, _h, indent=2)

    for epoch in range(start_epoch, total_epochs + 1):
        # Cost instrumentation, matching the point-cloud and latent-FM
        # trainers so per-epoch time and peak memory are comparable across
        # every model in the benchmark.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        _t0 = time.perf_counter()
        train_loss = adapter.run_epoch(bundle, train_loader, training=True, epoch=epoch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            tr_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            tr_mem = 0.0
        tr_time = time.perf_counter() - _t0
        # Cost instrumentation, greppable.  tr_time deliberately excludes the
        # validation pass, which runs further down.
        _step_time = tr_time / _steps_per_epoch
        print(
            f"[cost] epoch={epoch:04d} train_step_time_s_excl_val={_step_time:.4f}"
            f" steps_per_epoch={_steps_per_epoch}"
            f" train_epoch_time_s_excl_val={tr_time:.3f}"
            f" train_peak_gpu_mem_mb={tr_mem:.1f}",
            flush=True,
        )
        _cost["train_step_time_s_excl_val"] = round(_step_time, 4)
        _cost["train_epoch_time_s_excl_val"] = round(tr_time, 3)
        _cost["train_peak_gpu_mem_mb"] = round(float(tr_mem), 1)
        _total_steps += _steps_per_epoch
        _refresh_cost(epoch, logger._cumul_train_time_s + tr_time)
        print(
            f"[cost] epoch={epoch:04d} total_optimizer_steps={_total_steps}"
            f" duty_cycle={_cost['duty_cycle_compute_over_wall']}"
            f" wall_elapsed_s={_cost['wall_elapsed_s']:.1f}",
            flush=True,
        )
        if bundle.scheduler is not None:
            bundle.scheduler.step()

        val_loss = None
        if epoch % eval_every == 0 or epoch == 1:
            with torch.no_grad(), fixed_validation_draw(cfg["shared"]["conditioning"]):
                val_loss = adapter.run_epoch(bundle, val_loader, training=False, epoch=epoch)

            checkpoint = adapter.build_checkpoint(
                bundle=bundle,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            checkpoint["run_dir"] = str(run_dir)
            checkpoint["run_name"] = run_dir.name
            checkpoint["config"] = cfg
            checkpoint["best_val"] = min(best_val, float(val_loss))

            torch.save(checkpoint, run_dir / "last.pt")
            if float(val_loss) < best_val:
                best_val = float(val_loss)
                torch.save(checkpoint, run_dir / "best.pt")

        _tracker.log(step=epoch, train_loss=train_loss, val_loss=val_loss)
        _tracker.plot()
        if epoch % save_every == 0:
            epoch_eval_dir = evaluation_root / f"epoch_{epoch:04d}"
            epoch_eval_dir.mkdir(parents=True, exist_ok=True)
            metrics = adapter.visualize(
                bundle=bundle,
                dataset=val_set,
                save_dir=epoch_eval_dir,
                epoch=epoch,
                snapshot_index=0,
            )
            metric_text = ", ".join(f"{name}:{value:.4e}" for name, value in metrics.items())
            print(f"[eval] epoch={epoch:04d} {metric_text}")

        logger.append(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                      epoch_time_s=tr_time, peak_gpu_mem_mb=tr_mem)
        train_msg = (f"[train] epoch={epoch:04d} loss={train_loss:.6e}"
                     f"  time={tr_time:.1f}s  peak_mem={tr_mem:.0f}MB")
        if val_loss is not None:
            train_msg += f" val={val_loss:.6e}"
        print(train_msg)

        _elapsed = time.perf_counter() - _run_t0
        if (_archive_every_s > 0 and _archive_i < _archive_n
                and _elapsed >= _archive_start_s
                and (_elapsed - _archive_last) >= _archive_every_s):
            _arc_dir = run_dir / "archive"
            _arc_dir.mkdir(parents=True, exist_ok=True)
            _arc = adapter.build_checkpoint(
                bundle=bundle, epoch=epoch, train_loss=train_loss,
                val_loss=float(val_loss) if val_loss is not None else float("nan"),
            )
            _arc["run_dir"] = str(run_dir)
            _arc["run_name"] = run_dir.name
            _arc["config"] = cfg
            torch.save(_arc, _arc_dir / f"ckpt_{_archive_i:02d}_epoch{epoch:06d}.pt")
            print(f"[archive] {_archive_i + 1}/{_archive_n} epoch={epoch}", flush=True)
            _archive_i += 1
            _archive_last = _elapsed

        if _budget_h > 0 and _elapsed >= _budget_h * 3600.0:
            _budget_ckpt = adapter.build_checkpoint(
                bundle=bundle, epoch=epoch, train_loss=train_loss,
                val_loss=float(val_loss) if val_loss is not None else float("nan"),
            )
            _budget_ckpt["run_dir"] = str(run_dir)
            _budget_ckpt["run_name"] = run_dir.name
            _budget_ckpt["config"] = cfg
            torch.save(_budget_ckpt, run_dir / "budget.pt")
            _cost["budget_matched_epoch"] = int(epoch)
            with open(run_dir / "cost_train.json", "w", encoding="utf-8") as _h:
                json.dump(_cost, _h, indent=2)
            print(
                f"[budget] wall-clock budget {_budget_h}h reached at epoch {epoch}; "
                f"wrote budget.pt and stopping.",
                flush=True,
            )
            break

    _refresh_cost(_cost["epochs_completed"], logger._cumul_train_time_s)
    print("Training complete.")
    print(f"Best val loss: {best_val:.6e}")
    print(f"Run directory: {run_dir}")
    print(
        f"[cost] FINAL total_optimizer_steps={_cost['total_optimizer_steps']}"
        f" epochs={_cost['epochs_completed']}"
        f" cumul_train_compute_s={_cost['cumul_train_compute_s']:.1f}"
        f" wall_elapsed_s={_cost['wall_elapsed_s']:.1f}"
        f" duty_cycle={_cost['duty_cycle_compute_over_wall']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
