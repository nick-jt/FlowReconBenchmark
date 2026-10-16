#!/usr/bin/env python3
"""Cost instrumentation for the SiT-point baseline.

Emits the numbers the shared protocol requires, both greppably on stdout
(`[instr] key=value`) and as JSON in `<run_dir>/instrumentation.json`:

  train_step_time_s_excl_val   median wall-clock of ONE training epoch,
                               EXCLUDING validation (train_Gen_Baseline.py
                               times train and val epochs in separate calls;
                               loss_history.json's epoch_time_s is train-only)
  train_peak_gpu_mem_mb        max over epochs of torch.cuda.max_memory_allocated,
                               reset around the training epoch only
  inference_wall_clock_s       wall clock for ONE full 125^3 field
                               (1,953,125 points x 4 fields), K=1 sample,
                               chunked at the training token budget
  inference_peak_gpu_mem_mb    peak allocation during that full-field sample
  sampling_steps               ODE steps per chunk

Standalone script: defines nothing in the shared modules. Usage
    python instrument_sit.py --run-dir <dir> [--config <yaml>] [--checkpoint best|last]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from methods.common.config import ensure_absolute, infer_device, load_yaml, resolve_stage_config, safe_torch_load, validate_and_normalize_config
from methods.common.data import build_dataset
from methods.common.helpers_baseline import build_sparse_condition
from methods.common.registry import get_baseline_adapter
from methods.sit.adapter import sit_conditional_sample_points_chunked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SiT-point cost instrumentation")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=None,
                   help="Defaults to <run-dir>/run_config.yaml")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--snapshot-index", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--budget-hours", type=float, default=20.0)
    p.add_argument("--out", default=None,
                   help="Defaults to <run-dir>/instrumentation.json")
    return p.parse_args()


def training_cost(run_dir: Path, budget_hours: float, batch_size=None, skipped=0) -> dict:
    hist_path = run_dir / "loss_history.json"
    if not hist_path.exists():
        return {"note": "missing " + str(hist_path)}
    hist = json.loads(hist_path.read_text())
    times = [float(r["epoch_time_s"]) for r in hist if r.get("epoch_time_s") is not None]
    mems = [float(r["peak_gpu_mem_mb"]) for r in hist if r.get("peak_gpu_mem_mb") is not None]
    cumul = [(int(r["epoch"]), float(r.get("cumul_train_time_s") or 0.0)) for r in hist]
    budget_s = budget_hours * 3600.0
    budget_epoch = max((e for e, c in cumul if c <= budget_s), default=None)
    # Optimizer steps, so the budget is auditable in steps as well as seconds
    # even if I/O contention changes the seconds-per-step.
    import math as _m
    n_train = 150
    bs = int(batch_size) if batch_size else 24
    steps_per_epoch = _m.ceil(n_train / bs)
    n_steps_total = steps_per_epoch * len(hist)
    return {
        "epochs_completed": len(hist),
        "batches_per_epoch": steps_per_epoch,
        "optimizer_steps_total": n_steps_total,
        "optimizer_steps_skipped": int(skipped or 0),
        "optimizer_steps_applied": n_steps_total - int(skipped or 0),
        "s_per_optimizer_step": (sum(times) / n_steps_total) if times and n_steps_total else None,
        "train_step_time_s_median": statistics.median(times) if times else None,
        "train_step_time_s_mean": (sum(times) / len(times)) if times else None,
        "train_peak_gpu_mem_mb": max(mems) if mems else None,
        "total_train_wall_clock_h": (cumul[-1][1] / 3600.0) if cumul else None,
        "budget_hours": budget_hours,
        "budget_matched_epoch": budget_epoch,
    }


def main() -> None:
    args = parse_args()
    run_dir = ensure_absolute(args.run_dir)
    cfg_path = ensure_absolute(args.config) if args.config else run_dir / "run_config.yaml"
    cfg = validate_and_normalize_config(load_yaml(cfg_path))
    stage_cfg = resolve_stage_config(cfg)
    arch = stage_cfg["architecture"]
    sampling_cfg = stage_cfg["sampling"]
    n_steps = int(args.n_steps if args.n_steps is not None else sampling_cfg["sampling_N"])
    sampler_type = str(sampling_cfg["ode_solver"])
    node_subsample = int(arch.get("node_subsample") or 0)

    device = infer_device(None, cfg["shared"]["device_ids"])
    ckpt_path = run_dir / (args.checkpoint + ".pt")
    checkpoint = safe_torch_load(ckpt_path, map_location="cpu")

    dataset = build_dataset(cfg, split=args.split, stats_path=run_dir / "dataset_stats.pt")
    adapter = get_baseline_adapter(cfg["baseline_model"])
    bundle = adapter.build_for_training(
        cfg=cfg, device=device, run_dir=run_dir, train_set=dataset, val_set=dataset,
    )
    adapter.load_checkpoint(bundle, checkpoint)

    n_params = sum(p.numel() for p in bundle.model.parameters() if p.requires_grad)

    sample = dataset[args.snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    n_points = int(coords.shape[1])
    n_fields = int(truth.shape[-1])
    shared_cond = cfg["shared"]["conditioning"]

    with adapter.evaluation_weights(bundle):
        bundle.model.eval()
        obs = build_sparse_condition(
            coords_full=coords, fields_full=truth,
            cond_fields=shared_cond["vis_cond_fields"],
            n_obs_min=shared_cond["vis_n_obs_list"],
            n_obs_max=shared_cond["vis_n_obs_list"],
        )
        obs_coords, obs_values, obs_mask, _obs_idx, obs_field_ids = obs

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        _ = sit_conditional_sample_points_chunked(
            net=bundle.model, transport=bundle.components["transport"],
            coords=coords, obs_coords=obs_coords, obs_values=obs_values,
            obs_mask=obs_mask, obs_field_ids=obs_field_ids,
            n_fields=n_fields, device=device, n_steps=n_steps,
            sampler_type=sampler_type, chunk=int(node_subsample),
            sigma=float(arch.get("cond_fill_sigma", 0.05)),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        infer_s = time.perf_counter() - t0
        infer_mem = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                     if torch.cuda.is_available() else 0.0)

    payload = {
        "baseline_model": cfg["baseline_model"],
        "run_dir": str(run_dir),
        "config_path": str(cfg_path),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "trainable_parameters": int(n_params),
        "tokenizer": str(arch["tokenizer"]),
        "hidden_size": int(arch["hidden_size"]),
        "depth": int(arch["depth"]),
        "num_heads": int(arch["num_heads"]),
        "token_budget": node_subsample,
        "tokens_x_width_scalars": int(node_subsample * int(arch["hidden_size"])),
        "batch_size": int(stage_cfg["training"]["batch_size"]),
        "sampling_steps": n_steps,
        "ode_solver": sampler_type,
        "field_points": n_points,
        "field_chunks": (-(-n_points // node_subsample) if node_subsample else 1),
        "nfe_per_chunk": n_steps - 1,
        "inference_wall_clock_s": infer_s,
        "inference_peak_gpu_mem_mb": infer_mem,
        "training": training_cost(run_dir, args.budget_hours,
                                  batch_size=stage_cfg["training"]["batch_size"],
                                  skipped=(checkpoint.get("spike_state") or {}).get("skipped", 0)),
    }

    out_path = Path(args.out) if args.out else run_dir / "instrumentation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    tr = payload["training"]
    rows = [
        ("trainable_parameters", payload["trainable_parameters"]),
        ("train_step_time_s_excl_val", tr.get("train_step_time_s_median")),
        ("train_peak_gpu_mem_mb", tr.get("train_peak_gpu_mem_mb")),
        ("inference_wall_clock_s_full_125cube", infer_s),
        ("inference_peak_gpu_mem_mb", infer_mem),
        ("sampling_steps", n_steps),
        ("field_points", n_points),
        ("field_chunks", payload["field_chunks"]),
        ("total_train_wall_clock_h", tr.get("total_train_wall_clock_h")),
        ("budget_matched_epoch", tr.get("budget_matched_epoch")),
        ("optimizer_steps_total", tr.get("optimizer_steps_total")),
        ("optimizer_steps_skipped", tr.get("optimizer_steps_skipped")),
        ("s_per_optimizer_step", tr.get("s_per_optimizer_step")),
        ("json", str(out_path)),
    ]
    for k, v in rows:
        print("[instr] " + k + "=" + str(v), flush=True)


if __name__ == "__main__":
    main()
