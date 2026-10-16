"""Unified CoNFiLD training with the upstream model and optimizer semantics."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

try:
    from methods.confild.upstream_core import FieldStatistics, PackedJHUCubes, batches, build_latent_windows, import_upstream_decoder, item_to_snapshot_group, octahedral_gather, octahedral_transform
except ImportError:
    from .confild_upstream_core import (
        FieldStatistics,
        PackedJHUCubes,
        batches,
        build_latent_windows,
        import_upstream_decoder,
        item_to_snapshot_group,
        octahedral_gather,
        octahedral_transform,
    )


# Set when the module loads, i.e. at stage entry. Used for the duty cycle, so
# that startup I/O (statistics pass + snapshot cache) is counted against the
# budget rather than hidden.
_PROCESS_START = time.time()


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "CoNFiLD training is GPU-only in the unified trainer. "
            "Request a CUDA node and pass --device cuda:0."
        )
    torch.cuda.set_device(device)


def _packed_dataset(cfg: dict) -> PackedJHUCubes:
    path = Path(cfg["shared"]["paths"]["data_path"]).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        n_snapshots = int(handle["fields"].shape[1])
    train_count = int(n_snapshots * float(cfg["shared"]["data"]["train_ratio"]))
    if not 0 < train_count < n_snapshots:
        raise ValueError("CoNFiLD requires non-empty train and held-out splits.")
    return PackedJHUCubes(
        path,
        train_count=train_count,
        val_start=train_count,
        val_count=n_snapshots - train_count,
    )


def _statistics(dataset: PackedJHUCubes, run_dir: Path) -> FieldStatistics:
    path = run_dir / "field_statistics.pt"
    if path.exists():
        return FieldStatistics.from_state_dict(torch.load(path, map_location="cpu"))
    statistics = dataset.compute_train_statistics()
    torch.save(statistics.state_dict(), path)
    return statistics


def _stage1_checkpoint(
    *, cfg: dict, model: torch.nn.Module, latents: torch.Tensor,
    decoder_optimizer: torch.optim.Optimizer,
    latent_optimizer: torch.optim.Optimizer, epoch: int,
    dataset: PackedJHUCubes, statistics: FieldStatistics,
    best_train_rel_l2: float,
) -> dict:
    stage = cfg["confild_params"]["stage1"]
    return {
        "format_version": 2,
        "baseline_model": "confild",
        "training_stage": 1,
        "epoch": int(epoch),
        "decoder": model.state_dict(),
        "latents": latents.detach().cpu(),
        "decoder_optimizer": decoder_optimizer.state_dict(),
        "latent_optimizer": latent_optimizer.state_dict(),
        # Upstream steps the accumulated decoder gradients at the beginning of
        # the next epoch. Persist them so a unified resume does not drop that
        # pending update (the original checkpoint code does drop it).
        "decoder_gradients": [
            None if parameter.grad is None else parameter.grad.detach().cpu()
            for parameter in model.parameters()
        ],
        "field_statistics": statistics.state_dict(),
        "coord_min": dataset.coord_min,
        "coord_max": dataset.coord_max,
        "grid_shape": dataset.grid_shape,
        "train_indices": torch.as_tensor(dataset.train_indices),
        "val_indices": torch.as_tensor(dataset.val_indices),
        "best_train_rel_l2": float(best_train_rel_l2),
        "groups": int(stage["augmentation"]["groups"]),
        "architecture": copy.deepcopy(stage["architecture"]),
        "config": copy.deepcopy(cfg),
        "method": "CoNFiLD_upstream_SIREN_autodecoder",
    }


def _rel_l2_zscore(prediction, truth, statistics):
    """Relative L2 in the BENCHMARK z-scored units every baseline is scored in.

    The pm1 rel-L2 is not comparable across normalisation schemes and is
    optimistically biased under global min/max scaling, because the normalised
    field carries a DC offset (measured: p occupies -0.02..0.56, never straddling
    zero) which inflates the denominator. This one cancels the offset and is the
    quantity that lines up with ensemble_eval's reported error.
    """
    pz = statistics.benchmark_normalize(statistics.denormalize(prediction))
    tz = statistics.benchmark_normalize(statistics.denormalize(truth))
    return float(torch.linalg.vector_norm(pz - tz)
                 / torch.linalg.vector_norm(tz).clamp_min(1e-12))


def _diagnostic(
    model: torch.nn.Module, latents: torch.Tensor,
    snapshots: list[torch.Tensor] | None, dataset: PackedJHUCubes,
    statistics: FieldStatistics, groups: int, item_count: int,
    point_count: int, device: torch.device,
) -> dict[str, float]:
    model.eval()
    errors, errors_z = [], []
    selected = torch.linspace(0, latents.shape[0] - 1, min(item_count, latents.shape[0])).long()
    point_ids = torch.linspace(0, dataset.num_points - 1, min(point_count, dataset.num_points)).long()
    coords = dataset.coords[point_ids].to(device).unsqueeze(0)
    stats_device = statistics.to(device)
    with torch.no_grad():
        for item in selected.tolist():
            snapshot_index, group_index = item_to_snapshot_group(item, groups)
            normalised = (snapshots[snapshot_index] if snapshots is not None
                          else statistics.normalize(dataset.read_snapshot(snapshot_index)))
            if group_index:
                values = octahedral_gather(normalised, dataset.grid_shape, group_index, point_ids)
            else:
                values = normalised[point_ids]
            truth = values.to(device)
            prediction = model(coords, latents[item].view(1, 1, -1))[0]
            error = torch.linalg.vector_norm(prediction - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-12)
            errors.append(float(error))
            errors_z.append(_rel_l2_zscore(prediction, truth, stats_device))
    model.train()
    return {"train_rel_l2": float(np.mean(errors)),
            "train_rel_l2_std": float(np.std(errors)),
            "train_rel_l2_zscore": float(np.mean(errors_z))}


class FigureScheduler:
    """Fire ~n_figures times across a stage, by FRACTION OF PROGRESS.

    A fixed `every N epochs` cadence cannot work here: the three arms run at
    35-58 s/epoch and every stage is truncated by a wall-clock budget, so a
    constant N gives a different figure count per arm (and `save_every: 5000`
    on a stage that never reaches 5,000 steps gives none at all). Keying off
    max(elapsed/budget, done/total) yields ~n_figures for every arm regardless.
    """

    def __init__(self, n_figures: int, budget_s: float, total_units: int, started: float):
        self.n = max(1, int(n_figures))
        self.budget = float(budget_s or 0.0)
        self.total = max(1, int(total_units))
        self.started = started
        self.fired = 0

    def due(self, unit: int) -> bool:
        if self.fired == 0:
            self.fired = 1
            return True
        fraction = unit / self.total
        if self.budget > 0:
            fraction = max(fraction, (time.time() - self.started) / self.budget)
        if fraction >= self.fired / self.n:
            self.fired = min(self.n, int(fraction * self.n) + 1)
            return True
        return False


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_field_figure(path, grid_shape, field_names, title, truth=None,
                      prediction=None, rows_extra=()):
    """z-midplane slice: truth / prediction / |error| per channel.

    This is the panel that would have caught the four collapsed arms on sight:
    a decoder that has settled on a latent-independent mean field produces a
    prediction row that is visibly the same picture for every snapshot while
    the training loss keeps descending.
    """
    plt = _pyplot()
    nx, ny, nz = (int(v) for v in grid_shape)
    def cube(a):
        return None if a is None else np.asarray(a, dtype=np.float32).reshape(nx, ny, nz, -1)
    truth, prediction = cube(truth), cube(prediction)
    rows = []
    if truth is not None:
        rows.append(("truth", truth, None))
    if prediction is not None:
        rows.append(("prediction", prediction, None))
    if truth is not None and prediction is not None:
        rows.append(("|error|", np.abs(prediction - truth), "magma"))
    for label, array, cmap in rows_extra:
        rows.append((label, cube(array), cmap))
    reference = truth if truth is not None else prediction
    n_channel = min(len(field_names), reference.shape[-1])
    mid = nz // 2
    fig, axs = plt.subplots(len(rows), n_channel,
                            figsize=(3.05 * n_channel, 3.0 * len(rows)), squeeze=False)
    for c in range(n_channel):
        lo, hi = np.percentile(reference[:, :, mid, c], [1, 99])
        for r, (label, array, cmap) in enumerate(rows):
            ax = axs[r][c]
            plane = array[:, :, mid, c]
            if cmap is None:
                im = ax.imshow(plane, cmap="coolwarm", vmin=lo, vmax=hi)
            else:
                im = ax.imshow(plane, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(field_names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def latent_dependence(model, latents, coords, device, n_points=8192, n_latents=5):
    """Across-latent output std / field std. ~0 means the FiLM path is inert."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        ids = torch.randint(0, coords.shape[0], (n_points,), device=device)
        query = coords[ids].unsqueeze(0)
        picks = torch.linspace(0, latents.shape[0] - 1, n_latents).long()
        outs = torch.stack([model(query, latents[j].view(1, 1, -1))[0] for j in picks])
        ratio = float(outs.std(0).mean() / (outs[0].std() + 1e-12))
    if was_training:
        model.train()
    return ratio


def fit_latent(model, coords, target, latent_dim, device, steps=400,
               points=16384, lr=1.0e-2, seed=9):
    """Test-time latent fit with the decoder FROZEN (held-out codec bound).

    requires_grad is toggled off on the decoder for the duration: stage 1
    accumulates decoder gradients across a whole epoch and steps them at the
    next epoch boundary, so letting this backward touch `.grad` would corrupt
    the pending update.
    """
    flags = [q.requires_grad for q in model.parameters()]
    for q in model.parameters():
        q.requires_grad_(False)
    was_training = model.training
    model.eval()
    z = torch.nn.Parameter(torch.zeros(1, 1, latent_dim, device=device))
    optimizer = torch.optim.Adam([z], lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        ids = torch.randint(target.shape[0], (points,), generator=gen).to(device)
        loss = F.mse_loss(model(coords[ids].unsqueeze(0), z), target[ids].unsqueeze(0))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for q, flag in zip(model.parameters(), flags):
        q.requires_grad_(flag)
    if was_training:
        model.train()
    return z.detach()


def decode_field(model, latent, coords, n_points, device, chunk=262144):
    was_training = model.training
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, n_points, chunk):
            stop = min(start + chunk, n_points)
            pieces.append(model(coords[start:stop].unsqueeze(0), latent.view(1, 1, -1))[0])
    if was_training:
        model.train()
    return torch.cat(pieces, 0)


def _decode_full_field(model, latent, dataset, device, chunk: int = 262144) -> None:
    coords = dataset.coords.to(device)
    with torch.no_grad():
        for start in range(0, dataset.num_points, chunk):
            stop = min(start + chunk, dataset.num_points)
            model(coords[start:stop].unsqueeze(0), latent.view(1, 1, -1))


def benchmark_decode(model, latent, dataset, device, chunk: int = 262144) -> tuple[float, float]:
    """Wall-clock and peak GPU memory for decoding ONE full 125^3 x 4 field.

    Warm pass first so the number is a steady-state inference cost, not a
    kernel-autotune cost.
    """
    was_training = model.training
    model.eval()
    _decode_full_field(model, latent, dataset, device, chunk)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    _decode_full_field(model, latent, dataset, device, chunk)
    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    if was_training:
        model.train()
    return elapsed, peak


def _merge_instrumentation(path: Path, stage: int, payload: dict) -> dict:
    """Merge one stage's payload into `path` and recompute the cross-stage total."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[f"stage{int(stage)}"] = payload
    total: dict[str, float] = {}
    for name, record in existing.items():
        # Skip the previously written total, or it double-counts itself.
        if name == "total" or not isinstance(record, dict):
            continue
        for key in ("train_step_time_s", "inference_wallclock_s"):
            total[key] = total.get(key, 0.0) + float(record.get(key, 0.0))
        for key in ("train_peak_gpu_mem_mb", "inference_peak_gpu_mem_mb"):
            total[key] = max(total.get(key, 0.0), float(record.get(key, 0.0)))
        for key in ("optimizer_steps_decoder", "optimizer_steps_latent",
                    "optimizer_steps_diffusion"):
            if key in record:
                total[key] = total.get(key, 0.0) + float(record[key])
    job_start = os.environ.get("CONFILD_JOB_START")
    if job_start:
        try:
            job_elapsed = time.time() - float(job_start)
            total["job_elapsed_s"] = float(job_elapsed)
            # All optimizer time over the whole allocation: the fraction of the
            # ~19.5 h budget that was real compute.
            total["duty_cycle_job"] = float(
                total.get("train_step_time_s", 0.0) / max(job_elapsed, 1e-9))
        except ValueError:
            pass
    existing["total"] = total
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return total


def emit_instrumentation(run_dir: Path, stage: int, metrics: dict) -> dict:
    """Greppable + JSON. Keys are identical across every baseline in the paper."""
    payload = {"stage": int(stage), **{k: float(v) for k, v in metrics.items()}}
    process_elapsed = time.time() - _PROCESS_START
    payload["process_elapsed_s"] = float(process_elapsed)
    payload["duty_cycle"] = float(
        payload.get("train_step_time_s", 0.0) / max(process_elapsed, 1e-9))
    # Per-STAGE start, re-stamped by the launcher before each stage. A single
    # job-wide stamp made stage 2's denominator include the whole of stage 1
    # (measured duty_cycle_vs_slurm=0.024 for a stage that was really ~0.17),
    # which would have gone into the paper as a fake stall.
    slurm_start = os.environ.get("CONFILD_SLURM_START")
    if slurm_start:
        try:
            slurm_elapsed = time.time() - float(slurm_start)
            payload["slurm_elapsed_s"] = float(slurm_elapsed)
            payload["duty_cycle_vs_slurm"] = float(
                payload.get("train_step_time_s", 0.0) / max(slurm_elapsed, 1e-9))
        except ValueError:
            pass
    print(
        "[confild:instrument] "
        + " ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in payload.items()),
        flush=True,
    )
    # Per-stage file, plus a job-level file so the two stages actually total.
    # Each stage runs in its OWN run_dir, so a run-dir-local file can never see
    # the other stage; the job-level file lives in the shared save_root.
    _merge_instrumentation(run_dir / "instrumentation.json", stage, payload)
    job_id = os.environ.get("SLURM_JOB_ID")
    job_path = (run_dir.parent / f"instrumentation_job{job_id}.json") if job_id else None
    total = (_merge_instrumentation(job_path, stage, payload) if job_path
             else json.loads((run_dir / "instrumentation.json").read_text())["total"])
    print(
        "[confild:instrument] "
        + " ".join(f"total_{k}={v:.6g}" for k, v in sorted(total.items())),
        flush=True,
    )
    return payload


def train_stage1(cfg: dict, device: torch.device, run_dir: Path, resume: dict | None) -> dict:
    _require_cuda(device)
    stage = cfg["confild_params"]["stage1"]
    training = stage["training"]
    architecture = stage["architecture"]
    if bool(architecture.get("tie_latent_to_hidden", True)) and int(architecture["latent_dim"]) != int(architecture["hidden_features"]):
        raise ValueError(
            "Upstream CoNFiLD ties the lumped latent dimension to hidden_size; "
            "set stage1 architecture.latent_dim == hidden_features."
        )
    groups = int(stage["augmentation"]["groups"])
    dataset = _packed_dataset(cfg)
    statistics = _statistics(dataset, run_dir)
    cache = bool(training.get("cache_snapshots", True))
    # NORMALISE ONCE, THEN AUGMENT.
    #
    # The min/max are per-point arrays reduced over time. The octahedral action
    # moves a value from source point s to target point t, so normalising the
    # gathered value with the TARGET point's statistics uses the wrong point's
    # range for 47 of the 48 group elements -- measured across the JHU cube,
    # the per-point range has CV 0.10-0.20 and a p99/p1 spread of 1.57-2.41,
    # which puts 48-72% of points more than 10% off scale (36% for p).
    #
    # normalize(g.f ; g.stats) == g.normalize(f ; stats) exactly, for all 48
    # elements (verified numerically, including the sign flips under which min
    # and max swap and negate). So normalising the snapshots once up front and
    # applying the group action to the NORMALISED field is both exact and free:
    # the hot loop does one gather and no statistics indexing at all.
    snapshots = ([statistics.normalize(f) for f in dataset.snapshots(dataset.train_indices)]
                 if cache else None)
    n_items = len(dataset.train_indices) * groups

    Decoder = import_upstream_decoder(cfg["confild_params"]["upstream_root"])
    model = Decoder(
        in_coord_features=3,
        in_latent_features=int(architecture["latent_dim"]),
        out_features=dataset.num_fields,
        num_hidden_layers=int(architecture["layers"]),
        hidden_features=int(architecture["hidden_features"]),
    ).to(device)
    decoder_parameters = sum(parameter.numel() for parameter in model.parameters())
    latents = torch.nn.Parameter(torch.zeros(n_items, int(architecture["latent_dim"]), device=device))
    decoder_optimizer = torch.optim.Adam(model.parameters(), lr=float(training["decoder_learning_rate"]))
    latent_optimizer = torch.optim.Adam([latents], lr=float(training["latent_learning_rate"]))
    start_epoch = 0
    best = math.inf
    if resume is not None:
        model.load_state_dict(resume["decoder"])
        latents.data.copy_(resume["latents"].to(device))
        decoder_optimizer.load_state_dict(resume["decoder_optimizer"])
        latent_optimizer.load_state_dict(resume["latent_optimizer"])
        for parameter, gradient in zip(model.parameters(), resume.get("decoder_gradients", [])):
            if gradient is not None:
                parameter.grad = gradient.to(device)
        start_epoch = int(resume["epoch"]) + 1
        best = float(resume.get("best_train_rel_l2", math.inf))

    coords = dataset.coords.to(device)
    stats_device = statistics.to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["shared"]["seed"]) + start_epoch)
    diagnostic_every = int(training["diagnostic_every"])
    save_every = int(training["save_every"])
    total_epochs = int(training["epochs"])
    wallclock_budget_s = float(training.get("wallclock_budget_s", 0.0))
    started = time.time()
    train_step_time_s = 0.0
    train_peak_gpu_mem_mb = 0.0
    decoder_steps = 0
    latent_steps = 0

    # ---- periodic visual diagnostics -------------------------------------
    # Fixed subjects, so the series across a run (and across the three arms)
    # is comparable frame to frame.
    figures_dir = run_dir / "Evaluation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    field_names = list(cfg["shared"]["data"]["field_names"])
    vis = training.get("visualisation", {}) or {}
    scheduler = FigureScheduler(int(vis.get("n_figures", 30)), wallclock_budget_s,
                                total_epochs, started)
    vis_item = int(vis.get("train_item", 0))
    vis_val = int(vis.get("val_offset", 0))
    tta_steps = int(vis.get("tta_steps", 400))
    from methods.common.loss_plot import LossTracker
    tracker = LossTracker(run_dir, name="confild_stage1")

    vis_snapshot, vis_group = item_to_snapshot_group(vis_item, groups)
    _phys = (snapshots[vis_snapshot] if snapshots is not None
             else statistics.normalize(dataset.read_snapshot(vis_snapshot)))
    if vis_group:
        _phys = octahedral_transform(_phys, dataset.grid_shape, vis_group)
    vis_train_truth = _phys.to(device)
    _val_index = int(dataset.val_indices[vis_val])
    vis_val_truth = stats_device.normalize(dataset.read_snapshot(_val_index).to(device))
    del _phys
    print(f"[confild:stage1] figures -> {figures_dir} "
          f"(~{scheduler.n} per stage; train item {vis_item}, held-out snapshot "
          f"{_val_index})", flush=True)

    print(
        f"[confild:stage1] device={device} snapshots={len(dataset.train_indices)} "
        f"groups={groups} items={n_items} decoder_parameters={decoder_parameters:,} "
        f"latent_table_values={latents.numel():,}",
        flush=True,
    )

    for epoch in range(start_epoch, total_epochs):
        torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.time()
        model.train()
        # This placement matches upstream scripts/train.py:400-401. Decoder
        # gradients from epoch i are applied immediately before epoch i+1.
        if epoch != 0:
            decoder_optimizer.step()
            decoder_optimizer.zero_grad(set_to_none=True)
            decoder_steps += 1
        losses = []
        for item_ids_cpu in batches(torch.randperm(n_items, generator=generator), int(training["batch_size"])):
            point_ids = torch.randint(
                dataset.num_points,
                (item_ids_cpu.numel(), int(training["points_per_item"])),
                generator=generator,
            )
            batch_coords, batch_truth = [], []
            for row, item in enumerate(item_ids_cpu.tolist()):
                snapshot_index, group_index = item_to_snapshot_group(item, groups)
                normalised = (snapshots[snapshot_index] if snapshots is not None
                              else statistics.normalize(dataset.read_snapshot(snapshot_index)))
                ids = point_ids[row]
                # The snapshots are already normalised, so the group action is
                # applied to normalised values -- see the note above. Equivalent
                # to transforming the whole field and indexing, but
                # O(points_per_item) rather than O(1.95M).
                values = (
                    octahedral_gather(normalised, dataset.grid_shape, group_index, ids)
                    if group_index
                    else normalised[ids]
                )
                batch_coords.append(coords[ids.to(device)])
                batch_truth.append(values.to(device))
            prediction = model(
                torch.stack(batch_coords),
                latents[item_ids_cpu.to(device)].unsqueeze(1),
            )
            loss = F.mse_loss(prediction, torch.stack(batch_truth))
            latent_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            latent_optimizer.step()
            latent_steps += 1
            losses.append(float(loss.detach()))

        torch.cuda.synchronize(device)
        # Measured BEFORE the diagnostic block below, so it excludes validation.
        epoch_time_s = time.time() - epoch_started
        epoch_peak_mb = torch.cuda.max_memory_allocated(device) / 2**20
        train_step_time_s += epoch_time_s
        train_peak_gpu_mem_mb = max(train_peak_gpu_mem_mb, epoch_peak_mb)
        record = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "latent_rms": float(latents.detach().square().mean().sqrt()),
            "epoch_time_s": epoch_time_s,
            "peak_gpu_mem_mb": epoch_peak_mb,
            "train_step_time_s_cumulative": train_step_time_s,
            "decoder_steps": decoder_steps,
            "latent_steps": latent_steps,
            "elapsed_seconds": time.time() - started,
        }
        if (epoch + 1) % diagnostic_every == 0 or epoch == start_epoch:
            record.update(_diagnostic(
                model, latents, snapshots, dataset, statistics, groups,
                int(training["diagnostic_items"]), int(training["diagnostic_points"]), device,
            ))
            if record["train_rel_l2"] < best:
                best = record["train_rel_l2"]
                torch.save(_stage1_checkpoint(
                    cfg=cfg, model=model, latents=latents,
                    decoder_optimizer=decoder_optimizer, latent_optimizer=latent_optimizer,
                    epoch=epoch, dataset=dataset, statistics=statistics,
                    best_train_rel_l2=best,
                ), run_dir / "best.pt")
        if scheduler.due(epoch + 1):
            figure_started = time.time()
            record["latent_dependence"] = latent_dependence(model, latents, coords, device)
            # (a) a TRAINED item, decoded from its own learned latent. This is
            #     the direct picture of latent-independent collapse.
            train_pred = decode_field(model, latents[vis_item].detach(), coords,
                                      dataset.num_points, device)
            save_field_figure(
                figures_dir / f"stage1_train_item{vis_item}_ep{epoch:05d}.png",
                dataset.grid_shape, field_names,
                f"CoNFiLD stage 1 - trained item {vis_item} (snapshot {vis_snapshot}, "
                f"group {vis_group})   epoch {epoch}   "
                f"latent-dependence {record['latent_dependence']:.3f}",
                truth=vis_train_truth.cpu().numpy(), prediction=train_pred.cpu().numpy())
            # (b) a HELD-OUT snapshot via a frozen-decoder latent fit: the codec
            #     bound, and the panel that is comparable across the three arms.
            z = fit_latent(model, coords, vis_val_truth, int(architecture["latent_dim"]),
                           device, steps=tta_steps)
            val_pred = decode_field(model, z, coords, dataset.num_points, device)
            val_rel = float(torch.linalg.vector_norm(val_pred - vis_val_truth)
                            / torch.linalg.vector_norm(vis_val_truth).clamp_min(1e-12))
            record["heldout_codec_rel_l2"] = val_rel
            record["heldout_codec_rel_l2_zscore"] = _rel_l2_zscore(
                val_pred, vis_val_truth, stats_device)
            save_field_figure(
                figures_dir / f"stage1_heldout_snap{_val_index}_ep{epoch:05d}.png",
                dataset.grid_shape, field_names,
                f"CoNFiLD stage 1 - HELD-OUT snapshot {_val_index}, frozen-decoder "
                f"latent fit ({tta_steps} steps)   epoch {epoch}   rel-L2 {val_rel:.4f}",
                truth=vis_val_truth.cpu().numpy(), prediction=val_pred.cpu().numpy())
            del train_pred, val_pred
            record["figure_seconds"] = time.time() - figure_started
            print(f"[confild:figure] stage=1 epoch={epoch} "
                  f"latent_dependence={record['latent_dependence']:.4f} "
                  f"heldout_codec_rel_l2={val_rel:.4f} "
                  f"heldout_codec_rel_l2_zscore={record['heldout_codec_rel_l2_zscore']:.4f} "
                  f"seconds={record['figure_seconds']:.1f}", flush=True)
        tracker.log(step=epoch, train_mse=record["train_mse"],
                    latent_rms=record["latent_rms"],
                    train_rel_l2=record.get("train_rel_l2"),
                    latent_dependence=record.get("latent_dependence"),
                    heldout_codec_rel_l2=record.get("heldout_codec_rel_l2_zscore"))
        if (epoch % 10) == 0 or scheduler.fired >= scheduler.n:
            tracker.plot()
        _append_jsonl(run_dir / "history.jsonl", record)
        print("[confild:stage1] " + " ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
        if (epoch + 1) % save_every == 0 or epoch == total_epochs - 1:
            torch.save(_stage1_checkpoint(
                cfg=cfg, model=model, latents=latents,
                decoder_optimizer=decoder_optimizer, latent_optimizer=latent_optimizer,
                epoch=epoch, dataset=dataset, statistics=statistics,
                best_train_rel_l2=best,
            ), run_dir / "last.pt")
        if wallclock_budget_s and (time.time() - started) >= wallclock_budget_s:
            # Budget-matched stop. last.pt is the budget-matched checkpoint,
            # best.pt the diagnostic-selected one; both are reported.
            torch.save(_stage1_checkpoint(
                cfg=cfg, model=model, latents=latents,
                decoder_optimizer=decoder_optimizer, latent_optimizer=latent_optimizer,
                epoch=epoch, dataset=dataset, statistics=statistics,
                best_train_rel_l2=best,
            ), run_dir / "last.pt")
            total_epochs = epoch + 1
            print(f"[confild:stage1] wall-clock budget reached at epoch {epoch}", flush=True)
            break

    tracker.plot()
    inference_wallclock_s, inference_peak_gpu_mem_mb = benchmark_decode(
        model, latents[0].detach(), dataset, device
    )
    instrumentation = emit_instrumentation(run_dir, 1, {
        "train_step_time_s": train_step_time_s,
        "train_peak_gpu_mem_mb": train_peak_gpu_mem_mb,
        # Step counts keep the arms comparable even if I/O variance survives
        # staging: wall-clock can drift, optimizer steps cannot.
        "optimizer_steps_decoder": decoder_steps,
        "optimizer_steps_latent": latent_steps,
        "inference_wallclock_s": inference_wallclock_s,
        "inference_peak_gpu_mem_mb": inference_peak_gpu_mem_mb,
    })
    return {
        "epochs_completed": total_epochs,
        "best_train_rel_l2": best,
        "decoder_parameters": decoder_parameters,
        "latent_dimension": int(architecture["latent_dim"]),
        "latent_table_values": latents.numel(),
        "latent_bottleneck_scalars_per_snapshot": int(architecture["latent_dim"]),
        "optimizer_steps_decoder": decoder_steps,
        "optimizer_steps_latent": latent_steps,
        "instrumentation": instrumentation,
    }


def _import_upstream_diffusion(root: str | Path):
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from UnconditionalDiffusionTraining_and_Generation.src.script_util import create_gaussian_diffusion, create_model
    return create_model, create_gaussian_diffusion


def train_stage2(
    cfg: dict, device: torch.device, run_dir: Path,
    resume: dict | None, stage1_checkpoint: Path,
) -> dict:
    _require_cuda(device)
    stage = cfg["confild_params"]["stage2"]
    training, architecture = stage["training"], stage["architecture"]
    stage1 = torch.load(stage1_checkpoint, map_location="cpu")
    groups = int(stage1["groups"])
    latent_dimension = int(stage1["latents"].shape[-1])
    if int(architecture["model_image_size"]) != latent_dimension:
        raise ValueError(
            "CoNFiLD stage2 architecture.model_image_size must equal the "
            f"stage-1 latent dimension ({latent_dimension})."
        )
    channel_mult = tuple(int(value.strip()) for value in str(architecture["channel_mult"]).split(","))
    downsample_factor = 2 ** (len(channel_mult) - 1)
    if latent_dimension % downsample_factor or int(architecture["window_length"]) % downsample_factor:
        raise ValueError(
            "CoNFiLD latent and window dimensions must be divisible by the "
            f"UNet downsampling factor ({downsample_factor})."
        )
    windows, manifest = build_latent_windows(
        stage1["latents"].float(),
        n_snapshots=int(stage1["train_indices"].numel()),
        n_groups=groups,
        cube_length=int(architecture["cube_length"]),
        window_length=int(architecture["window_length"]),
        stride=int(architecture["window_stride"]),
    )
    latent_min, latent_max = windows.amin().reshape(1), windows.amax().reshape(1)
    normalized = ((windows - latent_min) / (latent_max - latent_min).clamp_min(1e-12) * 2.0 - 1.0).contiguous()
    (run_dir / "window_manifest.json").write_text(json.dumps({"windows": manifest}, indent=2) + "\n", encoding="utf-8")

    create_model, create_diffusion = _import_upstream_diffusion(cfg["confild_params"]["upstream_root"])
    model = create_model(
        image_size=int(architecture["model_image_size"]),
        num_channels=int(architecture["num_channels"]),
        num_res_blocks=int(architecture["num_res_blocks"]),
        num_heads=int(architecture["num_heads"]),
        num_head_channels=int(architecture["num_head_channels"]),
        attention_resolutions=str(architecture["attention_resolutions"]),
        channel_mult=str(architecture["channel_mult"]),
    ).to(device)
    diffusion_parameters = sum(parameter.numel() for parameter in model.parameters())
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    diffusion = create_diffusion(steps=int(architecture["diffusion_steps"]), noise_schedule="cosine")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=0.0)
    start_step = 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        ema_model.load_state_dict(resume["ema"])
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"]) + 1

    total_steps = int(training["steps"])
    wallclock_budget_s = float(training.get("wallclock_budget_s", 0.0))
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["shared"]["seed"]) + start_step)
    decoder_parameters = sum(value.numel() for value in stage1["decoder"].values())
    combined_parameters = decoder_parameters + diffusion_parameters
    budget = cfg["confild_params"].get("parameter_budget", {})
    if bool(budget.get("enforce", False)):
        tolerance = float(budget["relative_tolerance"])
        for baseline, reference in budget.items():
            if baseline in {"enforce", "relative_tolerance"}:
                continue
            relative_difference = abs(combined_parameters - int(reference)) / int(reference)
            if relative_difference > tolerance:
                raise ValueError(
                    f"CoNFiLD combined parameter count {combined_parameters:,} is "
                    f"{relative_difference:.1%} from {baseline} ({int(reference):,}), "
                    f"outside the configured {tolerance:.1%} tolerance."
                )
    print(
        f"[confild:stage2] device={device} windows={tuple(normalized.shape)} "
        f"diffusion_parameters={diffusion_parameters:,} "
        f"combined_inference_parameters={combined_parameters:,}",
        flush=True,
    )
    # Frozen stage-1 codec: needed for the periodic figures and reused for the
    # end-of-run inference benchmark.
    Decoder = import_upstream_decoder(cfg["confild_params"]["upstream_root"])
    stage1_architecture = stage1["architecture"]
    statistics = FieldStatistics.from_state_dict(stage1["field_statistics"]).to(device)
    decoder = Decoder(
        in_coord_features=3, in_latent_features=latent_dimension,
        out_features=int(statistics.benchmark_mean.numel()),
        num_hidden_layers=int(stage1_architecture["layers"]),
        hidden_features=int(stage1_architecture["hidden_features"]),
    ).to(device)
    decoder.load_state_dict(stage1["decoder"])
    decoder.eval()
    for q in decoder.parameters():
        q.requires_grad_(False)
    bench_dataset = _packed_dataset(cfg)
    bench_coords = bench_dataset.coords.to(device)

    started = time.time()
    train_step_time_s = 0.0
    train_peak_gpu_mem_mb = 0.0
    torch.cuda.reset_peak_memory_stats(device)

    figures_dir = run_dir / "Evaluation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    field_names = list(cfg["shared"]["data"]["field_names"])
    vis = training.get("visualisation", {}) or {}
    scheduler = FigureScheduler(int(vis.get("n_figures", 24)), wallclock_budget_s,
                                total_steps, started)
    from methods.common.loss_plot import LossTracker
    tracker = LossTracker(run_dir, name="confild_stage2")
    real_latent = stage1["latents"][0].float().to(device)
    real_truth = statistics.normalize(
        bench_dataset.read_snapshot(int(stage1["train_indices"][0])).to(device))
    print(f"[confild:stage2] figures -> {figures_dir} (~{scheduler.n} per stage)",
          flush=True)

    for step in range(start_step, total_steps):
        step_started = time.time()
        ids = torch.randint(normalized.shape[0], (int(training["batch_size"]),), generator=generator)
        batch = normalized[ids].to(device)
        timesteps = torch.randint(int(architecture["diffusion_steps"]), (batch.shape[0],), generator=generator).to(device)
        loss = diffusion.training_losses(model, batch, timesteps)["loss"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                ema_parameter.mul_(float(architecture["ema"])).add_(parameter, alpha=1.0 - float(architecture["ema"]))
        torch.cuda.synchronize(device)
        # Accumulated before any logging or checkpointing: training only.
        train_step_time_s += time.time() - step_started
        train_peak_gpu_mem_mb = max(
            train_peak_gpu_mem_mb, torch.cuda.max_memory_allocated(device) / 2**20
        )
        if step % int(training["log_every"]) == 0:
            record = {"step": step, "loss": float(loss.detach()), "elapsed_seconds": time.time() - started}
            _append_jsonl(run_dir / "history.jsonl", record)
            tracker.log(step=step, diffusion_loss=record["loss"])
            tracker.plot()
            print(f"[confild:stage2] step={step} loss={record['loss']:.6g}", flush=True)
        if scheduler.due(step + 1):
            figure_started = time.time()
            with torch.no_grad():
                sampled = diffusion.p_sample_loop(
                    ema_model, (1, 1, int(architecture["window_length"]), latent_dimension),
                    device=device, progress=False)
            drawn = ((sampled[0, 0, 0] + 1.0) * 0.5
                     * (latent_max - latent_min).to(device) + latent_min.to(device))
            # Left: a REAL stage-1 latent decoded (codec sanity, unchanged by
            # stage 2). Right: a latent DRAWN FROM THE PRIOR decoded -- if the
            # prior has not learned the latent manifold this is visibly not
            # turbulence, which no scalar diffusion loss will tell you.
            save_field_figure(
                figures_dir / f"stage2_codec_step{step:07d}.png",
                bench_dataset.grid_shape, field_names,
                f"CoNFiLD stage 2 - REAL stage-1 latent through the frozen codec   step {step}",
                truth=real_truth.cpu().numpy(),
                prediction=decode_field(decoder, real_latent, bench_coords,
                                        bench_dataset.num_points, device).cpu().numpy())
            save_field_figure(
                figures_dir / f"stage2_prior_sample_step{step:07d}.png",
                bench_dataset.grid_shape, field_names,
                f"CoNFiLD stage 2 - UNCONDITIONAL prior sample (EMA, "
                f"{int(architecture['diffusion_steps'])} steps)   step {step}",
                prediction=decode_field(decoder, drawn, bench_coords,
                                        bench_dataset.num_points, device).cpu().numpy())
            plt = _pyplot()
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            ax.hist(stage1["latents"].float().flatten().numpy(), bins=120, density=True,
                    alpha=0.55, label="stage-1 latent table")
            ax.hist(drawn.detach().cpu().numpy().ravel(), bins=120, density=True,
                    alpha=0.55, label="prior sample")
            ax.legend(fontsize=8); ax.set_title(f"latent marginals, step {step}", fontsize=10)
            fig.tight_layout()
            fig.savefig(figures_dir / f"stage2_latent_marginal_step{step:07d}.png",
                        dpi=105, bbox_inches="tight")
            plt.close(fig)
            print(f"[confild:figure] stage=2 step={step} "
                  f"sample_rms={float(drawn.square().mean().sqrt()):.4f} "
                  f"seconds={time.time() - figure_started:.1f}", flush=True)
        if (step + 1) % int(training["save_every"]) == 0 or step == total_steps - 1:
            payload = {
                "format_version": 2, "baseline_model": "confild", "training_stage": 2,
                "epoch": step, "step": step, "model": model.state_dict(),
                "ema": ema_model.state_dict(), "optimizer": optimizer.state_dict(),
                "latent_min": latent_min, "latent_max": latent_max,
                "stage1_checkpoint": str(stage1_checkpoint.resolve()),
                "architecture": copy.deepcopy(architecture), "config": copy.deepcopy(cfg),
                "method": "CoNFiLD_upstream_latent_image_diffusion",
            }
            torch.save(payload, run_dir / "last.pt")
            torch.save(payload, run_dir / "best.pt")
        if wallclock_budget_s and (time.time() - started) >= wallclock_budget_s:
            payload = {
                "format_version": 2, "baseline_model": "confild", "training_stage": 2,
                "epoch": step, "step": step, "model": model.state_dict(),
                "ema": ema_model.state_dict(), "optimizer": optimizer.state_dict(),
                "latent_min": latent_min, "latent_max": latent_max,
                "stage1_checkpoint": str(stage1_checkpoint.resolve()),
                "architecture": copy.deepcopy(architecture), "config": copy.deepcopy(cfg),
                "method": "CoNFiLD_upstream_latent_image_diffusion",
            }
            torch.save(payload, run_dir / "last.pt")
            torch.save(payload, run_dir / "best.pt")
            total_steps = step + 1
            print(f"[confild:stage2] wall-clock budget reached at step {step}", flush=True)
            break

    # End-to-end inference cost for ONE full 125^3 x 4 field: a complete
    # 1000-step DDPM sample of the latent code plus the SIREN decode.
    tracker.plot()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    sample_started = time.time()
    with torch.no_grad():
        sampled = diffusion.p_sample_loop(
            ema_model, (1, 1, int(architecture["window_length"]), latent_dimension),
            device=device, progress=False,
        )
    torch.cuda.synchronize(device)
    sample_seconds = time.time() - sample_started
    latent = ((sampled[0, 0, 0] + 1.0) * 0.5 * (latent_max - latent_min).to(device)
              + latent_min.to(device))
    decode_started = time.time()
    _decode_full_field(decoder, latent, bench_dataset, device)
    torch.cuda.synchronize(device)
    decode_seconds = time.time() - decode_started
    inference_peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 2**20
    print(
        f"[confild:instrument] stage=2 diffusion_sample_s={sample_seconds:.6g} "
        f"decode_s={decode_seconds:.6g} window_length={int(architecture['window_length'])}",
        flush=True,
    )
    instrumentation = emit_instrumentation(run_dir, 2, {
        "train_step_time_s": train_step_time_s,
        "train_peak_gpu_mem_mb": train_peak_gpu_mem_mb,
        "optimizer_steps_diffusion": float(total_steps - start_step),
        # One p_sample_loop produces window_length latents at once; charge a
        # single field its share of the sample plus its own full decode.
        "inference_wallclock_s": sample_seconds / max(1, int(architecture["window_length"])) + decode_seconds,
        "inference_diffusion_sample_s": sample_seconds,
        "inference_decode_s": decode_seconds,
        "inference_peak_gpu_mem_mb": inference_peak_gpu_mem_mb,
    })
    return {
        "instrumentation": instrumentation,
        "latent_bottleneck_scalars_per_snapshot": latent_dimension,
        "steps_completed": total_steps,
        "stage1_checkpoint": str(stage1_checkpoint),
        "decoder_parameters": decoder_parameters,
        "diffusion_parameters": diffusion_parameters,
        "combined_inference_parameters": combined_parameters,
        "latent_dimension": latent_dimension,
    }


def run_confild_training(
    cfg: dict, device: torch.device, run_dir: Path,
    resume: dict | None = None, stage1_checkpoint: Path | None = None,
) -> dict:
    if int(cfg["training_stage"]) == 1:
        return train_stage1(cfg, device, run_dir, resume)
    if stage1_checkpoint is None:
        raise ValueError("CoNFiLD stage 2 requires a stage-1 checkpoint.")
    return train_stage2(cfg, device, run_dir, resume, stage1_checkpoint)
