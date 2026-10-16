"""Benchmark protocol rules shared by every trainer and evaluator.

Held-out frames are split in two by index parity: odd indices form the TUNE
half, even indices the TEST half. Checkpoint selection, sampler settings and
every other choice are made on TUNE; every reported number is scored on TEST.
The validation pass that selects a checkpoint draws sensors with a fixed seed
at the evaluation count, so `best.pt` is the optimum of one fixed measurement
and not a noise pick over random sensor counts.
"""
from __future__ import annotations

import contextlib
import pathlib
from typing import Iterator

import numpy as np
import torch

PROTOCOL_2D = "kolm2d_matched_v2"      # TEST split, fixed-seed TUNE selection, no sensor overwrite
VAL_FIXED_SEED = 0                     # RNG seed of the selection sensor draw
PARAM_BUDGET = 6_500_000               # learned rows: 6.5 M +- PARAM_TOL
PARAM_TOL = 0.10
HELD_OUT_PARITY = {"val": None, "tune": 1, "test": 0}


def held_out_subset(indices: np.ndarray, split: str) -> np.ndarray:
    """`indices` are the sorted held-out frame indices; return the TUNE (odd
    positions), TEST (even positions) or full ("val") subset."""
    parity = HELD_OUT_PARITY[split]
    if parity is None:
        return indices
    return indices[parity::2]


def frame_list(n_total: int, n_frames: int, blocks: int = 1) -> list[int]:
    """The evaluated frames of a split: evenly spaced; with blocks > 1, spaced evenly
    inside each of `blocks` equal sub-blocks (stratified across held-out trajectories)."""
    if blocks <= 1:
        return [(i * n_total) // n_frames for i in range(n_frames)]
    if n_total % blocks or n_frames % blocks:
        raise SystemExit(f"[guard] stratify: {n_total} frames / {n_frames} "
                         f"picks not divisible by {blocks} blocks.")
    span, per = n_total // blocks, n_frames // blocks
    return [b * span + (i * span) // per
            for b in range(blocks) for i in range(per)]


@contextlib.contextmanager
def fixed_validation_draw(conditioning: dict, seed: int = VAL_FIXED_SEED) -> Iterator[list[int]]:
    """Make the enclosed validation pass draw `vis_n_obs_list` sensors per
    conditioned field under a fixed RNG seed. The conditioning dict's count
    range and the global RNG state are restored afterwards, so training noise
    is unaffected. Yields the per-field count used."""
    n_obs = [int(v) for v in conditioning["vis_n_obs_list"]]
    nf = len(conditioning["cond_fields"])
    if len(n_obs) == 1 and nf > 1:
        n_obs = n_obs * nf
    saved = (conditioning["n_obs_min_list"], conditioning["n_obs_max_list"])
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        conditioning["n_obs_min_list"] = list(n_obs)
        conditioning["n_obs_max_list"] = list(n_obs)
        torch.manual_seed(seed)
        yield n_obs
    finally:
        conditioning["n_obs_min_list"], conditioning["n_obs_max_list"] = saved
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def resume_best_val(reload_ckpt: dict, run_dir) -> float:
    """Best validation loss of a resumed run: the value persisted in the checkpoint,
    else the loss recorded in best.pt. Never last.pt's own val loss, which would make
    best.pt the best since the last resume."""
    if "best_val" in reload_ckpt:
        return float(reload_ckpt["best_val"])
    best = pathlib.Path(run_dir) / "best.pt"
    if best.exists():
        return float(torch.load(best, map_location="cpu", weights_only=False).get("val_loss", float("inf")))
    return float("inf")


def count_parameters(model: torch.nn.Module) -> dict:
    """Parameter count by the benchmark convention: `numel` over every
    parameter tensor, a complex element counting once (not as two reals)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_complex = sum(p.numel() for p in model.parameters() if p.is_complex())
    return {"total": int(total), "trainable": int(trainable), "complex_elements": int(n_complex),
            "convention": "numel; complex elements counted once"}


def within_budget(total: int, budget: int = PARAM_BUDGET, tol: float = PARAM_TOL) -> bool:
    return abs(total - budget) <= tol * budget


def log_param_count(model: torch.nn.Module, label: str) -> dict:
    c = count_parameters(model)
    flag = "" if within_budget(c["total"]) else f"  OUTSIDE budget {PARAM_BUDGET / 1e6:.1f}M +- {PARAM_TOL:.0%}"
    print(f"[params] {label}: total={c['total']:,} trainable={c['trainable']:,} "
          f"({c['convention']}){flag}", flush=True)
    return c
