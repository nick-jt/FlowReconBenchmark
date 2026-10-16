# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from methods.common.helpers_baseline import FIELD_NAMES
from methods.common.helpers_baseline import TurbulentCombustionH5Dataset
from methods.common.helpers_baseline import collate_snapshots
from methods.common.config import ensure_absolute


def build_dataset(cfg: dict, split: str, stats_path: Path) -> TurbulentCombustionH5Dataset:
    # SHIFT-WING is an unstructured mesh with a surface observation pool, so it
    # cannot go through the gridded H5 path. Only coordinate/point-based
    # baselines can consume it; grid models must not be pointed at this branch.
    shared = cfg.get("shared", {}).get("data", {})
    if str(shared.get("dataset", "")).lower() == "shiftwing":
        if split not in ("train", "val"):
            raise NotImplementedError("SHIFT-WING has no TUNE/TEST halves (per-case split); "
                                      f"split={split!r} is not defined there.")
        from methods.common.dataset_wing_baseline import build_wing_dataset
        return build_wing_dataset(cfg, split)

    shared = cfg["shared"]
    data_path = ensure_absolute(shared["paths"]["data_path"])
    return TurbulentCombustionH5Dataset(
        str(data_path),
        split=split,
        train_ratio=float(shared["data"]["train_ratio"]),
        seed=int(shared["seed"]),
        time_stride=int(shared["data"]["time_stride"]),
        field_names=tuple(shared["data"]["field_names"]) if shared["data"].get("field_names") is not None else FIELD_NAMES,
        stats_path=str(stats_path),
        sensor_pool=shared["conditioning"].get("sensor_pool"),
    )


def build_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    # OPT-IN worker reuse. TurbulentCombustionH5Dataset does no caching, and with
    # the default (persistent_workers=False) PyTorch tears down and respawns the
    # worker processes every epoch. Measured on this dataset (profile_io_sit.py,
    # 150 snapshots, batch 24, 7 batches/epoch): a data-only sweep costs
    #   num_workers=4  batch1 ~4-5 s, epoch ~5.9 s
    #   num_workers=0  batch1 ~0.6 s, epoch ~3.7 s
    # i.e. ~4 s of EVERY epoch is pure process spawn, not I/O -- staging the H5
    # to node-local /tmp changed nothing (5.9 s Lustre vs 5.9 s tmpfs, and the
    # full 4.7 GB sweep runs at 1.27 GB/s). Off by default so no other
    # baseline's behaviour changes; set JHU_PERSISTENT_WORKERS=1 to enable.
    _persist = (os.environ.get("JHU_PERSISTENT_WORKERS", "") == "1") and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
        persistent_workers=_persist,
    )
