"""Shared setup for the S3GM sampler-upgrade jobs (figures / tuning / final eval).

Loads the completed run's config + checkpoint through the same adapter path as
eval_s3gm3d.py, including the staged-data-path fallback. No repo files edited.
"""
from __future__ import annotations
from paths import REPO_ROOT, RUNS_DIR, SRC_DIR

import os
import sys
from pathlib import Path

RUN = Path(f"{RUNS_DIR}/JHU/baseline_s3gm/matched/Baseline_s3gm_Stage1_DemoN94_20260828_224632")

os.chdir(REPO_ROOT)

import torch  # noqa: E402
import methods.s3gm.s3gm3d as S
import methods.common.config as bconfig
import methods.common.data as bdata


def load_run(ckpt="best", device_str="cuda:0"):
    cfg = bconfig.validate_and_normalize_config(bconfig.load_yaml(RUN / "run_config.yaml"))

    _paths = cfg["shared"]["paths"]
    _dp = Path(_paths["data_path"])
    if not _dp.exists():
        _shared = _paths.get("data_path_shared")
        if _shared and Path(_shared).exists():
            print(f"[setup] staged data_path {_dp} absent; using shared {_shared}", flush=True)
            _paths["data_path"] = _shared
        else:
            raise FileNotFoundError(f"data_path {_dp} missing and no shared fallback")

    device = torch.device(device_str)
    print(f"[setup] device={device} "
          f"({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})",
          flush=True)

    stats_path = RUN / "dataset_stats.pt"
    val_set = bdata.build_dataset(cfg, split="val", stats_path=stats_path)
    train_set = bdata.build_dataset(cfg, split="train", stats_path=stats_path)
    print(f"[setup] val snapshots={len(val_set)} points={val_set.num_points} "
          f"fields={list(val_set.field_names)}", flush=True)

    adapter = S.S3GM3DAdapter()
    bundle = adapter.build_for_training(cfg, device, RUN, train_set, val_set)
    ck_path = Path(ckpt) if os.path.sep in ckpt else RUN / f"{ckpt}.pt"
    ck = bconfig.safe_torch_load(ck_path, map_location="cpu")
    adapter.load_checkpoint(bundle, ck)
    print(f"[setup] checkpoint={ck_path} epoch={ck.get('epoch')} "
          f"val_loss={ck.get('val_loss')}", flush=True)
    return cfg, adapter, bundle, val_set, ck
