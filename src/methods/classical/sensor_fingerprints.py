"""Dump the sensor-layout fingerprint for every (snapshot, sensor-count) used by
the classical baselines, straight from ``ensemble_eval``'s own code path.

This exists so bit-identity of the observation operator can be audited without
re-running any model.  It reproduces, verbatim, what ``ensemble_eval.main``
does per snapshot:

    torch.manual_seed(args.seed * 777 + int(snap))
    build_sparse_condition(coords[1,N,3], fields[1,N,C], cond_fields=[0,2],
                           n_obs_min=[n,n], n_obs_max=[n,n])

using the CANONICAL ``helpers.build_sparse_condition`` (whose ``torch.randint``
draws from the CPU generator), with the tensors on cuda:0 exactly as
``ensemble_eval`` places them.  ``helpers_baseline.build_sparse_condition``
draws that randint on the CUDA generator instead and therefore desynchronises
the stream -- it must not be used here.

Fingerprint = int(obs_indices[obs_mask > 0].sum()).

Usage: python sensor_fingerprints.py --out <dir>/sensor_fingerprints.json
"""
from __future__ import annotations
from paths import DATA_DIR, RUNS_DIR

import argparse
import json
import os
from pathlib import Path

import torch

os.environ.setdefault("JHU_SPLIT_MODE", "block")
os.environ.setdefault("JHU_SPLIT_GAP", "0")

from methods.dmfgen.helpers import TurbulentCombustionH5Dataset, build_sparse_condition

DATA = (f"{DATA_DIR}/jhu_homogeneous_turbulence/"
        "outputfiles_diverse/JHU_4cubes_stride100.h5")
STATS = (f"{RUNS_DIR}/JHU/pointcloud_ffm/"
         "iclr_jhu_xcube_spec02_DemoN29_20260822_140100/dataset_stats.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-obs", type=int, nargs="+",
                   default=[1953, 4883, 9766, 19531, 39062, 97656, 195312])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ds = TurbulentCombustionH5Dataset(DATA, split="val", train_ratio=0.75,
                                      field_names=("Ux", "Uy", "Uz", "p"),
                                      seed=42, time_stride=1, stats_path=STATS)
    item0 = ds[0]
    coords = item0["coords"].unsqueeze(0).to(a.device)
    fields = item0["fields"].unsqueeze(0).to(a.device)

    out = {"env": {"torch": torch.__version__,
                   "gpu": (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None),
                   "device": a.device, "seed": a.seed,
                   "build_sparse_condition": "helpers.py (canonical, CPU randint)"},
           "fingerprints": {}}
    for n in a.n_obs:
        rows = []
        for snap in range(len(ds)):
            torch.manual_seed(a.seed * 777 + int(snap))
            _, _, om, oi, _ = build_sparse_condition(
                coords_full=coords, fields_full=fields, cond_fields=[0, 2],
                n_obs_min=[n, n], n_obs_max=[n, n])
            s = int(oi[0][om[0] > 0].sum().item())
            k = int(om.sum().item())
            rows.append({"snapshot": snap, "sensors": k, "idx_sum": s})
            print(f"[seedcheck] n_obs={n} snap={snap} sensors={k} idx_sum={s}",
                  flush=True)
        out["fingerprints"][str(n)] = rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[out] wrote {a.out}")


if __name__ == "__main__":
    main()
