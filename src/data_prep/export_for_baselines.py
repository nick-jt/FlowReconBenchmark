"""Export our datasets into the layouts external baseline repos expect.

Running a published baseline on OUR data (rather than reimplementing it) keeps
us out of the position of tuning a competitor's model, which is how weak-baseline
comparisons happen. Each repo wants a different on-disk layout, so the
conversion lives here rather than being patched into their code.

Gen4Turbulence (3_flow_reconstruction) expects, relative to its task dir:
    data/u.npy      float32 [n_fields, n_time, nx, ny, nz]
    data/MIN_u.npy  float32 [n_fields]     per-field min over the train split
    data/MAX_u.npy  float32 [n_fields]     per-field max over the train split
Its loader crops [:, :, :256, :256], a no-op for our 125^3 and 152x126x192 cases.

Usage:
    python export_for_baselines.py --target gen4turb --dataset jhu \
        --out /projects/.../baselines/Gen4Turbulence/3_flow_reconstruction/data
"""
from paths import DATA_DIR

import argparse
from pathlib import Path

import h5py
import numpy as np

JHU = (f"{DATA_DIR}/JHU_4cubes_stride100.h5")
FIREBENCH = (f"{DATA_DIR}/FireBench_u10u12_merged.h5")

GRIDS = {"jhu": (125, 125, 125), "firebench": (152, 126, 192)}
PATHS = {"jhu": JHU, "firebench": FIREBENCH}


def load_fields(dataset: str, n_time: int | None):
    """Return [n_time, n_points, n_fields] float32 plus the grid shape."""
    path = PATHS[dataset]
    with h5py.File(path, "r") as f:
        dset = f["fields"]                      # [1, n_t, n_p, 1, 1, n_f]
        n_t = dset.shape[1] if n_time is None else min(n_time, dset.shape[1])
        out = np.empty((n_t, dset.shape[2], dset.shape[5]), dtype=np.float32)
        for t in range(n_t):
            out[t] = dset[0, t, :, 0, 0, :]
    return out, GRIDS[dataset]


def export_gen4turb(dataset: str, out_dir: Path, n_time: int | None,
                    train_frac: float, crop: int | None = None) -> None:
    fields, grid = load_fields(dataset, n_time)
    n_t, n_p, n_f = fields.shape
    if int(np.prod(grid)) != n_p:
        raise ValueError(f"grid {grid} does not match {n_p} points")

    # [n_t, n_p, n_f] -> [n_f, n_t, nx, ny, nz]
    u = fields.reshape(n_t, *grid, n_f).transpose(4, 0, 1, 2, 3)
    if crop is not None:
        # Gen4Turbulence's 3D UNet asserts every spatial dim is divisible by 8,
        # so a 125^3 cube cannot be fed to it. We centre-crop instead of padding
        # (these cubes are not periodic, so padding would fabricate structure).
        # Evaluate OUR model on the identical crop so the comparison is matched.
        if any(crop > g for g in grid):
            raise ValueError(f"crop {crop} exceeds grid {grid}")
        o = [(g - crop) // 2 for g in grid]
        u = u[:, :, o[0]:o[0] + crop, o[1]:o[1] + crop, o[2]:o[2] + crop]
        print(f"centre-cropped {grid} -> ({crop}, {crop}, {crop})")
    u = np.ascontiguousarray(u, dtype=np.float32)

    # Normalization stats from the TRAIN portion only, so the held-out frames
    # never inform the scaling the baseline sees.
    n_train = max(1, int(round(train_frac * n_t)))
    train = u[:, :n_train]
    u_min = train.reshape(n_f, -1).min(axis=1).astype(np.float32)
    u_max = train.reshape(n_f, -1).max(axis=1).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "u.npy", u)
    np.save(out_dir / "MIN_u.npy", u_min)
    np.save(out_dir / "MAX_u.npy", u_max)
    print(f"wrote {out_dir/'u.npy'}  shape={u.shape} dtype={u.dtype} "
          f"({u.nbytes/1e9:.2f} GB)")
    print(f"  MIN_u={np.round(u_min, 4)}")
    print(f"  MAX_u={np.round(u_max, 4)}  (stats over first {n_train}/{n_t} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=["gen4turb"])
    p.add_argument("--dataset", required=True, choices=["jhu", "firebench"])
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--n-time", type=int, default=None,
                   help="Limit exported snapshots (default: all).")
    p.add_argument("--crop", type=int, default=None,
                   help="Centre-crop each spatial dim (Gen4Turb needs /8).")
    p.add_argument("--train-frac", type=float, default=0.75,
                   help="Fraction of frames treated as train for min/max stats.")
    args = p.parse_args()

    if args.target == "gen4turb":
        export_gen4turb(args.dataset, Path(args.out), args.n_time,
                        args.train_frac, args.crop)


if __name__ == "__main__":
    main()
