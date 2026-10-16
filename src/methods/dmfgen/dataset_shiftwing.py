"""SHIFT-WING dataset: preprocessing and PyTorch dataset.

Preprocessing (run as a script) converts each raw SHIFT-WING sample
(unstructured RANS volume ``merged_volumes.vtu`` + surface ``merged_surfaces.vtp``)
into a compact per-case HDF5 with a near-body crop and fixed-size point subsets:

    volume/coords  [N_VOL, 3]   float32, meters (raw)
    volume/fields  [N_VOL, 4]   float32, nondimensional [u/U, v/U, w/U, Cp]
    surface/coords [N_SURF, 3]  float32, meters (raw)
    surface/values [N_SURF, 4]  float32, nondimensional [Cp, taux/q, tauy/q, tauz/q]
    attrs: mach, alpha, reynolds, U_inf, q_inf, p_inf, Cd, Cl, crop box

The unstructured mesh is adaptively refined near the body and in shock/wake
regions, so uniform subsampling over mesh nodes inherits the solver's own
spatial importance distribution.

Field-id convention (n_obs_field_types = 9):
    0..3  volumetric Ux, Uy, Uz, Cp (generated channels; id 3 is also used
          for surface pressure observations, which sample the same field at
          wall locations)
    4..6  wall shear stress components (observable only)
    7     freestream Mach parameter token
    8     angle-of-attack parameter token
"""

from __future__ import annotations
from paths import DATA_DIR

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

N_VOL = 400_000
N_SURF = 65_536

FIELD_NAMES = ["Ux", "Uy", "Uz", "Cp"]
SURF_VALUE_FIELD_IDS = [3, 4, 5, 6]  # Cp, taux, tauy, tauz
PARAM_FIELD_IDS = [7, 8]             # mach, alpha
N_OBS_FIELD_TYPES = 9


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def _crop_box(surface_bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Near-body crop: generous wake extension in +x, moderate margins else."""
    b = surface_bounds.reshape(3, 2)
    ext = b[:, 1] - b[:, 0]
    lo = b[:, 0] - 0.35 * ext
    hi = b[:, 1] + 0.35 * ext
    lo[0] = b[0, 0] - 0.25 * ext[0]
    hi[0] = b[0, 1] + 1.00 * ext[0]
    return lo, hi


def preprocess_sample(sample_dir: str, out_path: str, seed: int = 0) -> Dict:
    import pyvista as pv

    sample_dir = Path(sample_dir)
    params = json.load(open(sample_dir / "params.json"))
    forces = json.load(open(sample_dir / "forces.json"))

    u_inf = float(params["stream_velocity"])
    rho = float(params["air_density"])
    p_inf = float(params["pressure"])
    q_inf = 0.5 * rho * u_inf ** 2

    surf = pv.read(str(sample_dir / "merged_surfaces.vtp"))
    lo, hi = _crop_box(np.asarray(surf.bounds))

    vol = pv.read(str(sample_dir / "merged_volumes.vtu"))
    pts = np.asarray(vol.points)
    in_box = ((pts >= lo) & (pts <= hi)).all(axis=1)
    idx_all = np.flatnonzero(in_box)

    rng = np.random.default_rng(seed)
    if len(idx_all) < N_VOL:
        raise RuntimeError(f"{sample_dir}: only {len(idx_all)} in-box points (< {N_VOL})")
    vol_idx = rng.choice(idx_all, size=N_VOL, replace=False)
    vol_idx.sort()

    U = np.asarray(vol.point_data["Velocity (m/s)"])[vol_idx] / u_inf
    P = (np.asarray(vol.point_data["Pressure (Pa)"])[vol_idx] - p_inf) / q_inf
    vol_fields = np.concatenate([U, P[:, None]], axis=1).astype(np.float32)
    vol_coords = pts[vol_idx].astype(np.float32)

    s_pts = np.asarray(surf.points)
    surf_idx = rng.choice(len(s_pts), size=min(N_SURF, len(s_pts)), replace=False)
    surf_idx.sort()
    Cp_s = (np.asarray(surf.point_data["Pressure (Pa)"])[surf_idx] - p_inf) / q_inf
    tau = np.asarray(surf.point_data["Wall Shear Stress (N/m²)"])[surf_idx] / q_inf
    surf_values = np.concatenate([Cp_s[:, None], tau], axis=1).astype(np.float32)
    surf_coords = s_pts[surf_idx].astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        g = f.create_group("volume")
        g.create_dataset("coords", data=vol_coords, compression="gzip", compression_opts=1)
        g.create_dataset("fields", data=vol_fields, compression="gzip", compression_opts=1)
        g = f.create_group("surface")
        g.create_dataset("coords", data=surf_coords, compression="gzip", compression_opts=1)
        g.create_dataset("values", data=surf_values, compression="gzip", compression_opts=1)
        f.attrs.update({
            "mach": float(params["mach"]),
            "alpha": float(params["alpha"]),
            "reynolds": float(params["reynolds_number"]),
            "u_inf": u_inf, "q_inf": q_inf, "p_inf": p_inf,
            "Cd": float(forces.get("Cd", np.nan)),
            "Cl": float(forces.get("Cl", np.nan)),
            "crop_lo": lo, "crop_hi": hi,
            "sample": sample_dir.name,
        })

    return {
        "sample": sample_dir.name,
        "crop_lo": lo.tolist(), "crop_hi": hi.tolist(),
        "mach": float(params["mach"]), "alpha": float(params["alpha"]),
    }


def _worker(job):
    sample_dir, out_path, seed = job
    try:
        return preprocess_sample(sample_dir, out_path, seed)
    except Exception as e:  # keep going; report at the end
        return {"sample": Path(sample_dir).name, "error": str(e)}


def preprocess_all(raw_root: str, out_root: str, n_workers: int = 8,
                   train_cases: int = 200) -> None:
    from multiprocessing import Pool

    raw_root = Path(raw_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(
        d for d in raw_root.iterdir()
        if d.is_dir() and (d / "merged_volumes.vtu").exists()
        and (d / "merged_surfaces.vtp").exists()
    )
    jobs = [
        (str(d), str(out_root / f"{d.name}.h5"), i)
        for i, d in enumerate(sample_dirs)
        if not (out_root / f"{d.name}.h5").exists()
    ]
    print(f"[shiftwing] {len(sample_dirs)} samples found, {len(jobs)} to process")

    records: List[Dict] = []
    with Pool(n_workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_worker, jobs)):
            records.append(rec)
            tag = "ERR " + rec["error"] if "error" in rec else "ok"
            print(f"  [{i + 1}/{len(jobs)}] {rec['sample']}: {tag}", flush=True)

    # Finalize: split, global bbox, normalization stats over train cases.
    files = sorted(out_root.glob("sample_*.h5"))
    # Never let the train block swallow every case: an empty val split silently
    # disables model selection (best.pt would freeze at epoch 1).
    max_train = max(1, int(round(0.9 * len(files))))
    if train_cases > max_train:
        print(f"[shiftwing] train_cases={train_cases} exceeds 90% of the "
              f"{len(files)} available cases; clamping to {max_train}")
        train_cases = max_train
    rng = np.random.default_rng(42)
    order = rng.permutation(len(files))
    train_ids = set(order[:train_cases].tolist())
    split = {
        "train": [files[i].name for i in range(len(files)) if i in train_ids],
        "val": [files[i].name for i in range(len(files)) if i not in train_ids],
    }

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    v_sum = np.zeros(4); v_sq = np.zeros(4); v_n = 0
    s_sum = np.zeros(4); s_sq = np.zeros(4); s_n = 0
    machs, alphas = [], []
    for name in split["train"]:
        with h5py.File(out_root / name, "r") as f:
            lo = np.minimum(lo, np.asarray(f.attrs["crop_lo"]))
            hi = np.maximum(hi, np.asarray(f.attrs["crop_hi"]))
            vf = f["volume/fields"][:]
            sv = f["surface/values"][:]
            v_sum += vf.sum(0); v_sq += (vf ** 2).sum(0); v_n += len(vf)
            s_sum += sv.sum(0); s_sq += (sv ** 2).sum(0); s_n += len(sv)
            machs.append(float(f.attrs["mach"])); alphas.append(float(f.attrs["alpha"]))

    v_mean = v_sum / v_n
    v_std = np.sqrt(np.maximum(v_sq / v_n - v_mean ** 2, 1e-12))
    s_mean = s_sum / s_n
    s_std = np.sqrt(np.maximum(s_sq / s_n - s_mean ** 2, 1e-12))

    stats = {
        "bbox_lo": lo.tolist(), "bbox_hi": hi.tolist(),
        "volume_mean": v_mean.tolist(), "volume_std": v_std.tolist(),
        "surface_mean": s_mean.tolist(), "surface_std": s_std.tolist(),
        "mach_mean": float(np.mean(machs)), "mach_std": float(np.std(machs) + 1e-8),
        "alpha_mean": float(np.mean(alphas)), "alpha_std": float(np.std(alphas) + 1e-8),
        "split": split,
        "field_names": FIELD_NAMES,
        "surf_value_field_ids": SURF_VALUE_FIELD_IDS,
        "param_field_ids": PARAM_FIELD_IDS,
        "n_obs_field_types": N_OBS_FIELD_TYPES,
    }
    with open(out_root / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[shiftwing] wrote stats.json: {len(split['train'])} train / "
          f"{len(split['val'])} val cases")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class ShiftWingDataset(Dataset):
    """Per-case wing point clouds with a surface observation pool.

    __getitem__ returns (all float32 tensors):
        coords            [N_VOL, 3]   normalized to [0, 1] by the global bbox
        fields            [N_VOL, 4]   z-scored nondimensional volume fields
        obs_pool_coords   [N_SURF, 3]  normalized surface coordinates
        obs_pool_values   [N_SURF, 4]  z-scored surface observables
        obs_pool_field_ids[4]          long, ids of the pool value channels
        param_coords      [2, 3]       token coordinates (domain center)
        param_values      [2, 1]       z-scored mach, alpha
        param_field_ids   [2]          long
    """

    def __init__(self, processed_root: str, split: str = "train"):
        self.root = Path(processed_root)
        stats = json.load(open(self.root / "stats.json"))
        self.stats = stats
        self.files = [self.root / n for n in stats["split"][split]]
        self.lo = np.asarray(stats["bbox_lo"], dtype=np.float32)
        self.hi = np.asarray(stats["bbox_hi"], dtype=np.float32)
        self.v_mean = np.asarray(stats["volume_mean"], dtype=np.float32)
        self.v_std = np.asarray(stats["volume_std"], dtype=np.float32)
        self.s_mean = np.asarray(stats["surface_mean"], dtype=np.float32)
        self.s_std = np.asarray(stats["surface_std"], dtype=np.float32)

        # Interface parity with TurbulentCombustionH5Dataset consumers.
        self.field_names = list(stats["field_names"])
        self.num_fields = len(self.field_names)
        self.mean = torch.from_numpy(self.v_mean.copy())
        self.std = torch.from_numpy(self.v_std.copy())
        self.n_obs_field_types = int(stats["n_obs_field_types"])

        self.augment_reflect = (
            os.environ.get("WING_AUGMENT", "") == "reflect_y" and split == "train"
        )
        self._rng = np.random.default_rng(1234)
        if self.augment_reflect:
            print("[shiftwing] spanwise reflection augmentation ON (train split)")

    def __len__(self) -> int:
        return len(self.files)

    def _norm_coords(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / (self.hi - self.lo)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        with h5py.File(self.files[i], "r") as f:
            vc = f["volume/coords"][:]
            vf = f["volume/fields"][:]
            sc = f["surface/coords"][:]
            sv = f["surface/values"][:]
            mach = float(f.attrs["mach"])
            alpha = float(f.attrs["alpha"])

        # Spanwise reflection about the symmetry plane y = 0. The airframe is
        # left-right symmetric and the cases carry no sideslip, so the mirrored
        # flow is an exact solution of the same problem. Applied in physical
        # coordinates (the geometry is symmetric about y=0, not about the
        # bounding-box centre) and before normalization.
        if self.augment_reflect and self._rng.random() < 0.5:
            vc = vc.copy(); sc = sc.copy(); vf = vf.copy(); sv = sv.copy()
            vc[:, 1] = -vc[:, 1]
            sc[:, 1] = -sc[:, 1]
            vf[:, 1] = -vf[:, 1]          # Uy
            sv[:, 2] = -sv[:, 2]          # tau_y

        coords = self._norm_coords(vc)
        fields = (vf - self.v_mean) / self.v_std
        pool_coords = self._norm_coords(sc)
        pool_values = (sv - self.s_mean) / self.s_std

        st = self.stats
        params = np.array(
            [(mach - st["mach_mean"]) / st["mach_std"],
             (alpha - st["alpha_mean"]) / st["alpha_std"]],
            dtype=np.float32,
        )
        param_coords = np.full((2, 3), 0.5, dtype=np.float32)

        return {
            "coords": torch.from_numpy(coords),
            "fields": torch.from_numpy(fields),
            "obs_pool_coords": torch.from_numpy(pool_coords),
            "obs_pool_values": torch.from_numpy(pool_values),
            "obs_pool_field_ids": torch.tensor(SURF_VALUE_FIELD_IDS, dtype=torch.long),
            "param_coords": torch.from_numpy(param_coords),
            "param_values": torch.from_numpy(params[:, None]),
            "param_field_ids": torch.tensor(PARAM_FIELD_IDS, dtype=torch.long),
        }


def collate_wing(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=str,
                   default=f"{DATA_DIR}/shift_wing/"
                           "data/OnShape_luminary_crm_version001")
    p.add_argument("--out-root", type=str,
                   default=f"{DATA_DIR}/shift_wing/"
                           "processed")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--train-cases", type=int, default=200)
    args = p.parse_args()
    preprocess_all(args.raw_root, args.out_root, args.n_workers, args.train_cases)
