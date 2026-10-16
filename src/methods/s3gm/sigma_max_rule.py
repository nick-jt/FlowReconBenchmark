"""Song & Ermon's rule for the VE-SDE noise ceiling: sigma_max = the largest pairwise
Euclidean distance between training samples, in the units the score network sees.

    python -m methods.s3gm.sigma_max_rule --config configs/jhu3d/config_baseline_S3GM_xcube.yaml
    python -m methods.s3gm.sigma_max_rule --config configs/kolmogorov2d/config_baseline_S3GM_kolm.yaml

2D: the samples are the z-scored training frames. JHU (s3gm3d): the samples are the
z-scored t_win-plane slabs the trainer draws (every z offset of every training snapshot);
a random subset of --max-samples slabs bounds the pairwise scan. The distance scales with
the sample dimension, which is why the 2D value (460 on Kolmogorov) and a 20 chosen by hand
for 625k-dimensional slabs differ by an order of magnitude.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from methods.common import config as C
from methods.common.data import build_dataset


def max_pairwise_distance(x: torch.Tensor, chunk: int = 256) -> float:
    """x [n, d] float32: max_ij ||x_i - x_j|| through Gram blocks (no n x n x d tensor)."""
    sq = (x.double() ** 2).sum(1)
    best = 0.0
    for i in range(0, x.shape[0], chunk):
        g = x[i:i + chunk].double() @ x.double().T
        d2 = sq[i:i + chunk, None] + sq[None, :] - 2 * g
        best = max(best, float(d2.clamp_min(0).max().sqrt()))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = C.validate_and_normalize_config(C.load_yaml(C.ensure_absolute(a.config)))
    ds = build_dataset(cfg, split="train", stats_path=C.ensure_absolute(cfg["shared"]["paths"]["data_path"]).with_suffix(".stats.pt"))
    arch = cfg["s3gm_params"]["architecture"]
    rng = np.random.default_rng(a.seed)
    num_z = cfg["shared"]["data"].get("num_z")
    if num_z is None:
        idx = np.sort(rng.choice(len(ds), size=min(a.max_samples, len(ds)), replace=False))
        x = torch.stack([ds[int(i)]["fields"].reshape(-1) for i in idx])
        what = f"{len(idx)} of {len(ds)} z-scored training frames, {x.shape[1]} values each"
    else:
        t_win = int(arch.get("t_win", 10))
        nz, ny, nx = int(num_z), int(cfg["shared"]["data"]["num_y"]), int(cfg["shared"]["data"]["num_x"])
        n_off = nz - t_win + 1
        pairs = [(s, o) for s in range(len(ds)) for o in range(n_off)]
        sel = rng.choice(len(pairs), size=min(a.max_samples, len(pairs)), replace=False)
        slabs, cache = [], {}
        for k in np.sort(sel):
            s, o = pairs[k]
            if s not in cache:
                cache = {s: ds[s]["fields"].reshape(nz, ny, nx, -1)}   # one snapshot at a time
            slabs.append(cache[s][o:o + t_win].reshape(-1))
        x = torch.stack(slabs)
        what = (f"{len(sel)} of {len(pairs)} z-scored {t_win}-plane slabs "
                f"({len(ds)} training snapshots x {n_off} z offsets), {x.shape[1]} values each")
    d = max_pairwise_distance(x)
    print(f"[sigma_max_rule] {a.config}: {what}")
    print(f"[sigma_max_rule] max pairwise distance = {d:.1f}  (configured sigma_max "
          f"{cfg['s3gm_params']['diffusion']['sigma_max']})", flush=True)


if __name__ == "__main__":
    main()
