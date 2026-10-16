"""Route-2 conformalized recalibration from per-point ensemble dumps.

Consumes the .npz files written by dump_calib_points.py.  Fits, on the TUNE
split (odd snapshot indices), a conformal quantile of the studentized
residual |y - mean| / std per distance-to-nearest-sensor bin -- distance being
the spread profile's sufficient statistic -- and evaluates coverage frozen on
the TEST split (even indices), before and after, at several nominal levels.

The per-bin quantile with the finite-sample split-conformal correction
ceil((n+1)*(1-alpha))/n gives marginal coverage >= 1-alpha on exchangeable
points within a bin.  Honest-reporting notes baked into the output: the
repair is post hoc; it rescales stated intervals and cannot sharpen the
spread floor; apply the identical script to baseline dumps (latent FM) for a
two-sided comparison.

Login-node rule: this loads O(GB) of npz -- run it under sbatch/srun on a CPU
node, not on a login node.

Usage:
    python conformal_recalib.py --dump-dir <dir from dump_calib_points.py> \
        [--bins 8] [--levels 0.5 0.7 0.8 0.9 0.95] [--out conformal.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List

import numpy as np

EPS = 1e-8


def _load(dump_dir: str) -> List[dict]:
    files = sorted(glob.glob(os.path.join(dump_dir, "calib_points_snap*.npz")))
    if not files:
        raise SystemExit(f"no calib_points_snap*.npz in {dump_dir}")
    return [dict(np.load(f)) for f in files]


def _stats(d: dict):
    ens = d["ens"].astype(np.float32)            # [K, N, C]
    mean, std = ens.mean(0), ens.std(0, ddof=1)  # [N, C]
    truth = d["truth"].astype(np.float32)
    r = np.abs(truth - mean) / (std + EPS)       # studentized residual
    return mean, std, truth, r, d["dist"].astype(np.float32), ens


def _ens_cov(ens: np.ndarray, truth: np.ndarray, level: float) -> np.ndarray:
    """Empirical central-interval coverage mask from the raw ensemble."""
    lo = np.quantile(ens, (1 - level) / 2, axis=0)
    hi = np.quantile(ens, 1 - (1 - level) / 2, axis=0)
    return (truth >= lo) & (truth <= hi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-dir", required=True)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--levels", type=float, nargs="+",
                   default=[0.5, 0.7, 0.8, 0.9, 0.95])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    meta_path = os.path.join(args.dump_dir, "dump_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    field_names = meta.get("field_names") or []

    dumps = _load(args.dump_dir)
    tune = [d for d in dumps if int(d["snap"]) % 2 == 1]
    test = [d for d in dumps if int(d["snap"]) % 2 == 0]
    if not tune or not test:
        raise SystemExit(f"need both parities: tune={len(tune)} test={len(test)}")
    print(f"[conformal] tune={len(tune)} test={len(test)} snapshots")

    # ---- fit: distance bin edges + per-bin conformal quantiles on TUNE ----
    tune_dist = np.concatenate([d["dist"] for d in tune])
    edges = np.quantile(tune_dist, np.linspace(0, 1, args.bins + 1))
    # At high sensor density the distance distribution concentrates, so
    # consecutive quantile edges can coincide and produce an EMPTY bin -- which
    # made the split-conformal rank divide by zero. Collapsing duplicate edges
    # keeps every bin non-degenerate; the effective bin count is reported.
    edges = np.unique(edges)
    n_bins = len(edges) - 1
    if n_bins < args.bins:
        print(f"[conformal] distance distribution is degenerate at this density: "
              f"{args.bins} requested bins collapse to {n_bins} distinct ones")
    edges[0], edges[-1] = -np.inf, np.inf

    n_ch = _stats(tune[0])[0].shape[1]
    q = np.zeros((n_bins, n_ch, len(args.levels)))
    pooled = np.concatenate([_stats(d)[3] for d in tune])      # fallback residuals
    for b in range(n_bins):
        rs: List[np.ndarray] = []
        for d in tune:
            _, _, _, r, dist, _ = _stats(d)
            m = (dist >= edges[b]) & (dist < edges[b + 1])
            rs.append(r[m])
        rb = np.concatenate(rs)                          # [n_b, C]
        n = rb.shape[0]
        if n == 0:
            # No tuning point landed in this bin. Fall back to the pooled
            # quantile rather than emitting a zero threshold, which would have
            # claimed 0-width intervals and destroyed coverage for any test
            # point that does land here.
            rb, n = pooled, pooled.shape[0]
            print(f"[conformal] bin {b} empty on TUNE; using the pooled quantile")
        for li, lv in enumerate(args.levels):
            rank = min(math.ceil((n + 1) * lv) / n, 1.0)  # split-conformal
            q[b, :, li] = np.quantile(rb, rank, axis=0)

    # ---- evaluate frozen on TEST ----
    cov_before = np.zeros((n_ch, len(args.levels)))
    cov_after = np.zeros((n_ch, len(args.levels)))
    npts = 0
    sp_before, sp_after, err = np.zeros(n_ch), np.zeros(n_ch), np.zeros(n_ch)
    for d in test:
        mean, std, truth, r, dist, ens = _stats(d)
        b_idx = np.clip(np.searchsorted(edges, dist, side="right") - 1,
                        0, n_bins - 1)   # q has n_bins rows after edge collapse
        for li, lv in enumerate(args.levels):
            cov_before[:, li] += _ens_cov(ens, truth, lv).sum(0)
            half = q[b_idx, :, li] * std                 # [N, C]
            cov_after[:, li] += ((truth >= mean - half)
                                 & (truth <= mean + half)).sum(0)
        npts += truth.shape[0]
        err += ((truth - mean) ** 2).sum(0)
        sp_before += (std ** 2).sum(0)
        # implied post-repair std at the level closest to 0.9 (Gaussian scale)
        li90 = int(np.argmin([abs(l - 0.9) for l in args.levels]))
        sp_after += (((q[b_idx, :, li90] / 1.645) * std) ** 2).sum(0)
    cov_before /= npts
    cov_after /= npts
    sp_err_before = np.sqrt(sp_before / npts) / np.sqrt(err / npts)
    sp_err_after = np.sqrt(sp_after / npts) / np.sqrt(err / npts)

    names = field_names if len(field_names) == n_ch else [f"ch{i}" for i in range(n_ch)]
    # Source identity lives in the payload: the density tag in the filename is
    # the same for every model, so the filename alone cannot say whose dumps
    # these were (a misnamed latent-FM result once overwrote DMF-Gen's).
    result: Dict = {"dump_dir": os.path.realpath(args.dump_dir),
                    "levels": args.levels, "bins": n_bins, "bins_requested": args.bins,
                    "bin_edges": [float(e) for e in edges[1:-1]],
                    "split": "tune=odd/test=even",
                    "n_test_points": int(npts), "channels": {}}
    print(f"\n{'ch':>6} {'level':>6} {'cov before':>10} {'cov after':>10}")
    for c, name in enumerate(names):
        result["channels"][name] = {
            "cov_before": [float(v) for v in cov_before[c]],
            "cov_after": [float(v) for v in cov_after[c]],
            "sp_err_before": float(sp_err_before[c]),
            "sp_err_after_gauss90": float(sp_err_after[c]),
        }
        for li, lv in enumerate(args.levels):
            print(f"{name:>6} {lv:6.2f} {cov_before[c, li]:10.3f} "
                  f"{cov_after[c, li]:10.3f}")

    print("\nNOTE: post-hoc repair; the spread floor is untouched. Run the "
          "same script on the latent-FM dumps for the two-sided comparison.")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
