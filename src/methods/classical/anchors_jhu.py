"""Classical anchor baselines for the JHU cross-cube reconstruction protocol.

Three CPU-only, training-free (or SVD-only) estimators that a reviewer will ask
for as the floor of the table:

  1. ``kdtree``   nearest-neighbour interpolation from the sensors
                  (``scipy.spatial.cKDTree``, periodic box).
  2. ``idw``      inverse-distance-weighted interpolation over k neighbours.
  3. ``gappy_pod`` Everson & Sirovich (1995) gappy POD: a POD basis built from
                  the TRAINING cubes only, modal coefficients recovered by
                  least squares on the sparse sensor readings, all four
                  channels reconstructed from the basis.
  4. ``constant`` the trivial floor: predict the TRAIN-split per-channel mean
                  everywhere.
  5. ``lse``      linear stochastic estimation: ridge regression of all four
                  channels on the readings over the TRAINING snapshots
                  (``anchors_2d.lse_reconstruct``), penalty chosen on train cube 2.

Unobserved channels
-------------------
Only Ux (0) and Uz (2) carry sensors; Uy (1) and p (3) are never observed.  A
nearest-neighbour or IDW interpolator has literally nothing to interpolate for
those two channels, so it predicts the train-split mean there, which in z-score
units is exactly 0.  That is the honest choice: any other value would be
smuggling in information no sensor supplied.  Gappy POD is different -- it can
in principle infer Uy and p through the cross-channel correlations captured by
the POD basis -- so it reconstructs all four channels from the basis.

The constant-predictor floor
----------------------------
The fields are z-scored with TRAIN-split statistics, so the train-split mean is
identically 0 in the evaluation units and the constant predictor scores
relative L2 == 1.0 in every channel by construction.  That is the whole point:
1.0 is the "I know nothing" line, and CRPS = E|y| is the corresponding
probabilistic line.  Predicting the *validation*-split mean instead would score
slightly BELOW 1.0 while quietly using held-out information, so it is reported
here only as a labelled contrast (``constant_val_mean_CONTRAST``), never as the
floor.

Seeding / fairness
------------------
The sensor draws are produced by calling ``helpers.build_sparse_condition``
itself, under the same ``torch.manual_seed(seed * 777 + snapshot_index)`` that
``ensemble_eval.main`` uses, with the coordinate tensor on the same device.
``torch.randperm`` on CUDA and on CPU give different permutations from the same
seed, so the draw is done on CUDA (as ``ensemble_eval`` does) and the resulting
indices are then used by these CPU estimators.  The sensor sets are therefore
bit-identical to the ones our model and every other baseline saw.

Metrics come from ``ensemble_eval.ensemble_metrics`` so the JSON schema matches
every other method.  A deterministic estimator is passed as a two-member
ensemble of identical members: the fair-CRPS estimator then returns exactly the
MAE (the K -> 1 limit is 0/0 and cannot be evaluated directly), and the spread
is exactly 0.

Usage:
    python baseline_classical_jhu.py --methods kdtree idw gappy_pod constant \
        --n-obs 19531 --out-dir ../Save_TrainedModel/JHU/baseline_classical
"""

from __future__ import annotations
from paths import DATA_DIR, RUNS_DIR

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch

os.environ.setdefault("JHU_SPLIT_MODE", "block")
os.environ.setdefault("JHU_SPLIT_GAP", "0")

from methods.dmfgen.helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from methods.dmfgen.ensemble_eval import (check_canonical_fingerprint, ensemble_metrics, jhu_band_metrics,
                                      require_compute_node)
from methods.classical.anchors_2d import lse_reconstruct


# ---------------------------------------------------------------------------
# np.percentile fast path (see --verify-percentile)
# ---------------------------------------------------------------------------
# ensemble_metrics calls np.percentile(e, q, axis=0) on a [K, 1.95M] array four
# times per snapshot.  numpy handles axis=0 by moving the axis and running 1.95M
# separate partitions, which costs ~125 s per snapshot and completely dominates
# a CPU baseline that otherwise takes 0.3 s.  np.sort along the same axis is
# ~150x faster.  The replacement below is numpy's default 'linear' method
# written out explicitly -- linear interpolation between the two order
# statistics bracketing the virtual index q/100*(K-1) -- so it is exact, not an
# approximation.  Anything that is not a 2-D axis=0 float call falls straight
# through to numpy.  `--verify-percentile` runs one snapshot through both paths
# and asserts every metric matches bit for bit.

_np_percentile = np.percentile


def _percentile_axis0(a, q, axis=None, **kw):
    a_ = np.asarray(a)
    if axis != 0 or a_.ndim != 2 or kw or not np.issubdtype(a_.dtype, np.floating):
        return _np_percentile(a, q, axis=axis, **kw)
    x = np.sort(a_, axis=0)
    k = x.shape[0]
    qq = np.atleast_1d(np.asarray(q, dtype=np.float64))
    vi = qq / 100.0 * (k - 1)
    lo = np.floor(vi).astype(np.intp)
    hi = np.ceil(vi).astype(np.intp)
    fr = (vi - lo).astype(x.dtype)
    out = np.stack([x[l] + f * (x[h] - x[l]) for l, h, f in zip(lo, hi, fr)], axis=0)
    return out if np.ndim(q) else out[0]


def enable_percentile_fastpath():
    np.percentile = _percentile_axis0


def disable_percentile_fastpath():
    np.percentile = _np_percentile

# Delta layout: the JHU file is flat under datasets/ (the Kestrel
# jhu_homogeneous_turbulence/outputfiles_diverse/ tree was not reproduced).
DATA = f"{DATA_DIR}/JHU_4cubes_stride100.h5"
RUN_DIR = (f"{RUNS_DIR}/JHU/pointcloud_ffm/"
           "iclr_jhu_xcube_spec02_DemoN29_20260822_140100")
FIELD_NAMES = ("Ux", "Uy", "Uz", "p")
COND_FIELDS = [0, 2]


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


class Timer:
    def __init__(self):
        self.wall = 0.0
        self.cpu = 0.0

    def __enter__(self):
        self._w = time.perf_counter()
        self._c = time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall += time.perf_counter() - self._w
        self.cpu += time.process_time() - self._c


# ---------------------------------------------------------------------------
# sensors -- bit-identical to ensemble_eval
# ---------------------------------------------------------------------------

def draw_sensors(coords_t: torch.Tensor, fields_t: torch.Tensor, snap: int,
                 n_obs: int, seed: int, device: str):
    """Reproduce ensemble_eval.main's sensor draw exactly.

    ensemble_eval does, per snapshot:
        torch.manual_seed(seed * 777 + snap)
        build_sparse_condition(coords[1,N,3], fields[1,N,C], cond_fields,
                               n_obs_min=n_obs, n_obs_max=n_obs)
    with the tensors on `--device` (cuda:0 by default).  We call the same
    function under the same seed on the same device, so the observation index
    sets are identical bit for bit.
    """
    c = coords_t.unsqueeze(0).to(device)
    f = fields_t.unsqueeze(0).to(device)
    torch.manual_seed(seed * 777 + int(snap))
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=c, fields_full=f,
        cond_fields=COND_FIELDS, n_obs_min=[n_obs] * len(COND_FIELDS),
        n_obs_max=[n_obs] * len(COND_FIELDS),
    )
    idx = oi[0].cpu().numpy()
    fid = ofid[0].cpu().numpy()
    val = ov[0, :, 0].cpu().numpy()
    out = {}
    for fld in COND_FIELDS:
        m = fid == fld
        out[fld] = (idx[m].astype(np.int64), val[m].astype(np.float32))
    del c, f, oc, ov, om, oi, ofid
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def kd_predict(coords_box: np.ndarray, sensors: Dict, n_ch: int,
               mode: str, k: int, boxsize):
    """NN (mode='nn') or inverse-distance-weighted (mode='idw') interpolation.

    Unobserved channels are left at 0 == the train-split mean in z-score units.
    """
    from scipy.spatial import cKDTree
    pred = np.zeros((coords_box.shape[0], n_ch), dtype=np.float32)
    for fld, (idx, val) in sensors.items():
        tree = cKDTree(coords_box[idx], boxsize=boxsize)
        if mode == "nn":
            _, nn = tree.query(coords_box, k=1, workers=-1)
            pred[:, fld] = val[nn]
        else:
            kk = min(k, len(idx))
            d, nn = tree.query(coords_box, k=kk, workers=-1)
            exact = d[:, 0] < 1e-11
            d = np.maximum(d, 1e-12)
            w = 1.0 / d
            pred[:, fld] = (w * val[nn]).sum(1) / w.sum(1)
            pred[exact, fld] = val[nn[exact, 0]]
    return pred


class GappyPOD:
    """POD basis over training snapshots, coefficients from sparse sensors.

    The basis is never materialised.  With the snapshot (method-of-snapshots)
    decomposition  Xc = U S V^T  restricted to rank r, the modes are
    Phi = S^-1 U^T Xc, so any modal reconstruction  a @ Phi  equals  w @ Xc
    with  w = U S^-1 a.  Only Xc (n_train x N*C) is kept, which is 4.7 GB for
    150 snapshots -- materialising Phi as well would double that for nothing.
    """

    def __init__(self, Xc: np.ndarray, mu: np.ndarray, rank: int):
        self.Xc = Xc                       # [n_train, P] centred, float32
        self.mu = mu                       # [P]
        G = (Xc @ Xc.T).astype(np.float64)  # [n, n] Gram
        lam, U = np.linalg.eigh(G)
        order = np.argsort(lam)[::-1]
        lam, U = lam[order], U[:, order]
        lam = np.clip(lam, 1e-12, None)
        self.lam = lam
        self.U = U
        self.energy = np.cumsum(lam) / lam.sum()
        self.set_rank(rank)

    def set_rank(self, r: int):
        self.r = int(min(r, self.Xc.shape[0] - 1))
        self.Ur = self.U[:, :self.r]
        self.inv_s = 1.0 / np.sqrt(self.lam[:self.r])

    def modes_at(self, cols: np.ndarray) -> np.ndarray:
        """Phi[:r, cols].T  ->  [n_obs, r]."""
        sub = self.Xc[:, cols].astype(np.float64)              # [n_train, n_obs]
        return (sub.T @ self.Ur) * self.inv_s[None, :]         # [n_obs, r]

    def reconstruct(self, cols: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        A = self.modes_at(cols)
        a, *_ = np.linalg.lstsq(A, y_obs - self.mu[cols].astype(np.float64),
                                rcond=None)
        w = (self.Ur * self.inv_s[None, :]) @ a                # [n_train]
        return self.mu + (w.astype(np.float32) @ self.Xc)


def obs_columns(sensors: Dict, n_ch: int):
    """Flatten (point, channel) sensor addresses into columns of the [N*C] vector."""
    cols, vals = [], []
    for fld, (idx, val) in sorted(sensors.items()):
        cols.append(idx * n_ch + fld)
        vals.append(val)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    o = np.argsort(cols)
    return cols[o], vals[o].astype(np.float64)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(pred: np.ndarray, true: np.ndarray) -> Dict:
    """Deterministic prediction -> ensemble_eval metrics.

    K=1 cannot be scored (the fair-CRPS pair term is 0/0 and the ddof=1 spread
    is undefined), so the single member is duplicated.  For two identical
    members the fair estimator's pair term is exactly 0 and CRPS reduces to the
    MAE, which is the correct CRPS of a deterministic forecast.
    """
    ens = np.repeat(pred[None], 2, axis=0)
    return ensemble_metrics(ens, true, list(FIELD_NAMES))


def summarize(per_snap: List[Dict]) -> Dict:
    keys = list(per_snap[0]["aggregate"].keys())
    fields = list(per_snap[0]["per_field"].keys())
    return {
        "aggregate": {k: float(np.mean([s["aggregate"][k] for s in per_snap]))
                      for k in keys},
        "per_field": {f: {k: float(np.mean([s["per_field"][f][k] for s in per_snap]))
                          for k in keys} for f in fields},
        # kept so the sweep can be re-averaged over any snapshot subset (e.g.
        # the 12 snapshots the model's own sensor sweep happened to use).
        "per_snapshot": [
            {"snapshot": i,
             "rel_l2_mean": float(s["aggregate"]["rel_l2_mean"]),
             "crps": float(s["aggregate"]["crps"]),
             "per_field": {f: {"rel_l2_mean": float(s["per_field"][f]["rel_l2_mean"]),
                               "crps": float(s["per_field"][f]["crps"])}
                           for f in fields},
             **({"band": s["band"]} if "band" in s else {})}
            for i, s in enumerate(per_snap)],
        "n_snapshots": len(per_snap),
    }


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_split(ds: TurbulentCombustionH5Dataset, indices) -> np.ndarray:
    """[n, N, C] z-scored with train stats."""
    mean = ds.mean.numpy()
    std = ds.std.numpy()
    out = np.empty((len(indices), ds.num_points, ds.num_fields), dtype=np.float32)
    with h5py.File(ds.h5_path, "r") as f:
        for i, t in enumerate(indices):
            out[i] = (f["fields"][0, int(t), :, 0, 0, :].astype(np.float32) - mean) / std
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA)
    p.add_argument("--run-dir", default=RUN_DIR,
                   help="Only used for dataset_stats.pt, so the z-scoring is "
                        "byte-identical to the model's.")
    p.add_argument("--methods", nargs="+",
                   default=["constant", "kdtree", "idw", "gappy_pod"])
    p.add_argument("--n-obs", type=int, nargs="+", default=[19531],
                   help="Sensors per OBSERVED field. Several values = sweep.")
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="ensemble_eval --seed")
    p.add_argument("--sensor-device", default="cuda:0",
                   help="Device for the sensor draw; must match ensemble_eval's "
                        "--device for bit-identical index sets.")
    p.add_argument("--idw-k", type=int, default=8)
    p.add_argument("--pod-ranks", type=int, nargs="+",
                   default=[2, 5, 10, 20, 40, 60, 80, 99],
                   help="Rank candidates screened on TRAIN cube 2.")
    p.add_argument("--ridge-grid", type=float, nargs="+",
                   default=[0.0] + [10.0 ** e for e in range(-8, 1)],
                   help="LSE penalty in units of the largest squared singular value "
                        "of the sensed training matrix, screened on TRAIN cube 2.")
    p.add_argument("--no-periodic", action="store_true",
                   help="kept for compatibility; non-periodic is now the default")
    p.add_argument("--periodic", action="store_true",
                   help="use the WRONG periodic wrap (the cutout spans 12.11%% of the "
                        "2pi domain and is NOT periodic; audit 2026-08-28 s26) -- "
                        "only for reproducing the superseded numbers")
    p.add_argument("--verify-percentile", action="store_true",
                   help="Score one snapshot with and without the np.percentile "
                        "fast path and assert the metric dicts are identical.")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-dir", default=f"{RUNS_DIR}/JHU/baseline_classical")
    p.add_argument("--band-metrics", action="store_true",
                   help="Also store per-snapshot band energy / correlation for IDW.")
    p.add_argument("--eval-split", default="test", choices=["test", "tune", "val", "train"],
                   help="held-out half (protocol): 'test' = the reported 25 frames, "
                        "'tune' = the selection half, 'val' = all 50. "
                        "'train' scores the TRAIN cubes (in-sample leakage probe, "
                        "frame-difficulty control for the learned rows). Only "
                        "training-free interpolants are allowed there: gappy POD "
                        "would be fitted on the frames it scores.")
    args = p.parse_args()

    if args.eval_split == "train":
        if {"gappy_pod", "lse"} & set(args.methods):
            raise SystemExit("[guard] --eval-split train: gappy_pod / lse are fitted on "
                             "the train cubes and cannot score them; drop them.")
        if not args.tag or "insample" not in args.tag:
            raise SystemExit("[guard] --eval-split train requires --tag containing "
                             "'insample' (the split must key the output name).")

    require_compute_node()
    dev = args.sensor_device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        # Was warn-only; a CPU draw is a DIFFERENT sensor layout, so canonical
        # numbers must never come from it. ALLOW_LOGIN_EVAL=1 keeps the old
        # warn-and-continue behaviour for debugging.
        msg = ("CUDA unavailable: sensor draws would fall back to CPU and will "
               "NOT be bit-identical to ensemble_eval's CUDA draws.")
        if os.environ.get("ALLOW_LOGIN_EVAL") == "1":
            print("[warn] " + msg)
            dev = "cpu"
        else:
            raise SystemExit("[nodecheck] " + msg +
                             " Set ALLOW_LOGIN_EVAL=1 to override.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = str(Path(args.run_dir) / "dataset_stats.pt")
    held = args.eval_split if args.eval_split in ("val", "tune", "test") else "val"
    ds_val = TurbulentCombustionH5Dataset(
        args.data, split=held, train_ratio=0.75, field_names=FIELD_NAMES,
        seed=42, time_stride=1, stats_path=stats)
    ds_tr = TurbulentCombustionH5Dataset(
        args.data, split="train", train_ratio=0.75, field_names=FIELD_NAMES,
        seed=42, time_stride=1, stats_path=stats)
    N, C = ds_val.num_points, ds_val.num_fields
    print(f"[data] train {len(ds_tr)} snaps (t={ds_tr.indices[0]}..{ds_tr.indices[-1]}), "
          f"val {len(ds_val)} snaps (t={ds_val.indices[0]}..{ds_val.indices[-1]}), "
          f"N={N} C={C}", flush=True)
    print(f"[data] train-split mean={ds_val.mean.numpy()} std={ds_val.std.numpy()}",
          flush=True)

    # Coordinates rescaled so the periodic HIT box is exactly [0,1)^3.
    craw = ds_val.coords_raw.numpy().astype(np.float64)
    side = int(round(N ** (1.0 / 3.0)))
    lo = craw.min(0)
    dx = (craw.max(0) - lo) / (side - 1)
    box = side * dx
    coords_box = np.ascontiguousarray(((craw - lo) / box).astype(np.float64))
    boxsize = 1.0 if args.periodic else None
    print(f"[grid] side={side} dx={dx} periodic={boxsize is not None}", flush=True)

    ds_eval = ds_tr if args.eval_split == "train" else ds_val
    n_snap = min(args.n_snapshots, len(ds_eval))
    if args.eval_split == "train":
        # Same selection rule as ensemble_eval / eval_latentfm_ensemble on the
        # train split: np.random.default_rng(seed).choice(len(split), n),
        # so the learned in-sample runs and this control score the SAME frames
        # under the SAME per-position sensor seeds.
        rng = np.random.default_rng(args.seed)
        snaps = sorted(int(s) for s in
                       rng.choice(len(ds_eval), size=n_snap, replace=False))
        print(f"[insample] scoring {n_snap} of {len(ds_eval)} TRAIN frames "
              f"(split positions {snaps[:5]}...)", flush=True)
    else:
        snaps = list(range(n_snap))
        # Canonical snapshot selection is rng.choice(len(ds_val), 50, replace=False)
        # (ensemble_eval.py:291). range(n_snap) is SET-identical to it only when it
        # covers the whole split; a partial run scores a different snapshot set.
        if n_snap < len(ds_val):
            print(f"[warn] n_snapshots={n_snap} < len(val)={len(ds_val)}: snapshot "
                  "set is range(n_snap), NOT the canonical rng.choice subset -- "
                  "numbers are not comparable to canonical runs.", flush=True)

    # --- load the scored frames once (indexed by split position) ----------
    # Yval[i] must be the frame at split position i for every i in snaps, so
    # the train probe loads the whole split (150 frames) rather than a subset.
    t = Timer()
    with t:
        Yval = load_split(ds_eval, ds_eval.indices if args.eval_split == "train"
                          else ds_val.indices[:n_snap])
    print(f"[data] {args.eval_split} frames loaded in {t.wall:.1f}s, "
          f"peak RSS {peak_rss_gb():.1f} GB", flush=True)

    percentile_check = None
    if args.verify_percentile:
        probe = np.zeros((N, C), dtype=np.float32)
        probe[:, 0] = np.linspace(-2, 2, N, dtype=np.float32)
        disable_percentile_fastpath()
        t0 = time.perf_counter()
        ref = score(probe, Yval[0])
        t_ref = time.perf_counter() - t0
        enable_percentile_fastpath()
        t0 = time.perf_counter()
        fast = score(probe, Yval[0])
        t_fast = time.perf_counter() - t0
        same = json.dumps(ref, sort_keys=True) == json.dumps(fast, sort_keys=True)
        percentile_check = {"identical": bool(same),
                            "numpy_percentile_s": t_ref, "fastpath_s": t_fast}
        print(f"[verify] np.percentile fast path identical={same} "
              f"({t_ref:.1f}s -> {t_fast:.1f}s per snapshot)", flush=True)
        assert same, "percentile fast path changed the metrics"
    enable_percentile_fastpath()

    timing: Dict[str, Dict] = {}
    results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # 1. constant predictors  (sensor-count independent)
    # ------------------------------------------------------------------
    if "constant" in args.methods:
        # train-split mean == 0 in z-score units, by construction of the stats.
        tf = Timer()
        with tf:
            const_train = np.zeros((N, C), dtype=np.float32)
        ti = Timer()
        per = []
        for i in snaps:
            with ti:
                pred = const_train
            per.append(score(pred, Yval[i]))
        results["constant_train_mean"] = summarize(per)
        results["constant_train_mean"]["note"] = (
            "Predicts the TRAIN-split per-channel mean, which is exactly 0 in "
            "z-score units; relative L2 is therefore identically 1.0 by "
            "construction and CRPS is E|y| on the held-out cube. This is the "
            "no-information floor.")
        # contrast only: uses held-out information, NOT a legitimate baseline
        mu_val = Yval.mean(axis=(0, 1)).astype(np.float32)
        per = []
        for i in snaps:
            per.append(score(np.broadcast_to(mu_val, (N, C)).copy(), Yval[i]))
        results["constant_val_mean_CONTRAST"] = summarize(per)
        results["constant_val_mean_CONTRAST"]["note"] = (
            "NOT a legitimate baseline: predicts the VALIDATION-split mean, "
            "which uses held-out information. Reported only to show how little "
            "it buys over the train-split mean (val mean in z-units = "
            f"{mu_val.tolist()}).")
        timing["constant"] = {"fit_wall_s": tf.wall,
                              "infer_wall_s_per_field": ti.wall / n_snap,
                              "infer_cpu_s_per_field": ti.cpu / n_snap,
                              "peak_rss_gb": peak_rss_gb(), "device": "cpu"}
        print(f"[constant] train-mean aggregate relL2="
              f"{results['constant_train_mean']['aggregate']['rel_l2_mean']:.4f} "
              f"CRPS={results['constant_train_mean']['aggregate']['crps']:.4f}",
              flush=True)
        print("[constant] per-field CRPS: " + " ".join(
            f"{f}:{results['constant_train_mean']['per_field'][f]['crps']:.4f}"
            for f in FIELD_NAMES), flush=True)

    # ------------------------------------------------------------------
    # 2. gappy POD and LSE (fitted once, then swept over sensor counts)
    # ------------------------------------------------------------------
    pod = None
    lse_ridge = None
    if {"gappy_pod", "lse"} & set(args.methods):
        tfit = Timer()
        with tfit:
            Xtr = load_split(ds_tr, ds_tr.indices).reshape(len(ds_tr), -1)
            mu = Xtr.mean(axis=0)
            Xtr -= mu
        print(f"[pod] train matrix {Xtr.shape} loaded/centred in {tfit.wall:.1f}s, "
              f"peak RSS {peak_rss_gb():.1f} GB", flush=True)

        # --- rank selection on TRAIN cube 2 only, never on cube 3 --------
        n_c = len(ds_tr) // 3                       # 50 snapshots per cube
        basis_ids = np.arange(0, 2 * n_c)           # cubes 0,1
        holdout_ids = np.arange(2 * n_c, 3 * n_c)   # cube 2
        Xb = np.ascontiguousarray(Xtr[basis_ids])
        mub = Xb.mean(axis=0)
        Xb -= mub
        with tfit:
            sel = GappyPOD(Xb, mub + mu, rank=1)
        n_obs_sel = max(args.n_obs)
        rank_scan = {}
        hold_sub = holdout_ids[::10]                # 5 snapshots: enough to rank-order
        sens_cache = {}
        for j, gi in enumerate(hold_sub):
            true_field = torch.from_numpy((Xtr[gi] + mu).reshape(N, C))
            sens_cache[gi] = draw_sensors(ds_val.coords, true_field, snap=j,
                                          n_obs=n_obs_sel, seed=args.seed, device=dev)
        for r in (args.pod_ranks if "gappy_pod" in args.methods else []):
            if r >= len(basis_ids):
                continue
            with tfit:
                sel.set_rank(r)
            errs = []
            for gi in hold_sub:
                cols, vals = obs_columns(sens_cache[gi], C)
                true_vec = Xtr[gi] + mu
                with tfit:
                    rec = sel.reconstruct(cols, vals)
                errs.append(float(np.linalg.norm(rec - true_vec) /
                                  (np.linalg.norm(true_vec) + 1e-12)))
            rank_scan[r] = float(np.mean(errs))
            print(f"[pod] rank {r:3d}: holdout(train cube 2) relL2 {rank_scan[r]:.4f} "
                  f"cum-energy {sel.energy[r - 1]:.4f}", flush=True)
        if "lse" in args.methods:
            ridge_scan = {}
            for rg in args.ridge_grid:
                errs = []
                for gi in hold_sub:
                    cols, vals = obs_columns(sens_cache[gi], C)
                    true_vec = Xtr[gi] + mu
                    rec = lse_reconstruct(Xb, mub + mu, cols, vals, rg)
                    errs.append(float(np.linalg.norm(rec - true_vec) /
                                      (np.linalg.norm(true_vec) + 1e-12)))
                ridge_scan[rg] = float(np.mean(errs))
                print(f"[lse] ridge {rg:g}: holdout(train cube 2) relL2 {ridge_scan[rg]:.4f}",
                      flush=True)
            lse_ridge = min(ridge_scan, key=ridge_scan.get)
            timing["lse_fit"] = {"selected_ridge": lse_ridge,
                                 "ridge_scan_holdout_train_cube2": ridge_scan}
        del sel, Xb
        if "gappy_pod" in args.methods:
            best_rank = min(rank_scan, key=rank_scan.get) if rank_scan else 20
            with tfit:
                pod = GappyPOD(Xtr, mu, rank=best_rank)
            print(f"[pod] selected rank {pod.r} (cum-energy {pod.energy[pod.r - 1]:.4f}); "
                  f"fit wall {tfit.wall:.1f}s cpu {tfit.cpu:.1f}s "
                  f"peak RSS {peak_rss_gb():.1f} GB", flush=True)
            timing["gappy_pod_fit"] = {
                "fit_wall_s": tfit.wall, "fit_cpu_s": tfit.cpu,
                "peak_rss_gb": peak_rss_gb(), "device": "cpu",
                "selected_rank": pod.r,
                "rank_scan_holdout_train_cube2": rank_scan,
                "cum_energy_at_rank": float(pod.energy[pod.r - 1]),
                "cum_energy_curve": [float(v) for v in pod.energy],
            }

    # ------------------------------------------------------------------
    # 3. sensor-dependent methods
    # ------------------------------------------------------------------
    for n_obs in args.n_obs:
        frac = 100.0 * n_obs / N
        sens = {}
        tsens = Timer()
        with tsens:
            for i in snaps:
                sens[i] = draw_sensors(ds_val.coords, torch.from_numpy(Yval[i]),
                                       snap=i, n_obs=n_obs, seed=args.seed, device=dev)
        print(f"\n===== n_obs={n_obs} ({frac:.3f}% of {N}) : sensor draw "
              f"{tsens.wall:.1f}s on {dev} =====", flush=True)
        for i in snaps:
            _n = sum(int(v[0].size) for v in sens[i].values())
            _s = sum(int(v[0].sum()) for v in sens[i].values())
            if i == 29:
                print(f"[seedcheck] snap={i} sensors={_n} idx_sum={_s}", flush=True)
                check_canonical_fingerprint(i, _n, _s, args.seed, COND_FIELDS,
                                            [n_obs] * len(COND_FIELDS))

        for name, mode in (("kdtree", "nn"), ("idw", "idw")):
            if name not in args.methods:
                continue
            tt = Timer()
            per = []
            for i in snaps:
                with tt:
                    pred = kd_predict(coords_box, sens[i], C, mode, args.idw_k, boxsize)
                per.append(score(pred, Yval[i]))
                if args.band_metrics and name == "idw":
                    per[-1]["band"] = jhu_band_metrics(pred[None], Yval[i], ds_val.std.numpy())
            key = f"{name}_n{n_obs}"
            results[key] = summarize(per)
            results[key]["n_obs_per_observed_field"] = n_obs
            results[key]["sensor_fraction_pct"] = frac
            results[key]["note"] = (
                "Unobserved channels Uy and p are predicted as the train-split "
                "mean (0 in z-score units) -- no sensor constrains them.")
            timing[key] = {"fit_wall_s": 0.0,
                           "infer_wall_s_per_field": tt.wall / n_snap,
                           "infer_cpu_s_per_field": tt.cpu / n_snap,
                           "peak_rss_gb": peak_rss_gb(), "device": "cpu"}
            a = results[key]["aggregate"]
            pf = results[key]["per_field"]
            print(f"[{name} n={n_obs}] agg relL2={a['rel_l2_mean']:.4f} "
                  f"CRPS={a['crps']:.4f} | " +
                  " ".join(f"{f}:{pf[f]['rel_l2_mean']:.4f}/{pf[f]['crps']:.4f}"
                           for f in FIELD_NAMES) +
                  f" | {tt.wall / n_snap:.2f} s/field", flush=True)

        linear = []
        if pod is not None:
            linear.append(("gappy_pod", lambda cols, vals: pod.reconstruct(cols, vals)))
        if lse_ridge is not None:
            linear.append(("lse", lambda cols, vals: lse_reconstruct(Xtr, mu, cols, vals, lse_ridge)))
        for name, predict in linear:
            tt = Timer()
            per = []
            for i in snaps:
                cols, vals = obs_columns(sens[i], C)
                with tt:
                    pred = predict(cols, vals).reshape(N, C)
                per.append(score(pred, Yval[i]))
            key = f"{name}_n{n_obs}"
            results[key] = summarize(per)
            results[key]["n_obs_per_observed_field"] = n_obs
            results[key]["sensor_fraction_pct"] = frac
            if name == "gappy_pod":
                results[key]["rank"] = pod.r
            else:
                results[key]["ridge"] = lse_ridge
            timing[key] = {"infer_wall_s_per_field": tt.wall / n_snap,
                           "infer_cpu_s_per_field": tt.cpu / n_snap,
                           "peak_rss_gb": peak_rss_gb(), "device": "cpu"}
            a = results[key]["aggregate"]
            pf = results[key]["per_field"]
            print(f"[{name} n={n_obs}] agg relL2={a['rel_l2_mean']:.4f} "
                  f"CRPS={a['crps']:.4f} | " +
                  " ".join(f"{f}:{pf[f]['rel_l2_mean']:.4f}/{pf[f]['crps']:.4f}"
                           for f in FIELD_NAMES) +
                  f" | {tt.wall / n_snap:.2f} s/field", flush=True)
        del sens

    payload = {
        "split": args.eval_split,
        "protocol": {
            "data": args.data,
            "split": ("IN-SAMPLE: train cubes 0-2 scored (leakage control)"
                      if args.eval_split == "train" else
                      f"cross-cube (train cubes 0-2, eval cube 3), held-out half '{held}' "
                      "(protocol: odd positions = tune, even = test)"),
            "eval_split": args.eval_split,
            "snapshot_key": "split index",
            "snapshot_ids": [int(s) for s in snaps],
            "JHU_SPLIT_MODE": os.environ["JHU_SPLIT_MODE"],
            "JHU_SPLIT_GAP": os.environ["JHU_SPLIT_GAP"],
            "train_ratio": 0.75, "field_names": list(FIELD_NAMES),
            "cond_fields": COND_FIELDS, "n_points": N,
            "n_snapshots": n_snap, "seed": args.seed,
            "sensor_draw": ("helpers.build_sparse_condition under "
                            "torch.manual_seed(seed*777+snap) on " + dev +
                            " -- identical to ensemble_eval.main"),
            "standardization": "per-field z-score with TRAIN-split stats "
                               f"(mean={ds_val.mean.numpy().tolist()}, "
                               f"std={ds_val.std.numpy().tolist()})",
            "crps": "deterministic estimator scored as a 2-member identical "
                    "ensemble; fair CRPS then equals the MAE exactly",
            "periodic_kdtree": boxsize is not None,
            "idw_k": args.idw_k,
            "percentile_fastpath_check": percentile_check,
        },
        "timing_and_memory": timing,
        "results": results,
    }
    tag = args.tag or ("sweep" if len(args.n_obs) > 1 else f"n{args.n_obs[0]}")
    out = out_dir / f"classical_baselines_{tag}.json"
    json.dump(payload, open(out, "w"), indent=1)
    print(f"\n[out] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
