"""Local-informativeness diagnostic for the interpolation anchors.

For a field u (z-scored with TRAIN statistics, the units of every table) and a sensor
pool P, the expected squared error of nearest-sensor copying at a query point x whose
nearest sensor sits at displacement D(x) is the second-order structure function
S2(D) = E[(u(x+D) - u(x))^2] between pool points and query points. With S2 estimated
from TRAINING frames only and D(x) from random sensor draws of the evaluated count,

    eps_pred = sqrt( E_x[S2(D(x))] / E[u^2] )

predicts the relative L2 of nearest-neighbour interpolation without running it.
Denominators: 'train' uses E_train[u^2] (= 1 by the z-scoring); 'scored' uses the
mean square of each scored frame and averages the per-frame ratio over frames, as the
tables do. ell = smallest separation at which the radially averaged S2 reaches
E_train[u^2] (= 1): beyond it, copying the nearest reading is no better than the
training mean (R(ell) = 1/2 for a zero-mean field).

Sensor draws here are numpy uniform draws of the same count from the same pool
(the canonical draws use torch.randperm on CUDA and cannot be reproduced on CPU);
the statistic depends only on the distance distribution, which is identical.

    python -m analysis.diagnostics.idw_diag <kolmogorov|cylinder|jhu>
writes outputs/paper/diagnostics/idw_<regime>.json (Table tab:diagnostics, local columns). Run one
regime per process: the draws share one seeded generator, so a regime's draws depend on
which regimes ran before it in the same process.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.fft as sfft
from scipy.spatial import cKDTree

from paths import REPO_ROOT as REPO

OUT = REPO / "outputs/paper/diagnostics"
RNG = np.random.default_rng(20261006)
W = 64  # fft / kdtree workers


def held_out_test(first, last, n_frames, blocks=1):
    """TEST half (even positions) of the held-out block, then the protocol frame_list."""
    idx = np.arange(first, last)[0::2]
    n = len(idx)
    if blocks <= 1:
        pos = [(i * n) // n_frames for i in range(n_frames)]
    else:
        span, per = n // blocks, n_frames // blocks
        pos = [b * span + (i * span) // per for b in range(blocks) for i in range(per)]
    return idx[pos]


def classical(path, key, field):
    d = json.load(open(REPO / path))["results"]
    return d[key]["per_field"][field]["rel_l2_mean"] if key in d else None


def radial(S2grid, dist, edges):
    """Bin-average a structure-function grid by separation."""
    b = np.digitize(dist.ravel(), edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    num = np.bincount(b[ok], S2grid.ravel()[ok], minlength=len(edges) - 1)
    cnt = np.bincount(b[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore"):
        return num / np.maximum(cnt, 1), cnt


def ell_from(r, s2, level=1.0):
    """Smallest separation where the radial S2 reaches `level` (linear interpolation)."""
    above = np.where(np.isfinite(s2) & (s2 >= level))[0]
    if len(above) == 0:
        return None
    i = above[0]
    if i == 0:
        return float(r[0])
    return float(r[i - 1] + (level - s2[i - 1]) * (r[i] - r[i - 1]) / (s2[i] - s2[i - 1]))


def pred(num_mean, den_train, den_frames):
    return {"eps_pred_train": float(np.sqrt(num_mean / den_train)),
            "eps_pred_scored": float(np.mean(np.sqrt(num_mean / np.asarray(den_frames))))}


# ---------------------------------------------------------------------------
def kolmogorov():
    t0 = time.time()
    h5 = REPO / "data/kolmogorov2d/Kolmogorov2D_shu_stride4.h5"
    n = 256
    dx = 2 * np.pi / n
    with h5py.File(h5, "r") as f:
        F = f["fields"]
        # train stats over all train frames (as compute_train_stats)
        s = sq = 0.0; cnt = 0
        for a in range(0, 2560, 256):
            x = F[0, a:a + 256, :, 0, 0, 0].astype(np.float64)
            s += x.sum(); sq += (x ** 2).sum(); cnt += x.size
        mean = s / cnt; std = np.sqrt(sq / cnt - mean ** 2)
        tr = np.arange(0, 2560, 8)                       # 320 train frames
        S2 = np.zeros((n, n)); m2 = 0.0
        for t in tr:
            u = ((F[0, int(t), :, 0, 0, 0] - mean) / std).astype(np.float64).reshape(n, n)  # [y, x]
            Fu = sfft.fft2(u, workers=W)
            C = sfft.ifft2(np.abs(Fu) ** 2, workers=W).real / u.size   # periodic autocorrelation
            S2 += 2 * (np.mean(u ** 2) - C)
            m2 += np.mean(u ** 2)
        S2 /= len(tr); m2 /= len(tr)
        te = held_out_test(2560, 3200, 50)
        den = [float(np.mean(((F[0, int(t), :, 0, 0, 0] - mean) / std).astype(np.float64) ** 2)) for t in te]
    # radial profile with minimum-image distances
    o = np.minimum(np.arange(n), n - np.arange(n))
    dist = np.hypot(o[:, None], o[None, :])
    edges = np.arange(0, 64.5, 0.5)
    s2r, _ = radial(S2, dist, edges)
    rc = 0.5 * (edges[1:] + edges[:-1])
    ell = ell_from(rc, s2r, 1.0)
    jj, ii = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")   # [y, x]
    pts = np.stack([ii.ravel(), jj.ravel()], 1).astype(np.float64)
    rows = []
    sweep = "outputs/runs/kolmogorov2d/baseline_classical/classical_baselines_sweep_nonperiodic_2d.json"
    main_p = "outputs/runs/kolmogorov2d/baseline_classical/classical_baselines_main_1pct_periodic_2d.json"
    for ns in [65, 164, 655, 1965, 6554]:
        for periodic in ([False, True] if ns == 655 else [False]):
            num, gap = [], []
            for rep in range(10):
                sid = RNG.choice(n * n, ns, replace=False)
                tree = cKDTree(pts[sid], boxsize=(n if periodic else None))
                d, k = tree.query(pts, k=1, workers=W)
                disp = (pts[sid][k] - pts).astype(np.int64) % n   # (dx, dy) offsets mod period
                num.append(np.mean(S2[disp[:, 1], disp[:, 0]]))
                gap.append(np.mean(d))
            num = float(np.mean(num)); g = float(np.mean(gap))
            src = main_p if periodic else sweep
            r = {"setting": f"kolmogorov n={ns}" + (" periodic" if periodic else ""),
                 "field": "vorticity", "n_sensors": ns, "pool": "all 65536 grid points",
                 "mean_gap_cells": g, "mean_gap_phys": g * dx, "ell_cells": ell,
                 "ell_phys": None if ell is None else ell * dx,
                 "gap_over_ell": None if ell is None else g / ell,
                 "mean_S2_at_gap": num, **pred(num, m2, den),
                 "eps_nn_measured": classical(src, f"kdtree_n{ns}", "vorticity"),
                 "eps_idw_measured": classical(src, f"idw_n{ns}", "vorticity"),
                 "measured_source": src}
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "measured_source"}, flush=True)
    prof = {"r_cells": rc[:60].tolist(), "S2": s2r[:60].tolist(), "train_mean_square": m2}
    print(f"[kolm] done {time.time()-t0:.0f}s  ell={ell}", flush=True)
    return rows, prof


# ---------------------------------------------------------------------------
def jhu():
    t0 = time.time()
    h5 = REPO / "data/JHU_4cubes_stride100.h5"
    n, P = 125, 256
    dx = 2 * np.pi / 1024
    mean = np.array([0.024666232988238335, -0.11473891884088516, -0.08747481554746628, -0.010348735377192497])
    std = np.array([0.6237276196479797, 0.6208027601242065, 0.5550218224525452, 0.3945266008377075])
    chans = {"Ux": 0, "Uz": 2}
    tr = np.concatenate([np.arange(c * 50, c * 50 + 50, 5) for c in range(3)])   # 30 train frames
    te = held_out_test(150, 200, 25)                                                # 25 test frames
    m = np.zeros((P, P, P)); m[:n, :n, :n] = 1
    M = sfft.rfftn(m, workers=W)
    Ncnt = sfft.irfftn(np.conj(M) * M, s=(P, P, P), workers=W)
    S2 = {}; m2 = {}; den = {}
    with h5py.File(h5, "r") as f:
        F = f["fields"]
        for name, c in chans.items():
            acc = np.zeros((P, P, P)); ms = 0.0
            for t in tr:
                u = ((F[0, int(t), :, 0, 0, c] - mean[c]) / std[c]).astype(np.float64).reshape(n, n, n)
                up = np.zeros((P, P, P)); up[:n, :n, :n] = u
                U = sfft.rfftn(up, workers=W)
                Q = sfft.rfftn(up ** 2, workers=W)
                Pc = sfft.irfftn(np.conj(U) * U, s=(P, P, P), workers=W)
                A = sfft.irfftn(np.conj(M) * Q, s=(P, P, P), workers=W)       # sum m(x) u^2(x+D)
                B = np.roll(np.flip(A, axis=(0, 1, 2)), 1, axis=(0, 1, 2))     # A(-D)
                acc += A + B - 2 * Pc
                ms += np.mean(u ** 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                S2[name] = acc / len(tr) / np.maximum(Ncnt, 0.5)
            m2[name] = ms / len(tr)
            den[name] = [float(np.mean(((F[0, int(t), :, 0, 0, c] - mean[c]) / std[c]).astype(np.float64) ** 2)) for t in te]
            print(f"[jhu] S2 {name} done {time.time()-t0:.0f}s, train m2={m2[name]:.3f}, test m2 mean={np.mean(den[name]):.3f}", flush=True)
    # radial profiles over offsets |D| <= 62 (well-sampled overlap)
    o = np.arange(P); o = np.where(o < P // 2, o, o - P)
    ox, oy, oz = np.meshgrid(o, o, o, indexing="ij")
    dist = np.sqrt(ox ** 2 + oy ** 2 + oz ** 2)
    edges = np.arange(0, 63.5, 0.5); rc = 0.5 * (edges[1:] + edges[:-1])
    prof, ell = {}, {}
    for name in chans:
        s2r, _ = radial(np.nan_to_num(S2[name]), dist, edges)
        prof[name] = {"r_cells": rc.tolist(), "S2": s2r.tolist(), "train_mean_square": m2[name]}
        ell[name] = ell_from(rc, s2r, 1.0)
        print(f"[jhu] {name}: S2 at r=1,2,5,10,20,40,60 cells:",
              [round(float(np.interp(r, rc, s2r)), 4) for r in (1, 2, 5, 10, 20, 40, 60)], "ell", ell[name], flush=True)
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
    rows = []
    sweep = "outputs/runs/JHU/baseline_classical/classical_baselines_sweep.json"
    for ns in [1953, 4883, 9766, 19531, 39062, 97656, 195312]:
        for name in chans:
            num, gap = [], []
            for rep in range(3):
                sid = RNG.choice(n ** 3, ns, replace=False)
                d, k = cKDTree(g[sid]).query(g, k=1, workers=W)
                disp = (g[sid][k] - g).astype(np.int64) % P
                num.append(np.mean(S2[name][disp[:, 0], disp[:, 1], disp[:, 2]]))
                gap.append(np.mean(d))
            num = float(np.mean(num)); gg = float(np.mean(gap))
            e = ell[name]
            r = {"setting": f"jhtdb n={ns}", "field": name, "n_sensors": ns, "pool": "all 125^3 grid points",
                 "mean_gap_cells": gg, "mean_gap_phys": gg * dx, "ell_cells": e,
                 "ell_phys": None if e is None else e * dx,
                 "gap_over_ell": None if e is None else gg / e,
                 "mean_S2_at_gap": num, **pred(num, m2[name], den[name]),
                 "eps_nn_measured": classical(sweep, f"kdtree_n{ns}", name),
                 "eps_idw_measured": classical(sweep, f"idw_n{ns}", name),
                 "measured_source": sweep}
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "measured_source"}, flush=True)
    print(f"[jhu] done {time.time()-t0:.0f}s", flush=True)
    return rows, prof


# ---------------------------------------------------------------------------
def cylinder():
    t0 = time.time()
    h5 = REPO / "data/cylinder2d/Cylinder2D_mesh.h5"
    names = ["Ux", "Uy", "p"]
    with h5py.File(h5, "r") as f:
        xy = f["coordinates"][:, 0, 0, :2].astype(np.float64)
        wall = f["surface_indices"][:].astype(np.int64)
        F = f["fields"]
        N = xy.shape[0]
        s = np.zeros(3); sq = np.zeros(3); cnt = 0
        for a in range(0, 1200, 100):
            x = F[0, a:a + 100, :, 0, 0, :].astype(np.float64)
            s += x.sum((0, 1)); sq += (x ** 2).sum((0, 1)); cnt += x.shape[0] * x.shape[1]
        mean = s / cnt; std = np.sqrt(sq / cnt - mean ** 2)
        tr = np.arange(0, 1200, 30)                       # 40 train frames, 10 per Re
        Ztr = np.stack([(F[0, int(t), :, 0, 0, :] - mean) / std for t in tr]).astype(np.float64)
        te = held_out_test(1200, 1800, 50, blocks=2)
        Zte = np.stack([(F[0, int(t), :, 0, 0, :] - mean) / std for t in te]).astype(np.float64)
    m2 = (Ztr ** 2).mean((0, 1))
    den = (Zte ** 2).mean(1)                               # [frames, C]
    edges = np.concatenate([[0.0], np.geomspace(0.002, 40.0, 240)])
    rc = np.sqrt(np.maximum(edges[1:] * edges[:-1], 1e-12)); rc[0] = edges[1] / 2

    def s2_profile(anchor_idx):
        """Radial S2 between anchor points and all points, averaged over train frames."""
        dd = np.hypot(xy[anchor_idx, 0][:, None] - xy[None, :, 0], xy[anchor_idx, 1][:, None] - xy[None, :, 1])
        b = (np.digitize(dd, edges) - 1).ravel()
        ok = (b >= 0) & (b < len(edges) - 1)
        b = b[ok]
        cntb = np.bincount(b, minlength=len(edges) - 1).astype(np.float64)
        out = np.zeros((3, len(edges) - 1))
        for z in Ztr:
            for c in range(3):
                diff2 = ((z[anchor_idx, c][:, None] - z[None, :, c]) ** 2).ravel()[ok]
                out[c] += np.bincount(b, diff2, minlength=len(edges) - 1)
        with np.errstate(invalid="ignore"):
            prof = out / len(Ztr) / np.maximum(cntb, 1)
        prof[:, cntb == 0] = np.nan
        return prof, cntb

    anchors = RNG.choice(N, 3000, replace=False)
    S2_all, cnt_all = s2_profile(anchors)
    S2_wall, cnt_wall = s2_profile(wall)
    print(f"[cyl] S2 profiles done {time.time()-t0:.0f}s; train m2 {m2}", flush=True)
    ell = {names[c]: ell_from(rc, S2_all[c], 1.0) for c in range(3)}

    def interp(prof_c, d):
        ok = np.isfinite(prof_c)
        return np.interp(d, rc[ok], prof_c[ok])

    rows = []
    vol = "outputs/runs/cylinder2d/baseline_classical/classical_baselines_main_1pct_2d.json"
    srf = "outputs/runs/cylinder2d_surface/baseline_classical/classical_baselines_surface_2d.json"
    tree_all = None
    for task, pool, counts, chans, prof, src in [
            ("cylinder volume", np.arange(N), [238], [0, 1], S2_all, vol),
            ("cylinder surface", wall, [32, 64, 128, 360], [0, 1, 2], S2_wall, srf)]:
        for ns in counts:
            for c in chans:
                num, gap, gmed = [], [], []
                for rep in range(20):
                    sid = RNG.choice(pool, min(ns, len(pool)), replace=False)
                    d, _ = cKDTree(xy[sid]).query(xy, k=1, workers=W)
                    num.append(np.mean(interp(prof[c], d)))
                    gap.append(np.mean(d)); gmed.append(np.median(d))
                num = float(np.mean(num)); g = float(np.mean(gap))
                e = ell[names[c]]
                r = {"setting": f"{task} n={ns}", "field": names[c], "n_sensors": ns,
                     "pool": "all 23800 mesh cells" if task.endswith("volume") else "360 wall-adjacent cells",
                     "mean_gap_cells": None, "mean_gap_phys": g, "median_gap_phys": float(np.mean(gmed)),
                     "ell_cells": None, "ell_phys": e, "gap_over_ell": None if e is None else g / e,
                     "mean_S2_at_gap": num, **pred(num, m2[c], den[:, c]),
                     "eps_nn_measured": classical(src, f"kdtree_n{ns}", names[c]),
                     "eps_idw_measured": classical(src, f"idw_n{ns}", names[c]),
                     "measured_source": src,
                     "S2_profile": "pool-to-query (wall cells to all cells)" if task.endswith("surface") else "cell-to-cell, isotropic"}
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("measured_source",)}, flush=True)
    prof = {"r_D": rc.tolist(), "S2_all": {names[c]: S2_all[c].tolist() for c in range(3)},
            "S2_wall_to_all": {names[c]: S2_wall[c].tolist() for c in range(3)}, "train_mean_square": m2.tolist()}
    print(f"[cyl] done {time.time()-t0:.0f}s ell={ell}", flush=True)
    return rows, prof


if __name__ == "__main__":
    which = sys.argv[1:] or ["kolmogorov", "cylinder", "jhu"]
    for w in which:
        rows, prof = globals()[w]()
        json.dump({"rows": rows, "profile": prof}, open(OUT / f"idw_{w}.json", "w"), indent=1)
