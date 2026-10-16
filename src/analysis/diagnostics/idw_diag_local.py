"""Location-resolved version of the diagnostic for the inhomogeneous cylinder wake,
plus a JHU check with the structure function measured on the scored cube itself.

Cylinder: with n sensors drawn uniformly without replacement from a pool of P points,
the k-th nearest pool point to x is x's nearest sensor with probability
C(P-k, n-1)/C(P, n). The expected squared nearest-neighbour error at x is the
probability-weighted mean of E_train[(u(x)-u(y_k))^2] over pool points y_k, so

    eps_local = sqrt( mean_x sum_k w_k E_train[(u(x)-u(y_k))^2] / E_train[u^2] ),

which needs only training frames and the sensor count. On a homogeneous grid it reduces
to the structure-function form used in idw_diag.py.

    python -m analysis.diagnostics.idw_diag_local
writes outputs/paper/diagnostics/idw_local.json.
"""
import json

import h5py
import numpy as np
import scipy.fft as sfft
from scipy.spatial import cKDTree
from scipy.special import gammaln

from paths import REPO_ROOT as REPO

OUT = REPO / "outputs/paper/diagnostics"
W = 64


def nearest_weights(P, n, K):
    k = np.arange(1, K + 1)
    lw = (gammaln(P - k + 1) - gammaln(n - 1 + 1) - gammaln(P - k - (n - 1) + 1)) \
        - (gammaln(P + 1) - gammaln(n + 1) - gammaln(P - n + 1))
    w = np.where(P - k >= n - 1, np.exp(lw), 0.0)
    return w / w.sum()


def held_out_test(first, last, n_frames, blocks=1):
    idx = np.arange(first, last)[0::2]
    n = len(idx)
    span, per = n // blocks, n_frames // blocks
    pos = [b * span + (i * span) // per for b in range(blocks) for i in range(per)]
    return idx[pos]


def cylinder():
    h5 = REPO / "data/cylinder2d/Cylinder2D_mesh.h5"
    names = ["Ux", "Uy", "p"]
    with h5py.File(h5, "r") as f:
        xy = f["coordinates"][:, 0, 0, :2].astype(np.float64)
        wall = f["surface_indices"][:].astype(np.int64)
        F = f["fields"]
        s = np.zeros(3); sq = np.zeros(3); cnt = 0
        for a in range(0, 1200, 100):
            x = F[0, a:a + 100, :, 0, 0, :].astype(np.float64)
            s += x.sum((0, 1)); sq += (x ** 2).sum((0, 1)); cnt += x.shape[0] * x.shape[1]
        mean = s / cnt; std = np.sqrt(sq / cnt - mean ** 2)
        tr = np.arange(0, 1200, 15)                  # 80 train frames, 20 per Re
        Z = np.stack([(F[0, int(t), :, 0, 0, :] - mean) / std for t in tr]).astype(np.float32)  # [T, N, C]
    N = xy.shape[0]
    m2 = (Z.astype(np.float64) ** 2).mean((0, 1))
    wake = (xy[:, 0] > 1.0) & (np.abs(xy[:, 1]) < 4.0)
    near = np.hypot(xy[:, 0], xy[:, 1]) < 1.5
    rows = []
    vol = json.load(open(REPO / "outputs/runs/cylinder2d/baseline_classical/classical_baselines_main_1pct_2d.json"))["results"]
    srf = json.load(open(REPO / "outputs/runs/cylinder2d_surface/baseline_classical/classical_baselines_surface_2d.json"))["results"]
    for task, pool, counts, chans, res in [("cylinder volume", np.arange(N), [238], [0, 1], vol),
                                           ("cylinder surface", wall, [32, 64, 128, 360], [0, 1, 2], srf)]:
        Ppool = len(pool)
        K = min(Ppool, 1200)
        dist, kidx = cKDTree(xy[pool]).query(xy, k=K, workers=W)          # [N, K] sorted
        nb = pool[kidx]
        # E_train[(u(x)-u(y_k))^2] for every (x, k), chunked over x
        D2 = np.zeros((len(chans), N, K), dtype=np.float64)
        for c_i, c in enumerate(chans):
            for a in range(0, N, 2000):
                b = min(N, a + 2000)
                d = Z[:, a:b, c][:, :, None] - Z[:, nb[a:b], c]             # [T, chunk, K]
                D2[c_i, a:b] = (d.astype(np.float64) ** 2).mean(0)
        for n in counts:
            w = nearest_weights(Ppool, n, K)
            gap = (dist * w).sum(1)                                           # expected nearest distance per x
            for c_i, c in enumerate(chans):
                per_x = (D2[c_i] * w).sum(1)                                  # expected squared NN error per x
                e = float(np.sqrt(per_x.mean() / m2[c]))
                frac_wake = float(per_x[wake].sum() / per_x.sum())
                r = {"setting": f"{task} n={n}", "field": names[c], "n_sensors": n,
                     "eps_pred_local": e, "eps_nn_measured": res[f"kdtree_n{n}"]["per_field"][names[c]]["rel_l2_mean"],
                     "eps_idw_measured": res[f"idw_n{n}"]["per_field"][names[c]]["rel_l2_mean"],
                     "mean_gap_D": float(gap.mean()), "mean_gap_D_wake": float(gap[wake].mean()),
                     "mean_gap_D_near_body": float(gap[near].mean()),
                     "frac_cells_wake": float(wake.mean()), "frac_sq_error_from_wake": frac_wake}
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    # wake-region correlation length: isotropic S2 over pairs with both ends in the wake box
    wi = np.where(wake)[0]
    rng = np.random.default_rng(1)
    anc = rng.choice(wi, 2000, replace=False)
    edges = np.concatenate([[0.0], np.geomspace(0.01, 20.0, 160)])
    rc = np.sqrt(np.maximum(edges[1:] * edges[:-1], 1e-12)); rc[0] = edges[1] / 2
    dd = np.hypot(xy[anc, 0][:, None] - xy[wi, 0][None], xy[anc, 1][:, None] - xy[wi, 1][None])
    bb = (np.digitize(dd, edges) - 1).ravel(); ok = (bb >= 0) & (bb < len(edges) - 1); bb = bb[ok]
    cntb = np.bincount(bb, minlength=len(edges) - 1)
    ell_w = {}
    m2w = (Z[:, wi].astype(np.float64) ** 2).mean((0, 1))
    for c in range(3):
        acc = np.zeros(len(edges) - 1)
        for z in Z:
            acc += np.bincount(bb, ((z[anc, c][:, None] - z[wi, c][None]) ** 2).ravel()[ok], minlength=len(edges) - 1)
        prof = acc / len(Z) / np.maximum(cntb, 1)
        okp = cntb > 50
        above = np.where(okp & (prof >= m2w[c]))[0]
        ell_w[names[c]] = None if len(above) == 0 else float(rc[above[0]])
        print(f"[cyl wake] {names[c]} m2_wake={m2w[c]:.3f} ell_wake(S2=m2_wake)={ell_w[names[c]]}", flush=True)
    return rows, {"ell_wake_D": ell_w, "wake_box": "x>1, |y|<4 (D units)", "m2_wake": m2w.tolist()}


def jhu_check():
    """S2 measured on the scored cube (cube 3, TEST half) instead of the training cubes."""
    h5 = REPO / "data/JHU_4cubes_stride100.h5"
    n, P = 125, 256
    mean = np.array([0.024666232988238335, -0.11473891884088516, -0.08747481554746628, -0.010348735377192497])
    std = np.array([0.6237276196479797, 0.6208027601242065, 0.5550218224525452, 0.3945266008377075])
    te = held_out_test(150, 200, 25)
    tr = np.concatenate([np.arange(c * 50, c * 50 + 50, 5) for c in range(3)])
    m = np.zeros((P, P, P)); m[:n, :n, :n] = 1
    M = sfft.rfftn(m, workers=W)
    Ncnt = sfft.irfftn(np.conj(M) * M, s=(P, P, P), workers=W)
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(7)
    out = []
    sweep = json.load(open(REPO / "outputs/runs/JHU/baseline_classical/classical_baselines_sweep.json"))["results"]
    with h5py.File(h5, "r") as f:
        Fd = f["fields"]
        for name, c in {"Ux": 0, "Uz": 2}.items():
            stats = {}
            for lab, frames in (("test_cube", te), ("train_cubes", tr)):
                acc = np.zeros((P, P, P)); ms = []
                for t in frames:
                    u = ((Fd[0, int(t), :, 0, 0, c] - mean[c]) / std[c]).astype(np.float64).reshape(n, n, n)
                    up = np.zeros((P, P, P)); up[:n, :n, :n] = u
                    U = sfft.rfftn(up, workers=W); Q = sfft.rfftn(up ** 2, workers=W)
                    Pc = sfft.irfftn(np.conj(U) * U, s=(P, P, P), workers=W)
                    A = sfft.irfftn(np.conj(M) * Q, s=(P, P, P), workers=W)
                    acc += A + np.roll(np.flip(A, axis=(0, 1, 2)), 1, axis=(0, 1, 2)) - 2 * Pc
                    ms.append(np.mean(u ** 2))
                with np.errstate(invalid="ignore", divide="ignore"):
                    stats[lab] = (acc / len(frames) / np.maximum(Ncnt, 0.5), np.array(ms))
            # within-cube variance (mean removed per frame) for context
            for ns in [1953, 19531, 195312]:
                sid = rng.choice(n ** 3, ns, replace=False)
                d, k = cKDTree(g[sid]).query(g, k=1, workers=W)
                disp = (g[sid][k] - g).astype(np.int64) % P
                S2t, mst = stats["test_cube"]
                S2r, msr = stats["train_cubes"]
                num_t = np.mean(S2t[disp[:, 0], disp[:, 1], disp[:, 2]])
                num_r = np.mean(S2r[disp[:, 0], disp[:, 1], disp[:, 2]])
                r = {"field": name, "n_sensors": ns,
                     "eps_pred_testcube_S2_testcube_norm": float(np.mean(np.sqrt(num_t / mst))),
                     "eps_pred_train_S2_train_norm": float(np.sqrt(num_r / msr.mean())),
                     "mean_S2_at_gap_test": float(num_t), "mean_S2_at_gap_train": float(num_r),
                     "test_mean_square": float(mst.mean()), "train_mean_square": float(msr.mean()),
                     "eps_nn_measured": sweep[f"kdtree_n{ns}"]["per_field"][name]["rel_l2_mean"],
                     "eps_idw_measured": sweep[f"idw_n{ns}"]["per_field"][name]["rel_l2_mean"]}
                out.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return out


if __name__ == "__main__":
    rows, extra = cylinder()
    chk = jhu_check()
    json.dump({"cylinder_local": rows, "cylinder_wake": extra, "jhu_testcube_check": chk},
              open(OUT / "idw_local.json", "w"), indent=1)
