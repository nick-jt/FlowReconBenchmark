"""Kinematic oracles for the unobserved JHTDB channels U_y and p.

Bounds what kinematics alone recovers on the held-out 125^3 cube, with the
truth supplied everywhere in place of sensors:

* U_y from continuity. Given the true U_x and U_z, dU_y/dy = -(dU_x/dx + dU_z/dz)
  fixes U_y up to its y-average, a function of (x, z). The oracle integrates the
  divergence along y and sets that average to the training mean, which is all a
  method without U_y data can do.
* p from the pressure Poisson equation lap(p) = -du_i/dx_j du_j/dx_i. Given the
  true velocity, p is fixed up to a harmonic function set by the boundary
  pressure. The oracle solves with the boundary at the training mean; the solve
  with the true boundary is the discretization check.

Errors are relative L2 of fields z-scored with the canonical training statistics
(the metric of the main tables), over the 25 TEST frames (absolute 150, 152, ...,
198). Second-order central differences; homogeneous-Dirichlet DST-I Poisson solve
on the interior, with the boundary values moved to the right-hand side.

  python -m analysis.kinematic_oracle_jhu
Writes outputs/paper/figures/kinematic_oracle_jhu.json.
"""
import argparse
import json

import h5py
import numpy as np
import torch
from scipy.fft import dstn, idstn

from paths import DATA_DIR, PAPER_DIR, RUNS_DIR

N = 125
TEST_FRAMES = list(range(150, 200, 2))
STATS = RUNS_DIR / "JHU/baseline_fno/fno3d_xcube_DemoN90_20261003_145236/dataset_stats.pt"


def rel_l2(pred, true, mu, sd):
    return float(np.linalg.norm((pred - true) / sd) / np.linalg.norm((true - mu) / sd))


def uy_oracle(ux, uz, dx, mu_y):
    """U_y from continuity, y-average set to the training mean. Arrays are (x, y, z)."""
    div = np.gradient(ux, dx, axis=0) + np.gradient(uz, dx, axis=2)
    uy = -np.cumsum(div, axis=1) * dx
    return uy - uy.mean(axis=1, keepdims=True) + mu_y


def poisson_dirichlet(rhs, boundary, dx):
    """Solve lap(p) = rhs on the interior with p = boundary on the faces."""
    r = rhs[1:-1, 1:-1, 1:-1].copy()
    r[0] -= boundary[0, 1:-1, 1:-1] / dx**2
    r[-1] -= boundary[-1, 1:-1, 1:-1] / dx**2
    r[:, 0] -= boundary[1:-1, 0, 1:-1] / dx**2
    r[:, -1] -= boundary[1:-1, -1, 1:-1] / dx**2
    r[:, :, 0] -= boundary[1:-1, 1:-1, 0] / dx**2
    r[:, :, -1] -= boundary[1:-1, 1:-1, -1] / dx**2
    m = N - 2
    lam = (2 * np.cos(np.pi * np.arange(1, m + 1) / (m + 1)) - 2) / dx**2
    eig = lam[:, None, None] + lam[None, :, None] + lam[None, None, :]
    p = boundary.copy()
    p[1:-1, 1:-1, 1:-1] = idstn(dstn(r, type=1) / eig, type=1)
    return p


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    stats = torch.load(STATS, weights_only=False)
    mu, sd = stats["mean"].numpy(), stats["std"].numpy()

    f = h5py.File(DATA_DIR / "JHU_4cubes_stride100.h5", "r")
    xyz = f["coordinates"][:, 0, 0, :]
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))      # grid axes (x, y, z)
    dx = float(np.diff(np.unique(xyz[:, 0]))[0])

    res = {}
    for fr in TEST_FRAMES:
        a = f["fields"][0, fr, :, 0, 0, :][order].astype(np.float64).reshape(N, N, N, 4)
        ux, uy, uz, p = (a[..., c] for c in range(4))
        uy_hat = uy_oracle(ux, uz, dx, mu[1])
        ymean = np.broadcast_to(uy.mean(axis=1, keepdims=True), uy.shape)
        u = (ux, uy, uz)
        grad = [[np.gradient(u[i], dx, axis=j) for j in range(3)] for i in range(3)]
        rhs = -sum(grad[i][j] * grad[j][i] for i in range(3) for j in range(3))
        res[fr] = {
            "uy_oracle": rel_l2(uy_hat, uy, mu[1], sd[1]),
            "uy_ymean_share": float(np.linalg.norm(ymean - mu[1]) / np.linalg.norm(uy - mu[1])),
            "uy_oracle_excl_ymean": float(np.linalg.norm(uy_hat - mu[1] - (uy - ymean))
                                          / np.linalg.norm(uy - mu[1])),
            "p_oracle": rel_l2(poisson_dirichlet(rhs, np.full_like(p, mu[3]), dx), p, mu[3], sd[3]),
            "p_true_boundary": rel_l2(poisson_dirichlet(rhs, p, dx), p, mu[3], sd[3]),
        }
        print(fr, json.dumps(res[fr]), flush=True)

    keys = res[TEST_FRAMES[0]].keys()
    out = {"test_mean": {k: float(np.mean([res[fr][k] for fr in TEST_FRAMES])) for k in keys},
           "per_frame": res, "stats": str(STATS), "frames": TEST_FRAMES}
    path = PAPER_DIR / "figures" / "kinematic_oracle_jhu.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["test_mean"]), "->", path)


if __name__ == "__main__":
    main()
