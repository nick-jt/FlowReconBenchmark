"""Scattered vs uniform-lattice sensing for the training-free anchors on the
Kolmogorov TEST half (frozen-protocol frames), CPU only.

Scattered: 655 sensors per frame, uniform without replacement, CPU torch draw
seeded as seed*777+snap (statistically equivalent to, not bit-identical with,
the canonical CUDA draws). Lattice: the 25x26 = 650 centered lattice of
eval/litprotocol_2d.py. IDW k=8 and nearest neighbour, non-periodic
distances, exactly the anchors_2d estimators; relative L2 in z-score units.

    python -m analysis.kolm_layout_idw
writes outputs/paper/kolm_layout_idw.json."""
import json
import h5py, numpy as np, torch
from paths import REPO_ROOT
from protocol import frame_list, held_out_subset
from methods.classical.anchors_2d import kd_predict, build_coords_box
from eval.litprotocol_2d import uniform_lattice
from scipy.spatial import cKDTree

H5 = REPO_ROOT / "data/kolmogorov2d/Kolmogorov2D_shu_stride4.h5"
man = json.load(open(REPO_ROOT / "data/kolmogorov2d/kolmogorov2d_manifest.json"))
mu, sd = man["train_stats"]["vorticity"]["mean"], man["train_stats"]["vorticity"]["std"]
f = h5py.File(H5, "r")
craw = f["coordinates"][:, 0, 0, :2].astype(np.float64)
coords, box = build_coords_box(craw, periodic=False)
N = craw.shape[0]
test_idx = held_out_subset(np.arange(2560, 3200), "test")
snaps = frame_list(len(test_idx), 50, 1)
lat = uniform_lattice(256, 256, 25, 26)
dx = 1.0 / 256  # one grid cell in the unit box

def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
def gaps(idx):
    d, _ = cKDTree(coords[idx]).query(coords, k=1, workers=-1)
    d = d / dx
    return d.mean(), np.percentile(d, 99), d.max()

res = {"scattered": {"idw": [], "nn": [], "gap": []}, "lattice": {"idw": [], "nn": [], "gap": []}}
for p in snaps:
    y = (f["fields"][0, int(test_idx[p]), :, 0, 0, 0].astype(np.float64) - mu) / sd
    torch.manual_seed(0 * 777 + int(p))
    sc = torch.randperm(N)[:655].numpy()
    for name, idx in (("scattered", sc), ("lattice", lat)):
        s = {0: (idx, y[idx].astype(np.float32))}
        res[name]["idw"].append(rel(kd_predict(coords, s, 1, "idw", 8, box)[:, 0], y))
        res[name]["nn"].append(rel(kd_predict(coords, s, 1, "nn", 1, box)[:, 0], y))
        res[name]["gap"].append(gaps(idx))
out = {}
for name in res:
    g = np.array(res[name]["gap"])
    out[name] = {"n_sensors": 655 if name == "scattered" else int(lat.size),
                 "idw_rel_l2": float(np.mean(res[name]["idw"])), "nn_rel_l2": float(np.mean(res[name]["nn"])),
                 "gap_mean_cells": float(g[:, 0].mean()), "gap_p99_cells": float(g[:, 1].mean()),
                 "gap_max_cells": float(g[:, 2].mean())}
d = np.array(res["lattice"]["idw"]) - np.array(res["scattered"]["idw"])
out["paired_idw_diff"] = {"mean": float(d.mean()), "frames_improved": int((d < 0).sum()), "n": int(d.size)}
out["frames_h5"] = [int(test_idx[p]) for p in snaps]
print(json.dumps(out, indent=1))
json.dump(out, open(REPO_ROOT / "outputs/paper/kolm_layout_idw.json", "w"), indent=1)
