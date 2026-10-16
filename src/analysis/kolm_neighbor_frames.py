"""Model-free 'lookup' reconstruction of the 50 Kolmogorov TEST frames from their
temporal neighbours in the stored (stride-4) data -- what a random frame split
would place in the training set. No sensors are used. Relative L2 in z-score units.

    python -m analysis.kolm_neighbor_frames
writes outputs/paper/kolm_neighbor_frames.json."""
import json, h5py, numpy as np
from paths import REPO_ROOT
from protocol import frame_list, held_out_subset
f = h5py.File(REPO_ROOT / "data/kolmogorov2d/Kolmogorov2D_shu_stride4.h5", "r")
man = json.load(open(REPO_ROOT / "data/kolmogorov2d/kolmogorov2d_manifest.json"))
mu, sd = man["train_stats"]["vorticity"]["mean"], man["train_stats"]["vorticity"]["std"]
X = lambda i: (f["fields"][0, int(i), :, 0, 0, 0].astype(np.float64) - mu) / sd
test_idx = held_out_subset(np.arange(2560, 3200), "test")
frames = [int(test_idx[p]) for p in frame_list(len(test_idx), 50, 1)]
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
copy, interp = [], []
for i in frames:
    t = i % 80                       # position inside its 80-frame trajectory
    y = X(i)
    nb = [j for j in (i - 1, i + 1) if 0 <= (j % 80) == (t + (j - i)) and 0 <= t + (j - i) < 80]
    copy.append(min(rel(X(j), y) for j in nb))
    if len(nb) == 2:
        interp.append(rel(0.5 * (X(nb[0]) + X(nb[1])), y))
print(f"copy nearest stored neighbour: rel L2 {np.mean(copy):.3f} (n={len(copy)})")
print(f"average of both neighbours:   rel L2 {np.mean(interp):.3f} (n={len(interp)} interior frames)")
json.dump({"frames_h5": frames, "copy_nearest_rel_l2": float(np.mean(copy)), "n_copy": len(copy),
           "average_neighbours_rel_l2": float(np.mean(interp)), "n_average": len(interp)},
          open(REPO_ROOT / "outputs/paper/kolm_neighbor_frames.json", "w"), indent=1)
