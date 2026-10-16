"""Dense-sensor limit of gappy POD: every point of every OBSERVED channel is a
sensor. Coefficients from the normal equations, which need only the Gram matrix of
the observed columns (no N-by-r matrix is formed). The gap between this limit and
the projection floor isolates what the observed channels cannot tell the fit about
the unobserved ones; the gap between the actual sensor density and this limit is
the sampling (observability) part.

    python -m analysis.diagnostics.pod_dense_limit {cylinder|jhu} <rank>
updates outputs/paper/diagnostics/pod_dense.json."""
import json, os, sys
import numpy as np
os.environ.setdefault("JHU_SPLIT_MODE", "block"); os.environ.setdefault("JHU_SPLIT_GAP", "0")
from paths import REPO_ROOT as REPO

OUT = REPO / "outputs/paper/diagnostics/pod_dense.json"
case = sys.argv[1]; r = int(sys.argv[2])

if case == "cylinder":
    from methods.classical.anchors_2d import GappyPOD, compute_train_stats, load_split, read_layout
    from protocol import held_out_subset
    h5 = str(REPO / "data/cylinder2d/Cylinder2D_mesh.h5")
    J = json.load(open(REPO / "outputs/runs/cylinder2d/baseline_classical/classical_baselines_main_1pct_2d.json"))["protocol"]
    craw, T, C = read_layout(h5); N = craw.shape[0]
    tr = np.arange(J["train_frames"]); te = held_out_subset(np.arange(J["train_frames"], T), "test")[J["frame_indices"]]
    mean, std = compute_train_stats(h5, tr, C)
    X = load_split(h5, tr, N, C, mean, std).reshape(len(tr), -1); Y = load_split(h5, te, N, C, mean, std).reshape(len(te), -1)
    obs = J["cond_fields"]; names = J["field_names"]
else:
    from methods.classical.anchors_jhu import GappyPOD, load_split, FIELD_NAMES, COND_FIELDS
    from methods.dmfgen.helpers import TurbulentCombustionH5Dataset
    data = str(REPO / "data/JHU_4cubes_stride100.h5")
    st = str(next((REPO / "outputs/runs/JHU/baseline_fno").glob("fno3d_xcube_DemoN90_*")) / "dataset_stats.pt")
    dtr = TurbulentCombustionH5Dataset(data, split="train", train_ratio=0.75, field_names=FIELD_NAMES, seed=42, time_stride=1, stats_path=st)
    dte = TurbulentCombustionH5Dataset(data, split="test", train_ratio=0.75, field_names=FIELD_NAMES, seed=42, time_stride=1, stats_path=st)
    N, C = dte.num_points, dte.num_fields
    X = load_split(dtr, dtr.indices).reshape(len(dtr), -1); Y = load_split(dte, dte.indices[:25]).reshape(25, -1)
    obs = COND_FIELDS; names = list(FIELD_NAMES)
mu = X.mean(axis=0); X -= mu
pod = GappyPOD(X, mu, rank=r)
W = pod.Ur * pod.inv_s[None, :]                                   # modes = W^T X
cols = np.sort(np.concatenate([np.arange(N) * C + f for f in obs]))
Xo = X[:, cols].astype(np.float64)
G = Xo @ Xo.T                                                     # [n_train, n_train]
AtA = W.T @ G @ W
res = []
for j in range(Y.shape[0]):
    a = np.linalg.solve(AtA, W.T @ (Xo @ (Y[j, cols] - mu[cols]).astype(np.float64)))
    rec = (mu + (W @ a).astype(np.float32) @ X).reshape(N, C); y = Y[j].reshape(N, C)
    res.append([float(np.linalg.norm(rec[:, c] - y[:, c]) / np.linalg.norm(y[:, c])) for c in range(C)])
res = np.array(res)
d = {"case": case, "r": r, "dense_limit_relL2": float(res.mean(axis=1).mean()),
     "dense_limit_relL2_per_field": dict(zip(names, res.mean(axis=0).round(5).tolist())),
     "cond_dense_HPhi": float(np.sqrt(np.linalg.cond(AtA)))}
print(d, flush=True)
prev = json.load(open(OUT)) if OUT.exists() else {}
prev[f"{case}_r{r}"] = d
json.dump(prev, open(OUT, "w"), indent=1)
