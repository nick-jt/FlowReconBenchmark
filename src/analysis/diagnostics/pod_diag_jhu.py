"""Gappy POD compressibility vs observability on the JHTDB cross-cube task.
Same quantities as pod_diag_2d.py, with the JHU anchor's own pipeline
(methods.classical.anchors_jhu): TurbulentCombustionH5Dataset block split
(train cubes 0-2 = 150 frames, TEST = even positions of cube 3, 25 frames),
z-scoring with the stats the reported anchor JSON records (FNO3D run), basis over
all 150 train frames, 4 channels interleaved, sensors on Ux and Uz only.

    python -m analysis.diagnostics.pod_diag_jhu
updates outputs/paper/diagnostics/pod_jhu.json."""
import json, os, time
import numpy as np
import torch
os.environ.setdefault("JHU_SPLIT_MODE", "block"); os.environ.setdefault("JHU_SPLIT_GAP", "0")
from methods.classical.anchors_jhu import (GappyPOD, load_split, draw_sensors, obs_columns,
                                                  FIELD_NAMES, COND_FIELDS)
from methods.dmfgen.helpers import TurbulentCombustionH5Dataset

from paths import REPO_ROOT as REPO

OUT = REPO / "outputs/paper/diagnostics/pod_jhu.json"
DATA = str(REPO / "data/JHU_4cubes_stride100.h5")
STATS = str(next((REPO / "outputs/runs/JHU/baseline_fno").glob("fno3d_xcube_DemoN90_*")) / "dataset_stats.pt")
J = json.load(open(REPO / "outputs/runs/JHU/baseline_classical/classical_baselines_sweep.json"))
RANKS = [5, 20, 40, 80, 99]
COUNTS = [1953, 19531]

ds_te = TurbulentCombustionH5Dataset(DATA, split="test", train_ratio=0.75, field_names=FIELD_NAMES,
                                     seed=42, time_stride=1, stats_path=STATS)
ds_tr = TurbulentCombustionH5Dataset(DATA, split="train", train_ratio=0.75, field_names=FIELD_NAMES,
                                     seed=42, time_stride=1, stats_path=STATS)
want = np.array(J["protocol"]["standardization"].split("mean=")[1].split("]")[0].strip("[").split(","), float)
assert np.allclose(ds_te.mean.numpy(), want, rtol=1e-6), (ds_te.mean, want)
N, C = ds_te.num_points, ds_te.num_fields
n_snap = J["results"]["gappy_pod_n19531"]["n_snapshots"]
t0 = time.time()
Xtr = load_split(ds_tr, ds_tr.indices).reshape(len(ds_tr), -1)
mu = Xtr.mean(axis=0); Xtr -= mu
pod = GappyPOD(Xtr, mu, rank=max(RANKS))
Y = load_split(ds_te, ds_te.indices[:n_snap]).reshape(n_snap, -1)
print(f"train {Xtr.shape} test {Y.shape} (test frames {list(ds_te.indices[:n_snap])[:4]}...) {time.time()-t0:.0f}s", flush=True)

def per_field_rel(pred, y):
    P = pred.reshape(-1, C); Yv = y.reshape(-1, C)
    return [float(np.linalg.norm(P[:, c] - Yv[:, c]) / (np.linalg.norm(Yv[:, c]) + 1e-12)) for c in range(C)]

proj_c = (Xtr @ (Y - mu).T).astype(np.float64)
sens = {n: [draw_sensors(ds_te.coords, torch.from_numpy(Y[i].reshape(N, C)), snap=i, n_obs=n, seed=0, device="cpu")
            for i in range(n_snap)] for n in COUNTS}
out = {"case": "jhu", "N": N, "C": C, "fields": list(FIELD_NAMES), "cond_fields": COND_FIELDS,
       "n_train_frames": int(len(ds_tr)), "n_test_frames": n_snap, "stats": STATS,
       "energy_curve_r": {str(r): float(pod.energy[r - 1]) for r in [1, 2, 5, 10, 20, 40, 60, 80, 99, 120, 149]},
       "rank_selected_by_anchor": J["timing_and_memory"]["gappy_pod_fit"]["selected_rank"],
       "rank_scan_holdout_train_cube2": J["timing_and_memory"]["gappy_pod_fit"]["rank_scan_holdout_train_cube2"],
       "settings": []}
for r in RANKS:
    pod.set_rank(r)
    W = pod.Ur * pod.inv_s[None, :]
    a_true = W.T @ proj_c
    fpf, ftot, rp = [], [], []
    for j in range(n_snap):
        rec = pod.mu + (W @ a_true[:, j]).astype(np.float32) @ pod.Xc
        rp.append(rec); fpf.append(per_field_rel(rec, Y[j]))
        ftot.append(float(np.linalg.norm(rec - Y[j]) / np.linalg.norm(Y[j])))
    fpf = np.array(fpf)
    base = {"r": r, "energy_fraction_train": float(pod.energy[r - 1]),
            "projection_floor_relL2": float(fpf.mean(axis=1).mean()),
            "projection_floor_relL2_per_field": dict(zip(FIELD_NAMES, fpf.mean(axis=0).round(5).tolist())),
            "projection_floor_total_norm": float(np.mean(ftot))}
    for n in COUNTS:
        conds, smin, smax, smn, gpf, gtot, ctot = [], [], [], [], [], [], []
        for j in range(n_snap):
            cols, vals = obs_columns(sens[n][j], C)
            A = pod.modes_at(cols)
            s = np.linalg.svd(A, compute_uv=False)
            conds.append(s[0] / s[-1]); smin.append(s[-1]); smax.append(s[0]); smn.append(s[-1] * np.sqrt(N * C / len(cols)))
            rec = pod.reconstruct(cols, vals)
            gpf.append(per_field_rel(rec, Y[j])); gtot.append(float(np.linalg.norm(rec - Y[j]) / np.linalg.norm(Y[j])))
            ctot.append(float(np.linalg.norm(rec - rp[j]) / np.linalg.norm(Y[j])))
        gpf = np.array(gpf); meas = J["results"][f"gappy_pod_n{n}"]
        st = dict(base, n_obs_per_observed_field=n, n_rows_HPhi=int(len(cols)),
                  cond_HPhi_median=float(np.median(conds)), cond_HPhi_p10_p90=[float(np.percentile(conds, 10)), float(np.percentile(conds, 90))],
                  sigma_min_median=float(np.median(smin)), sigma_max_median=float(np.median(smax)),
                  sigma_min_normalized_median=float(np.median(smn)),
                  gappy_relL2_cpu_draws=float(gpf.mean(axis=1).mean()),
                  gappy_relL2_cpu_draws_per_field=dict(zip(FIELD_NAMES, gpf.mean(axis=0).round(5).tolist())),
                  gappy_total_norm_cpu=float(np.mean(gtot)), coef_error_total_norm_cpu=float(np.mean(ctot)),
                  anchor_rank_at_this_count=int(meas["rank"]))
        if meas["rank"] == r:
            st["gappy_relL2_measured_canonical"] = meas["aggregate"]["rel_l2_mean"]
            st["gappy_relL2_measured_canonical_per_field"] = {f: meas["per_field"][f]["rel_l2_mean"] for f in meas["per_field"]}
        out["settings"].append(st)
        print(f"[jhu] r={r:3d} n={n:6d} energy {st['energy_fraction_train']:.4f} floor {st['projection_floor_relL2']:.4f} "
              f"{st['projection_floor_relL2_per_field']} cond {st['cond_HPhi_median']:.3g} gappy(cpu) {st['gappy_relL2_cpu_draws']:.4f} "
              f"canon {st.get('gappy_relL2_measured_canonical', float('nan')):.4f} | tot gappy {np.mean(gtot):.4f} coef {np.mean(ctot):.4f} "
              f"floor {base['projection_floor_total_norm']:.4f}", flush=True)
prev = json.load(open(OUT)) if OUT.exists() else {}
prev["jhu"] = out
json.dump(prev, open(OUT, "w"), indent=1)
print("wrote", OUT)
