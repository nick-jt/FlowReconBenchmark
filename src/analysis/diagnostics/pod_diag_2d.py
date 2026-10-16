"""Gappy POD compressibility vs observability on the 2D regimes.

Reuses the anchor's own pipeline (methods.classical.anchors_2d): TRAIN-split
z-scoring, centring on the train mean, method-of-snapshots basis over all train
frames with channels interleaved (column = point*C + channel), and the TEST frames
the anchor JSON scored. For each (rank r, sensor setting):

  energy      cumulative train energy of the first r modes (anchor's own curve)
  floor       relative L2 of projecting each TEST frame onto mu + span(Phi_r):
              the best any coefficient fit can reach with that basis
  cond        sigma_max / sigma_min of H Phi_r (rows of Phi_r at the sensors)
  gappy       relative L2 of the least-squares fit on the sensors (same draw rule,
              CPU randperm: statistically equivalent to, not bit-identical with,
              the canonical CUDA draws; the canonical number is read from the JSON)

Relative L2 follows ensemble_metrics: per field ||pred-y||/||y|| in z-units, aggregate
= mean over fields, then mean over frames. Because gappy - proj lies in span(Phi_r)
and y - proj is orthogonal to it, the all-channel error splits exactly:
  e_gappy^2 = e_coef^2 + e_floor^2  (total-norm relative errors, reported too).

    python -m analysis.diagnostics.pod_diag_2d cylinder cylinder_surface kolmogorov
updates outputs/paper/diagnostics/pod_2d.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

from methods.classical.anchors_2d import (GappyPOD, compute_train_stats, load_split,
                                                 read_layout, draw_sensors, obs_columns)
from protocol import held_out_subset

from paths import REPO_ROOT as REPO

OUT = REPO / "outputs/paper/diagnostics/pod_2d.json"
CASES = {
    "cylinder": dict(h5=REPO / "data/cylinder2d/Cylinder2D_mesh.h5",
                     json=REPO / "outputs/runs/cylinder2d/baseline_classical/classical_baselines_main_1pct_2d.json"),
    "cylinder_surface": dict(h5=REPO / "data/cylinder2d/Cylinder2D_mesh.h5",
                             json=REPO / "outputs/runs/cylinder2d_surface/baseline_classical/classical_baselines_surface_2d.json"),
    "kolmogorov": dict(h5=REPO / "data/kolmogorov2d/Kolmogorov2D_shu_stride4.h5",
                       json=REPO / "outputs/runs/kolmogorov2d/baseline_classical/classical_baselines_main_1pct_periodic_2d.json"),
}
# (rank list, sensor counts) per case; the anchor's chosen rank is always included
PLAN = {
    "cylinder": ([5, 10, 20, 40, 80], [238]),
    "cylinder_surface": ([5, 10, 20, 40], [128, 360]),
    "kolmogorov": ([20, 80, 160, 320], [655]),
}


def per_field_rel(pred, y, C):
    P = pred.reshape(-1, C); Y = y.reshape(-1, C)
    return [float(np.linalg.norm(P[:, c] - Y[:, c]) / (np.linalg.norm(Y[:, c]) + 1e-12)) for c in range(C)]


def run(case):
    cfg = CASES[case]; ranks, counts = PLAN[case]
    J = json.load(open(cfg["json"])); pr = J["protocol"]
    C = len(pr["field_names"]); cond_fields = pr["cond_fields"]; train_frames = pr["train_frames"]
    craw, T, F = read_layout(str(cfg["h5"])); N = craw.shape[0]; assert F == C
    train_idx = np.arange(train_frames)
    test_idx = held_out_subset(np.arange(train_frames, T), "test")
    snaps = pr["frame_indices"]
    mean, std = compute_train_stats(str(cfg["h5"]), train_idx, C)
    want = np.array(pr["standardization"].split("mean=")[1].split("]")[0].strip("[").split(","), float)
    assert np.allclose(mean, want, rtol=1e-5, atol=1e-7), (mean, want)
    t0 = time.time()
    Xtr = load_split(str(cfg["h5"]), train_idx, N, C, mean, std).reshape(len(train_idx), -1)
    mu = Xtr.mean(axis=0); Xtr -= mu
    pod = GappyPOD(Xtr, mu, rank=max(ranks))
    Y = load_split(str(cfg["h5"]), test_idx[snaps], N, C, mean, std).reshape(len(snaps), -1)
    print(f"[{case}] train {Xtr.shape} test {Y.shape} loaded+basis {time.time()-t0:.0f}s", flush=True)

    valid_mask = None
    if pr.get("sensor_pool") == "surface":
        import h5py
        with h5py.File(cfg["h5"], "r") as f:
            pool = np.asarray(f["surface_indices"][:], np.int64) if "surface_indices" in f else \
                np.load(Path(cfg["h5"]).with_suffix(".surface_indices.npy")).astype(np.int64)
        valid_mask = torch.zeros(N, dtype=torch.bool); valid_mask[torch.from_numpy(pool)] = True
    coords_t = torch.from_numpy(craw.astype(np.float32))

    # projection coefficients of every TEST frame onto the full basis
    proj_c = (Xtr @ (Y - mu).T).astype(np.float64)          # [n_train, n_test]  = Xc (y-mu)
    out = {"case": case, "N": N, "C": C, "fields": pr["field_names"], "cond_fields": cond_fields,
           "n_train_frames": int(len(train_idx)), "n_test_frames": len(snaps),
           "energy_curve_r": {str(r): float(pod.energy[r - 1]) for r in sorted(set(ranks + [1, 2, 5, 10, 20, 40, 80, 160, 320]))
                              if r <= len(pod.energy)},
           "rank_selected_by_anchor": pr["pod_rank_by_n_obs"],
           "settings": []}
    sens_cache = {}
    for n_obs in counts:
        sens_cache[n_obs] = [draw_sensors(coords_t, torch.from_numpy(Y[j].reshape(N, C)), snap=i, n_obs=n_obs,
                                          seed=pr["seed"], device="cpu", cond_fields=cond_fields,
                                          valid_mask=valid_mask) for j, i in enumerate(snaps)]
    for r in ranks:
        pod.set_rank(r)
        W = pod.Ur * pod.inv_s[None, :]                       # [n_train, r]
        a_true = (W.T @ proj_c)                               # [r, n_test]  modal coefficients of the TEST frames
        floor_pf, floor_tot = [], []
        recon_proj = []
        for j in range(len(snaps)):
            rec = pod.mu + (W @ a_true[:, j]).astype(np.float32) @ pod.Xc
            recon_proj.append(rec)
            floor_pf.append(per_field_rel(rec, Y[j], C))
            floor_tot.append(float(np.linalg.norm(rec - Y[j]) / np.linalg.norm(Y[j])))
        floor_pf = np.array(floor_pf)
        base = {"r": r, "energy_fraction_train": float(pod.energy[r - 1]),
                "projection_floor_relL2": float(floor_pf.mean(axis=1).mean()),
                "projection_floor_relL2_per_field": dict(zip(pr["field_names"], floor_pf.mean(axis=0).round(5).tolist())),
                "projection_floor_total_norm": float(np.mean(floor_tot))}
        if case.startswith("cylinder"):
            lo = floor_pf[:25].mean(axis=1).mean(); hi = floor_pf[25:].mean(axis=1).mean()
            base["projection_floor_relL2_Re80_Re250"] = [float(lo), float(hi)]
        for n_obs in counts:
            if r > n_obs * len(cond_fields):
                continue
            conds, smin, smax, smin_norm, gp_pf, gp_tot, coef_tot = [], [], [], [], [], [], []
            for j, i in enumerate(snaps):
                cols, vals = obs_columns(sens_cache[n_obs][j], C)
                A = pod.modes_at(cols)                           # [M, r]
                s = np.linalg.svd(A, compute_uv=False)
                conds.append(s[0] / s[-1]); smin.append(s[-1]); smax.append(s[0])
                smin_norm.append(s[-1] * np.sqrt(N * C / len(cols)))
                rec = pod.reconstruct(cols, vals)
                gp_pf.append(per_field_rel(rec, Y[j], C))
                gp_tot.append(float(np.linalg.norm(rec - Y[j]) / np.linalg.norm(Y[j])))
                coef_tot.append(float(np.linalg.norm(rec - recon_proj[j]) / np.linalg.norm(Y[j])))
            gp_pf = np.array(gp_pf)
            key = f"gappy_pod_n{n_obs}"
            meas = J["results"].get(key, {})
            st = dict(base, n_obs_per_observed_field=n_obs, n_rows_HPhi=int(len(cols)),
                      cond_HPhi_median=float(np.median(conds)), cond_HPhi_p10_p90=[float(np.percentile(conds, 10)), float(np.percentile(conds, 90))],
                      sigma_min_median=float(np.median(smin)), sigma_max_median=float(np.median(smax)),
                      sigma_min_normalized_median=float(np.median(smin_norm)),
                      gappy_relL2_cpu_draws=float(gp_pf.mean(axis=1).mean()),
                      gappy_relL2_cpu_draws_per_field=dict(zip(pr["field_names"], gp_pf.mean(axis=0).round(5).tolist())),
                      gappy_total_norm_cpu=float(np.mean(gp_tot)), coef_error_total_norm_cpu=float(np.mean(coef_tot)),
                      anchor_rank_at_this_count=int(pr["pod_rank_by_n_obs"].get(str(n_obs), -1)))
            if case.startswith("cylinder"):
                st["gappy_relL2_cpu_Re80_Re250"] = [float(gp_pf[:25].mean(axis=1).mean()), float(gp_pf[25:].mean(axis=1).mean())]
            if meas and meas.get("rank") == r:
                st["gappy_relL2_measured_canonical"] = meas["aggregate"]["rel_l2_mean"]
                st["gappy_relL2_measured_canonical_per_field"] = {f: meas["per_field"][f]["rel_l2_mean"] for f in meas["per_field"]}
                st["rank_scan_holdout"] = meas.get("rank_scan_holdout")
            out["settings"].append(st)
            print(f"[{case}] r={r:3d} n={n_obs:4d} energy {st['energy_fraction_train']:.4f} floor {st['projection_floor_relL2']:.4f} "
                  f"cond {st['cond_HPhi_median']:.3g} gappy(cpu) {st['gappy_relL2_cpu_draws']:.4f} "
                  f"canon {st.get('gappy_relL2_measured_canonical', float('nan')):.4f} | tot: gappy {np.mean(gp_tot):.4f} "
                  f"coef {np.mean(coef_tot):.4f} floor {base['projection_floor_total_norm']:.4f}", flush=True)
        if not any(s["r"] == r for s in out["settings"]):
            out["settings"].append(base)
    return out


if __name__ == "__main__":
    res = {c: run(c) for c in sys.argv[1:]}
    prev = json.load(open(OUT)) if OUT.exists() else {}
    prev.update(res)
    json.dump(prev, open(OUT, "w"), indent=1)
    print("wrote", OUT)
