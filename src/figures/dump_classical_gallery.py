#!/usr/bin/env python3
"""Classical-method field dumps for the 2D reconstruction galleries
(figs_pof/fig_recon_gallery_{kolmogorov,cylinder}.py): kolm_classical.npz and
cyl_classical.npz in Save_TrainedModel_pof/field_dumps/.

Protocol = the learned dumps' (launcher case dump_2d_gallery): ONE canonical
frame of the TEST half (Kolmogorov position 128, cylinder 150; both are scored
frames of the 50-frame fleet), sensors via helpers.build_sparse_condition under
torch.manual_seed(seed*777+snap) with snap = position in the test half; bit-identical to the fleet only when
drawn on CUDA on the same GPU SKU (CPU draw => meta.sensor_source marks APPROX
and the figures star the panel). Predictions are in z-units of the TRAIN-split
stats (norm_mean/norm_std shipped in the npz), like the learned dumps.
Reuses baseline_classical_2d.py's estimators verbatim.
"""
from paths import DATA_DIR, REPO_ROOT

import argparse, json, os, time
from pathlib import Path
import numpy as np, torch

import methods.classical.anchors_2d as C
from protocol import PROTOCOL_2D, held_out_subset

PRESETS = {
    "kolmogorov2d": dict(h5=f"{DATA_DIR}/kolmogorov2d/Kolmogorov2D_shu_stride4.h5",
                         fields=["vorticity"], cond=[0], train_frames=2560, snap=128, n_obs=655,
                         ranks=[160], periodic=False, tag="kolm_classical"),
    "cylinder2d":   dict(h5=f"{DATA_DIR}/cylinder2d/Cylinder2D_mesh.h5",
                         fields=["Ux", "Uy", "p"], cond=[0, 1], train_frames=1200, snap=150, n_obs=238,
                         ranks=[20], periodic=False, tag="cyl_classical"),
    "cylinder2d_surface": dict(h5=f"{DATA_DIR}/cylinder2d/Cylinder2D_mesh.h5",
                         fields=["Ux", "Uy", "p"], cond=[0, 1, 2], train_frames=1200, snap=150, n_obs=128,
                         ranks=[5], periodic=False, tag="cylsurf_classical", sensor_pool="surface"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/pof/field_dumps"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--idw-k", type=int, default=8)
    ap.add_argument("--sensor-device", default="cuda:0")
    ap.add_argument("--snap", type=int, default=None, help="override the canonical test-half frame")
    ap.add_argument("--n-obs", type=int, default=None)
    ap.add_argument("--sensor-indices-npz", default=None,
                    help="INJECT the sensor set recorded in this npz "
                         "(sensor_indices, sensor_field_ids) instead of drawing. "
                         "CUDA randperm is not portable across GPU SKUs at every "
                         "size -- the 23,800-point cylinder mesh draws differently "
                         "on GH200 than on the H100 the fleet used -- so reusing "
                         "recorded indices is the only way to reproduce another "
                         "machine's draw.")
    a = ap.parse_args()
    P = PRESETS[a.dataset]
    snap = P["snap"] if a.snap is None else a.snap
    n_obs = P["n_obs"] if a.n_obs is None else a.n_obs
    dev = a.sensor_device if torch.cuda.is_available() else "cpu"
    approx = not dev.startswith("cuda")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    craw, T, Cn = C.read_layout(P["h5"]); N = craw.shape[0]
    assert Cn == len(P["fields"])
    train_idx = np.arange(P["train_frames"])
    test_idx = held_out_subset(np.arange(P["train_frames"], T), "test")   # snap = position in the TEST half
    mean, std = C.compute_train_stats(P["h5"], train_idx, Cn)
    coords_box, boxsize = C.build_coords_box(craw, periodic=P["periodic"])
    coords_t = torch.from_numpy(coords_box.astype(np.float32))
    abs_frame = int(test_idx[snap])
    Y = C.load_split(P["h5"], np.array([abs_frame]), N, Cn, mean, std)[0]   # [N, C] z-units

    valid_mask = None
    if P.get("sensor_pool") == "surface":
        import h5py
        with h5py.File(P["h5"], "r") as f:
            pool = np.asarray(f["surface_indices"][:], dtype=np.int64)
        valid_mask = torch.zeros(N, dtype=torch.bool); valid_mask[torch.from_numpy(pool)] = True
    if a.sensor_indices_npz:
        R = np.load(a.sensor_indices_npz, allow_pickle=False)
        ridx = np.asarray(R["sensor_indices"], dtype=np.int64)
        rfid = np.asarray(R["sensor_field_ids"], dtype=np.int64)
        sens = {int(f): (ridx[rfid == f], Y[ridx[rfid == f], int(f)].astype(np.float32))
                for f in P["cond"]}
        injected = f"INJECTED from {Path(a.sensor_indices_npz).name} (recorded draw, not redrawn)"
    else:
        sens = C.draw_sensors(coords_t, torch.from_numpy(Y), snap=snap, n_obs=n_obs, seed=a.seed,
                              device=dev, cond_fields=P["cond"], valid_mask=valid_mask)
        injected = None
    s_idx = np.concatenate([sens[f][0] for f in P["cond"]]); s_fid = np.concatenate([np.full(sens[f][0].size, f) for f in P["cond"]])
    print(f"[sensors] snap={snap} abs={abs_frame} n={s_idx.size} idx_sum={int(s_idx.sum())} device={dev}", flush=True)

    save = dict(truth=Y.astype(np.float32), coords_raw=craw.astype(np.float32),
                norm_mean=mean.astype(np.float32), norm_std=std.astype(np.float32),
                sensor_indices=s_idx.astype(np.int64), sensor_field_ids=s_fid.astype(np.int64))
    timing = {}
    for name, mode in (("kdtree", "nn"), ("idw", "idw")):
        t0 = time.perf_counter(); save[f"pred_{name}"] = C.kd_predict(coords_box, sens, Cn, mode, a.idw_k, boxsize).astype(np.float32)
        timing[name] = time.perf_counter() - t0
    Xtr = C.load_split(P["h5"], train_idx, N, Cn, mean, std).reshape(len(train_idx), -1); mu = Xtr.mean(axis=0); Xtr -= mu
    for r in P["ranks"]:
        pod = C.GappyPOD(Xtr, mu, rank=r); cols, vals = C.obs_columns(sens, Cn)
        t0 = time.perf_counter(); save[f"pred_gappy_pod_r{r}"] = pod.reconstruct(cols, vals).reshape(N, Cn).astype(np.float32)
        timing[f"gappy_pod_r{r}"] = time.perf_counter() - t0
    rel = {k: [float(np.linalg.norm(v[:, j] - Y[:, j]) / (np.linalg.norm(Y[:, j]) + 1e-12)) for j in range(Cn)]
           for k, v in save.items() if k.startswith("pred_")}
    meta = dict(protocol=f"{PROTOCOL_2D}_dump", dataset=a.dataset, data=P["h5"], split="test", split_len=len(test_idx),
                snapshot_index=snap, absolute_frame=abs_frame, seed=a.seed, n_obs=n_obs, cond_fields=P["cond"],
                sensor_pool=P.get("sensor_pool"), n_sensors=int(s_idx.size), sensor_idx_sum=int(s_idx.sum()),
                sensor_source=injected or ("helpers.build_sparse_condition CUDA draw, torch.manual_seed(seed*777+snap)" if not approx
                               else "APPROX: CPU draw (torch.randperm differs from the CUDA fleet draw)"),
                units="STANDARDIZED (train-split z-score; physical = arr*norm_std+norm_mean)", idw_k=a.idw_k,
                field_names=P["fields"], rel_l2_per_field=rel, infer_seconds=timing, gpu=(torch.cuda.get_device_name(0) if not approx else None))
    save["meta"] = np.array(json.dumps(meta, indent=1)); save["names"] = np.array(P["fields"])
    path = out / f"{P['tag']}.npz"; np.savez_compressed(path, **save)
    print(json.dumps(rel, indent=1)); print(f"[out] wrote {path} ({path.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
