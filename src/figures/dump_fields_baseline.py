#!/usr/bin/env python3
"""Dump reconstructed FIELDS (truth / pred_mean / pred_sample / pred_std) to
npz for one fixed snapshot, for the baseline methods, matching the snapshot
and sensor draw of the existing qualitative dumps (qual_jhu.npz /
qual_firebench.npz) exactly.

WHY THIS FILE EXISTS (and why it is standalone)
-----------------------------------------------
The env hook ENSEMBLE_NPZ inside model_baseline.visualize_reconstruction_
{latentfm,sit} dumps ensembles, but (a) the deterministic (Senseiver)
visualizer has no such hook, and (b) all visualize_* functions draw sensors
through helpers_baseline.build_sparse_condition, which is NOT RNG-equivalent
to the canonical helpers.build_sparse_condition used by qualitative_jhu.py /
qualitative_firebench.py (the baseline variant draws its per-field count with
torch.randint(..., device=cuda) while helpers.py draws it on CPU, so the
subsequent CUDA randperm diverges; see eval_sit_ensemble.py:25-32).

This driver therefore reproduces the QUAL scripts' sensor protocol
bit-for-bit and drives each baseline's OWN sampler (never a reimplementation):

  sensor layout : torch.manual_seed(<sensor-seed>) immediately before
                  helpers.build_sparse_condition(...)   [canonical import]
                  qual_jhu.npz       : seed 100 + snapshot (=103), fields [0,2],
                                       n_obs [19531]   (1% per channel)
                  qual_firebench.npz : seed 1000 + snapshot (=1003),
                                       fields [0,1,2], n_obs [36772]
  snapshot      : dataset[<snapshot>] on --split val (qual scripts use the
                  val split of the same H5 with the same env split vars)
  samplers      : SiT   -> model_baseline.sit_conditional_sample_points_chunked
                  LFM   -> bundle.model.sample({"obs_value_grid","obs_mask_grid"})
                           (same call as eval_latentfm_ensemble.py)
                  Sens. -> bundle.model(coords, oc, ov, om, ofid)  (deterministic)
  ensemble noise: base = <sensor-seed> * 131; torch.manual_seed(base*10000 + k)
                  before member k (protocol-shaped, documented in meta)

Model loading mirrors eval_sit_ensemble.py / eval_latentfm_ensemble.py /
evaluate_{Gen,Det}_Baseline.py: run_config.yaml + adapter.build_for_training
+ adapter.load_checkpoint + adapter.evaluation_weights (EMA where defined).

Verification against the qual dump (--check-qual):
  * truth corr   : Pearson corr of standardized truth vs qual truth (subsampled)
                   -> ~1.0 iff the same physical snapshot (robust to each
                   run's own dataset_stats.pt standardization).
  * sensor dist  : distance-to-nearest-sensor field recomputed exactly as in
                   the qual scripts and compared to qual's stored `dist`
                   -> max|diff| == 0 iff the identical sensor set was drawn.
Results are printed AND stored in the npz meta; mismatches do not abort the
dump (the fields are still usable, just flagged).

Outputs (npz, float32, PHYSICAL units unless noted):
  truth        [N, C]   ground truth
  pred_mean    [N, C]   ensemble mean (deterministic: the single prediction)
  pred_sample  [N, C]   first ensemble member (deterministic: the prediction)
  pred_std     [N, C]   ensemble std, physical units (deterministic: zeros)
  coords       [N, D]   normalized coords (dataset "coords")
  coords_raw   [N, D]   raw coords when the dataset provides them
  sensor_indices, sensor_field_ids   the drawn sensor set (valid entries)
  names        [C]      field names
  norm_mean, norm_std [C]  the run's standardization (phys = norm*std+mean)
  meta         json string with full provenance + check results

Touches no shared module. Lives in the pof2026-benchmark worktree only.
"""
from __future__ import annotations
from paths import REPO_ROOT, RUNS_DIR, SRC_DIR

import argparse
import json
import os
import sys
import time
from pathlib import Path
import methods.common.config as bconfig
import methods.common.data as bdata
import methods.common.helpers_baseline as hb
import methods.common.registry as bregistry
import methods.sit.adapter as sit_adapter

# The live MAIN checkout's src: run dirs, model_baseline.py and the canonical
# helpers.py the whole campaign runs against. The worktree copy is NOT used
# for imports so the dump runs the exact code the checkpoints were
# trained/evaluated with. Override with DUMP_SRC_DIR.


def check_snapshot_args(args) -> None:
    if (args.snapshot is None) == (args.frame is None):
        raise SystemExit("Pass exactly one of --snapshot / --frame.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Field dump for baseline methods on one snapshot")
    p.add_argument("--method", required=True,
                   choices=["sit", "senseiver", "latent_fm"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best", choices=["best", "last", "budget"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--snapshot", type=int, default=None,
                   help="Index INTO THIS RUN'S split. Mutually exclusive with "
                        "--frame.")
    p.add_argument("--frame", type=int, default=None,
                   help="ABSOLUTE time index in the H5. Use this to pin the "
                        "same physical snapshot across runs whose train_ratio "
                        "differs (qual_jhu val[3] = frame 153; qual_firebench "
                        "val[3] = frame 112 of the merged FB h5).")
    p.add_argument("--sensor-seed", type=int, required=True,
                   help="torch.manual_seed value right before the sensor draw. "
                        "qual_jhu: 100+snapshot; qual_firebench: 1000+snapshot.")
    p.add_argument("--cond-fields", type=int, nargs="+", required=True)
    p.add_argument("--n-obs", type=int, nargs="+", required=True,
                   help="Passed as n_obs_min=n_obs_max; a single value "
                        "broadcasts per conditioned field, exactly as the "
                        "qual scripts pass it.")
    p.add_argument("--K", type=int, default=8,
                   help="Ensemble size for generative methods (senseiver: 1).")
    p.add_argument("--n-steps", type=int, default=None,
                   help="ODE steps. Default: sit -> run config sampling_N; "
                        "latent_fm -> 4 (paper's matched flow-model NFE).")
    p.add_argument("--rechunk", action="store_true",
                   help="SiT only: the per-step re-chunk sampler (a fresh random "
                        "chunk partition at every drift evaluation).")
    p.add_argument("--field-names", nargs="+", default=None,
                   help="Override shared.data.field_names (labels only, "
                        "positional). Older FB run configs have null and fall "
                        "back to wrong combustion labels.")
    p.add_argument("--data-path", default=None,
                   help="Override shared.paths.data_path (needed when the run "
                        "trained from a node-local /tmp staging copy of the "
                        "H5 that no longer exists).")
    p.add_argument("--out", required=True, help="Output .npz path.")
    p.add_argument("--check-qual", default=None,
                   help="Path to qual_{jhu,firebench}.npz for the truth/sensor "
                        "match checks.")
    p.add_argument("--dry-run", action="store_true",
                   help="CPU/login-safe: resolve and validate every path and "
                        "the config, print the effective protocol, exit. "
                        "No torch / CUDA / model loading.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Dry run: no heavy imports, no GPU. Validates everything path-shaped.
# --------------------------------------------------------------------------

def dry_run(args: argparse.Namespace) -> None:
    import yaml
    ok = True

    def check(label, path, required=True):
        nonlocal ok
        exists = Path(path).exists()
        print(f"[dry-run] {label:18s} {'OK ' if exists else 'MISSING'} {path}")
        if required and not exists:
            ok = False
        return exists

    run_dir = Path(args.run_dir).resolve()
    check("run_dir", run_dir)
    check("checkpoint", run_dir / f"{args.ckpt}.pt")
    check("dataset_stats", run_dir / "dataset_stats.pt")
    cfg_path = run_dir / "run_config.yaml"
    if check("run_config", cfg_path):
        cfg = yaml.safe_load(open(cfg_path))
        bm = cfg.get("baseline_model")
        data = cfg.get("shared", {}).get("data", {})
        data_path = args.data_path or cfg.get("shared", {}).get("paths", {}).get("data_path")
        check("data_path", data_path)
        expect = {"sit": "sit", "senseiver": "senseiver", "latent_fm": "latent_fm"}
        if bm != expect[args.method]:
            print(f"[dry-run] baseline_model MISMATCH: config says {bm!r}, "
                  f"--method is {args.method!r}")
            ok = False
        print(f"[dry-run] grid nx,ny,nz = {data.get('num_x')},"
              f"{data.get('num_y')},{data.get('num_z')}  "
              f"fields={data.get('field_names')}")
        if args.method == "latent_fm":
            s2 = cfg.get("latent_fm_params", {}).get("stage2", {})
            cm = s2.get("conditioning", {}).get("cond_mode", "image")
            sm = s2.get("architecture", {}).get("latent_scale_mode", "none")
            needs_fixes = cm == "image_norm" or str(sm) not in ("none", "None")
            print(f"[dry-run] latent_fm cond_mode={cm} latent_scale_mode={sm} "
                  f"needs_lfm_fixes={needs_fixes}")
    if args.check_qual:
        check("check_qual npz", args.check_qual)
    out_parent = Path(args.out).resolve().parent
    print(f"[dry-run] out dir            {out_parent} "
          f"(writable parent exists: {out_parent.parent.exists() or out_parent.exists()})")
    print(f"[dry-run] protocol: method={args.method} split={args.split} "
          f"snapshot={args.snapshot} frame={args.frame} "
          f"sensor_seed={args.sensor_seed} "
          f"cond_fields={args.cond_fields} n_obs={args.n_obs} K={args.K} "
          f"n_steps={args.n_steps}")
    print(f"[dry-run] {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


# --------------------------------------------------------------------------
# Real run
# --------------------------------------------------------------------------





def _fixes_active(cfg: dict) -> bool:
    """True when the run config uses the latent-FM fidelity repairs (masked-mean
    conditioning and/or latent rescaling), which live inside the model now."""
    s2 = cfg.get("latent_fm_params", {}).get("stage2", {})
    cond_mode = str(s2.get("conditioning", {}).get("cond_mode", "image"))
    scale_mode = str(s2.get("architecture", {}).get("latent_scale_mode", "none"))
    return cond_mode == "image_norm" or scale_mode not in ("none", "None")


def relocate_stage1(ae: str, local_root: Path) -> str:
    """Map a checkpoint's baked-in stage-1 path onto this host.

    Checkpoints record the autoencoder by ABSOLUTE path on the machine that
    trained them (origin: /home/ntricard/...). If that path is absent here,
    re-root it under this checkout by its Save_TrainedModel-relative tail --
    the weights are unchanged, only the path is rewritten -- and fail loudly
    rather than silently binding a different autoencoder.
    """
    if Path(ae).exists() or "Save_TrainedModel/" not in ae:
        return ae
    local = RUNS_DIR / ae.split("Save_TrainedModel/", 1)[1]
    if not local.exists():
        raise SystemExit(f"[stage1] recorded autoencoder {ae} is not on this host "
                         f"and no local counterpart exists at {local}")
    print(f"[stage1] relocated\n   from {ae}\n     to {local}", flush=True)
    return str(local)


def main() -> None:
    args = parse_args()
    check_snapshot_args(args)
    if args.rechunk and args.method != "sit":
        raise SystemExit("--rechunk applies to --method sit only.")
    if args.dry_run:
        dry_run(args)

    import numpy as np
    import torch

    from methods.dmfgen.helpers import build_sparse_condition
    from methods.dmfgen.ensemble_eval import require_compute_node
    require_compute_node()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[gpu] {gpu_name}", flush=True)

    run_dir = bconfig.ensure_absolute(args.run_dir)
    cfg = bconfig.validate_and_normalize_config(bconfig.load_yaml(run_dir / "run_config.yaml"))
    if cfg["baseline_model"] != args.method:
        raise SystemExit(f"[cfg] run config baseline_model="
                         f"{cfg['baseline_model']!r} != --method {args.method!r}")
    if args.data_path:
        old = cfg["shared"]["paths"]["data_path"]
        cfg["shared"]["paths"]["data_path"] = args.data_path
        print(f"[cfg] data_path override: {old} -> {args.data_path}", flush=True)
    if args.field_names:
        cfg["shared"]["data"]["field_names"] = list(args.field_names)
        print(f"[cfg] field_names override: {args.field_names}", flush=True)

    ckpt_path = run_dir / f"{args.ckpt}.pt"
    checkpoint = bconfig.safe_torch_load(ckpt_path, map_location="cpu")

    fixes_active = False
    if args.method == "latent_fm":
        cfg["training_stage"] = 2
        if checkpoint.get("ae_checkpoint"):
            ae = relocate_stage1(str(checkpoint["ae_checkpoint"]), REPO_ROOT)
            cfg["latent_fm_params"]["stage2"]["stage1_checkpoint"] = ae
            print(f"[eval] stage1 ckpt {ae}", flush=True)
        fixes_active = _fixes_active(cfg)

    device = bconfig.infer_device(None, cfg["shared"]["device_ids"])
    dataset = bdata.build_dataset(cfg, split=args.split,
                               stats_path=run_dir / "dataset_stats.pt")
    n_fields = dataset.num_fields
    n_pts = dataset.num_points
    field_names = [str(n) for n in dataset.field_names]
    print(f"[eval] split={args.split} n={len(dataset)} n_pts={n_pts} "
          f"fields={field_names} ckpt={ckpt_path} "
          f"epoch={checkpoint.get('epoch')}", flush=True)

    # Resolve the snapshot: --frame pins the ABSOLUTE time index so the same
    # physical snapshot is dumped even when this run's train_ratio (and hence
    # its val list) differs from the run that produced the qual dump.
    ds_frames = np.asarray(dataset.indices)
    if args.frame is not None:
        pos = np.nonzero(ds_frames == int(args.frame))[0]
        if len(pos) == 0:
            raise SystemExit(f"[eval] frame {args.frame} not in this run's "
                             f"{args.split} split (frames {ds_frames.tolist()})")
        snap_idx = int(pos[0])
    else:
        if args.snapshot >= len(dataset):
            raise SystemExit(f"[eval] snapshot {args.snapshot} out of range "
                             f"(split has {len(dataset)})")
        snap_idx = int(args.snapshot)
    frame = int(ds_frames[snap_idx])
    print(f"[eval] snapshot index {snap_idx} of {args.split} split "
          f"= absolute frame {frame}", flush=True)

    adapter = bregistry.get_baseline_adapter(cfg["baseline_model"])
    bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=run_dir,
                                        train_set=dataset, val_set=dataset)
    adapter.load_checkpoint(bundle, checkpoint)

    stage_cfg = bconfig.resolve_stage_config(cfg)

    item = dataset[snap_idx]
    coords = item["coords"].unsqueeze(0).to(device)
    fields = item["fields"].unsqueeze(0).to(device)
    truth_norm = item["fields"].numpy().astype(np.float32)

    # ---- sensor draw: bit-identical to qualitative_{jhu,firebench}.py -------
    torch.manual_seed(args.sensor_seed)
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=list(args.cond_fields),
        n_obs_min=list(args.n_obs), n_obs_max=list(args.n_obs),
    )
    valid = om[0].bool()
    n_sensors = int(om.sum())
    idx_sum = int(oi[0][valid].sum())  # oi is batched [1, M]; index the batch row
    print(f"[sensors] seed={args.sensor_seed} sensors={n_sensors} "
          f"idx_sum={idx_sum}", flush=True)

    # ---- per-method sampling (each model's OWN sampler) ---------------------
    K = 1 if args.method == "senseiver" else int(args.K)
    ens = []
    noise_base = args.sensor_seed * 131
    t0 = time.perf_counter()
    with adapter.evaluation_weights(bundle):
        bundle.model.eval()
        with torch.no_grad():
            if args.method == "senseiver":
                recon = bundle.model(coords, oc, ov, om, ofid)
                ens.append(recon[0].detach().float().cpu().numpy())
                n_steps, ode_solver = 0, "none"
            elif args.method == "sit":
                arch = stage_cfg["architecture"]
                sampling_cfg = stage_cfg["sampling"]
                n_steps = int(args.n_steps if args.n_steps is not None
                              else sampling_cfg["sampling_N"])
                ode_solver = str(sampling_cfg["ode_solver"])
                node_subsample = int(arch.get("node_subsample") or 0)
                if node_subsample <= 0:
                    raise SystemExit("[sit] run is not a point-tokenizer "
                                     "(node_subsample) model; this dump "
                                     "targets SiT-point.")
                for k in range(K):
                    torch.manual_seed(noise_base * 10_000 + k)
                    s = sit_adapter.sit_conditional_sample_points_chunked(
                        net=bundle.model,
                        transport=bundle.components["transport"],
                        coords=coords, obs_coords=oc, obs_values=ov,
                        obs_mask=om, obs_field_ids=ofid, n_fields=n_fields,
                        device=device, n_steps=n_steps,
                        sampler_type=ode_solver, chunk=node_subsample,
                        sigma=float(arch.get("cond_fill_sigma", 0.05)),
                        rechunk=args.rechunk,
                    )
                    ens.append(s[0].detach().float().cpu().numpy())
                    print(f"[sample] member {k + 1}/{K} done "
                          f"({time.perf_counter() - t0:.1f}s)", flush=True)
            else:  # latent_fm
                sampling_cfg = stage_cfg["sampling"]
                n_steps = int(args.n_steps if args.n_steps is not None else 4)
                ode_solver = str(sampling_cfg["ode_solver"])
                num_x = int(cfg["shared"]["data"]["num_x"])
                num_y = int(cfg["shared"]["data"]["num_y"])
                num_z = int(cfg["shared"]["data"]["num_z"])
                gv, gm = hb.build_obs_grid_mask3d(
                    ov, om, ofid, oi, n_fields, n_pts,
                    num_z, num_y, num_x, num_z, num_y, num_x)
                cond = {"obs_value_grid": gv, "obs_mask_grid": gm}
                for k in range(K):
                    torch.manual_seed(noise_base * 10_000 + k)
                    grid = bundle.model.sample(cond, n_steps=n_steps,
                                               ode_solver=ode_solver)
                    ens.append(hb.grid3d_to_pointcloud(grid, num_z, num_y, num_x)
                               [0].detach().float().cpu().numpy())
                    print(f"[sample] member {k + 1}/{K} done "
                          f"({time.perf_counter() - t0:.1f}s)", flush=True)
    ens = np.stack(ens, axis=0).astype(np.float32)  # [K, N, C], standardized
    print(f"[sample] ensemble {ens.shape} in {time.perf_counter() - t0:.1f}s",
          flush=True)

    # ---- physical units -----------------------------------------------------
    mean = dataset.mean.cpu().numpy().reshape(1, -1).astype(np.float32)
    std = dataset.std.cpu().numpy().reshape(1, -1).astype(np.float32)
    truth_phys = truth_norm * std + mean
    pred_mean_phys = ens.mean(0) * std + mean
    pred_sample_phys = ens[0] * std + mean
    pred_std_phys = (ens.std(0) if K > 1
                     else np.zeros_like(ens[0])) * std

    rel_l2 = float(np.linalg.norm(pred_mean_phys - truth_phys)
                   / (np.linalg.norm(truth_phys) + 1e-12))
    print(f"[metric] rel_l2(pred_mean, truth) = {rel_l2:.5f}", flush=True)

    # ---- checks against the qual dump --------------------------------------
    checks = {}
    xyz = item["coords"].numpy()
    if args.check_qual:
        q = np.load(args.check_qual)
        qt = np.asarray(q["truth"], dtype=np.float32)
        if qt.shape != truth_norm.shape:
            checks["truth_corr"] = None
            checks["shape_mismatch"] = [list(qt.shape), list(truth_norm.shape)]
            print(f"[check] SHAPE MISMATCH qual{qt.shape} vs {truth_norm.shape}",
                  flush=True)
        else:
            sub = slice(None, None, 13)
            a = truth_norm[sub].ravel()
            b = qt[sub].ravel()
            corr = float(np.corrcoef(a, b)[0, 1])
            checks["truth_corr"] = corr
            print(f"[check] truth corr vs qual = {corr:.6f} "
                  f"({'MATCH' if corr > 0.9999 else 'MISMATCH'})", flush=True)
        if "dist" in q.files:
            from scipy.spatial import cKDTree
            s_idx = np.unique(oi[0, valid].long().cpu().numpy())
            d, _ = cKDTree(xyz[s_idx]).query(xyz, k=1)
            dq = np.asarray(q["dist"])
            if dq.shape == d.shape:
                mx = float(np.abs(d - dq).max())
                checks["dist_max_absdiff"] = mx
                print(f"[check] sensor-dist max|diff| vs qual = {mx:.3e} "
                      f"({'MATCH' if mx < 1e-6 else 'MISMATCH'})", flush=True)
            else:
                checks["dist_max_absdiff"] = None
                print(f"[check] dist shape mismatch {dq.shape} vs {d.shape}",
                      flush=True)

    # ---- save ---------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "method": args.method,
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "snapshot_index": snap_idx,
        "absolute_frame": frame,
        "data_path": str(cfg["shared"]["paths"]["data_path"]),
        "sensor_seed": int(args.sensor_seed),
        "sensor_draw": "helpers.build_sparse_condition (canonical) under "
                       "torch.manual_seed(sensor_seed); matches "
                       "qualitative_{jhu,firebench}.py",
        "cond_fields": list(args.cond_fields),
        "n_obs": list(args.n_obs),
        "n_sensors": n_sensors,
        "sensor_idx_sum": idx_sum,
        "K": K,
        "n_steps": int(n_steps),
        "ode_solver": ode_solver,
        "sit_sampler": (("rechunk_per_step" if args.rechunk else "chunked")
                        if args.method == "sit" else None),
        "noise_seeding": f"torch.manual_seed({noise_base}*10000+k)",
        "units": "physical (arr = standardized * norm_std + norm_mean)",
        "gpu": gpu_name,
        "lfm_fixes_active": bool(fixes_active),
        "rel_l2_pred_mean": rel_l2,
        "checks_vs_qual": checks,
        "check_qual_file": args.check_qual,
        "env": {k: os.environ.get(k) for k in
                ("JHU_SPLIT_MODE", "JHU_SPLIT_GAP", "SLURM_JOB_ID")},
    }
    save = dict(
        truth=truth_phys.astype(np.float32),
        pred_mean=pred_mean_phys.astype(np.float32),
        pred_sample=pred_sample_phys.astype(np.float32),
        pred_std=pred_std_phys.astype(np.float32),
        coords=xyz.astype(np.float32),
        sensor_indices=oi[0, valid].long().cpu().numpy(),
        sensor_field_ids=ofid[0, valid].long().cpu().numpy(),
        names=np.array(field_names),
        norm_mean=mean.ravel(), norm_std=std.ravel(),
        meta=np.array(json.dumps(meta, indent=1)),
    )
    cr = item.get("coords_raw")
    if cr is not None:
        save["coords_raw"] = cr.numpy().astype(np.float32)
    np.savez_compressed(out, **save)
    print(f"[done] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
