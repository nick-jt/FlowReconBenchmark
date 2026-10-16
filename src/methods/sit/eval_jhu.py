#!/usr/bin/env python3
"""Seeded posterior-ensemble evaluation for the SiT-point baseline.

WHY THIS FILE EXISTS
--------------------
`ensemble_eval.py`'s CLI driver (`main()`) cannot evaluate a SiT run: its
`load_run()` reads `<run>/args.json` (SiT runs write `run_config.yaml`),
calls `evaluate_ffm._build_model` (builds a PointCloudFFM), and its
`sample_ensemble()` calls `model.sample_source(...)` and
`model.model.n_fields`, neither of which exists on `SiTPhysics`. Only its
*metric function*, `ensemble_eval.ensemble_metrics`, is model-agnostic, and
that is what the shared protocol pins.

This driver therefore reproduces `ensemble_eval.main()`'s SEEDING SEMANTICS
exactly, so sensor layouts and snapshot selection are bit-identical to the
canonical path, while driving SiT's own sampler:

  snapshot selection : rng = np.random.default_rng(seed)
                       snap_ids = rng.choice(len(dataset), n_snapshots, replace=False)
  sensor layout      : torch.manual_seed(seed * 777 + snap)  immediately before
                       build_sparse_condition(...)
  ensemble noise     : per-snapshot base = seed * 131 + si, then
                       torch.manual_seed(base * 10_000 + k) before sample k

IMPORTANT -- it imports `build_sparse_condition` from **helpers.py**, the same
implementation `ensemble_eval.py` uses, NOT the `helpers_baseline.py` variant
that `model_baseline.py` binds. The two are not RNG-equivalent: the baseline
variant draws its per-field count with `torch.randint(..., device=device)`
(CUDA generator) whereas helpers.py draws it on CPU, so the following
`torch.randperm(n_pts, device=cuda)` starts from a different CUDA RNG state
and yields a DIFFERENT sensor set for the same seed. Importing the canonical
one is what makes the layouts actually match.

Touches no shared module.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from methods.dmfgen.helpers import build_sparse_condition
from methods.dmfgen.ensemble_eval import (check_canonical_fingerprint, ensemble_metrics, jhu_band_metrics,
                                      require_compute_node, save_ensemble_figure)
from methods.common.config import ensure_absolute, infer_device, load_yaml, resolve_stage_config, safe_torch_load, validate_and_normalize_config
from methods.common.data import build_dataset
from methods.common.registry import get_baseline_adapter
from methods.sit.adapter import sit_conditional_sample_points_chunked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Seeded ensemble eval for the SiT-point baseline")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=None, help="Default <run-dir>/run_config.yaml")
    p.add_argument("--ckpt", default="best", choices=["best", "last"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0, 2])
    p.add_argument("--n-obs", type=int, nargs="+", default=[19531, 19531])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op-seed", type=int, default=1000)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--rechunk", action="store_true",
                   help="Per-step sampler: integrate all points as one ODE state and redraw the "
                        "chunk partition at every drift evaluation (stamped as sit_sampler).")
    p.add_argument("--parity", default="all", choices=["all", "odd", "even"],
                   help="Keep only the odd (TUNE) or even (TEST) held-out snapshots of the seeded "
                        "draw; seeds stay tied to the position in the full draw.")
    p.add_argument("--band-metrics", action="store_true",
                   help="Also store band energy / correlation of the members (observed channels).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_compute_node()
    run_dir = ensure_absolute(args.run_dir)
    cfg_path = ensure_absolute(args.config) if args.config else run_dir / "run_config.yaml"
    cfg = validate_and_normalize_config(load_yaml(cfg_path))
    stage_cfg = resolve_stage_config(cfg)
    arch = stage_cfg["architecture"]
    sampling_cfg = stage_cfg["sampling"]
    n_steps = int(args.n_steps if args.n_steps is not None else sampling_cfg["sampling_N"])
    sampler_type = str(sampling_cfg["ode_solver"])
    node_subsample = int(arch.get("node_subsample") or 0)

    device = infer_device(None, cfg["shared"]["device_ids"])
    checkpoint = safe_torch_load(run_dir / (args.ckpt + ".pt"), map_location="cpu")
    dataset = build_dataset(cfg, split=args.split, stats_path=run_dir / "dataset_stats.pt")
    adapter = get_baseline_adapter(cfg["baseline_model"])
    bundle = adapter.build_for_training(
        cfg=cfg, device=device, run_dir=run_dir, train_set=dataset, val_set=dataset,
    )
    adapter.load_checkpoint(bundle, checkpoint)

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / ("Evaluation_seeded_" + args.ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- canonical snapshot selection (identical to ensemble_eval.main) -------
    rng = np.random.default_rng(args.seed)
    snap_ids = rng.choice(len(dataset), size=min(args.n_snapshots, len(dataset)),
                          replace=False)
    print(f"[seedcheck] n_dataset={len(dataset)} n_snapshots={len(snap_ids)} "
          f"seed={args.seed} op_seed={args.op_seed} K={args.K} n_steps={n_steps} "
          f"cond_fields={args.cond_fields} n_obs={args.n_obs}", flush=True)

    with adapter.evaluation_weights(bundle):
        bundle.model.eval()
        for si, snap in enumerate(snap_ids):
            if si % args.num_shards != args.shard:
                continue
            snap = int(snap)
            if args.parity != "all" and snap % 2 != (args.parity == "odd"):
                continue
            out_path = out_dir / f"crps_snap{snap}.json"
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"[skip] snap {snap} already done", flush=True)
                continue

            item = dataset[snap]
            coords = item["coords"].unsqueeze(0).to(device)
            fields = item["fields"].unsqueeze(0).to(device)

            # --- canonical sensor layout ---------------------------------
            torch.manual_seed(args.seed * 777 + snap)
            oc, ov, om, oi, ofid = build_sparse_condition(
                coords_full=coords, fields_full=fields,
                cond_fields=args.cond_fields,
                n_obs_min=args.n_obs, n_obs_max=args.n_obs,
            )
            # op_seed generator: no-op for this protocol (noise=0, no occlusion,
            # no field dropout) but constructed so the contract is explicit.
            _ = torch.Generator(device=ov.device).manual_seed(args.op_seed + snap)
            print(f"[seedcheck] snap={snap} sensors={int(om.sum())} "
                  f"idx_sum={int(oi[om.bool()].sum())}", flush=True)
            check_canonical_fingerprint(snap, int(om.sum()), int(oi[om.bool()].sum()),
                                        args.seed, args.cond_fields, args.n_obs)

            # --- canonical ensemble noise seeding -------------------------
            base = args.seed * 131 + si
            ens = []
            t0 = time.perf_counter()
            for k in range(args.K):
                torch.manual_seed(base * 10_000 + k)
                s = sit_conditional_sample_points_chunked(
                    net=bundle.model, transport=bundle.components["transport"],
                    coords=coords, obs_coords=oc, obs_values=ov, obs_mask=om,
                    obs_field_ids=ofid, n_fields=int(fields.shape[-1]),
                    device=device, n_steps=n_steps, sampler_type=sampler_type,
                    chunk=int(node_subsample),
                    sigma=float(arch.get("cond_fill_sigma", 0.05)), rechunk=args.rechunk,
                )
                ens.append(s[0].detach().float().cpu().numpy())
            sec_per_member = (time.perf_counter() - t0) / args.K   # .cpu() syncs each member
            ens = np.stack(ens, axis=0)

            m = ensemble_metrics(ens, fields[0].detach().float().cpu().numpy(),
                                 list(dataset.field_names))
            m["snapshot"] = snap
            m["seed"] = args.seed
            m["ckpt"] = args.ckpt
            m["n_steps"] = n_steps
            m["sit_sampler"] = "rechunk_per_step" if args.rechunk else "chunked"
            m["sec_per_member"] = sec_per_member
            if args.band_metrics:
                m["band"] = jhu_band_metrics(ens, fields[0].detach().float().cpu().numpy(),
                                             dataset.std.numpy())
            out_path.write_text(json.dumps(m, indent=1))

            # Truth / ensemble mean / one sample / predictive SPREAD, on the
            # z-midplane. The K samples are already drawn, so this is free --
            # it is the ensemble-std view the periodic training figure cannot
            # afford (each extra sample there costs a full ~144 s field).
            try:
                cr = item.get("coords_raw")
                cr = cr.cpu().numpy() if cr is not None else coords[0].cpu().numpy()
                save_ensemble_figure(
                    ens, fields[0].detach().float().cpu().numpy(), cr,
                    list(dataset.field_names),
                    out_dir / f"spread_snap{snap}.png",
                    tag=(f"SiT-point {args.ckpt}  snap {snap}  K={args.K} "
                         f"NFE={n_steps}  relL2={m['aggregate']['rel_l2_mean']:.4f}"))
            except Exception as exc:      # a plot must never kill an eval
                print(f"  [warn] spread figure snap {snap} failed: {exc}", flush=True)
            agg = m["aggregate"]
            print(f"[ensemble] snap={snap} K={args.K} " + " ".join(
                f"{k}={v:.5f}" for k, v in agg.items()), flush=True)

    print(f"[done] shard {args.shard}/{args.num_shards} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
