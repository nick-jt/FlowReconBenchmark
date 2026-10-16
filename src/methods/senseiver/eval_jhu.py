"""Senseiver evaluation on the shared ICLR cross-cube JHU protocol.

Runs one checkpoint over every held-out snapshot with seeded sensor draws and
emits, in a single JSON:

  * per-field and aggregate metrics from `ensemble_eval.ensemble_metrics`
    (Senseiver is deterministic, so K=1 and CRPS reduces to MAE);
  * sensor-consistency (`obs_consistency.observation_consistency_metrics`);
  * a nearest-sensor (Voronoi) interpolation floor computed from the *same*
    sensor draw, as the sanity baseline the model must beat;
  * a sensor-count sweep for the monotonicity acceptance gate;
  * inference wall-clock and peak GPU memory for one full 125^3 field.

Sensor draws replicate `ensemble_eval.py`'s canonical driver EXACTLY, so this
baseline sees byte-identical sensor layouts to every other method:

  rng = np.random.default_rng(seed)                       # ensemble_eval.py:291
  snap_ids = rng.choice(len(dataset), n_snapshots, replace=False)   # :292-293
  for snap in snap_ids:
      torch.manual_seed(seed * 777 + int(snap))           # ensemble_eval.py:300
      build_sparse_condition(...)                         # from helpers.py

Each draw prints a `[seedcheck] snap=.. sensors=.. idx_sum=..` fingerprint.
At --seed 0 --n-obs 19531 with cond_fields [0, 2], snapshot 29 must give
sensors=39062 idx_sum=37987162596.

Evaluation uses a FIXED sensor count -- 19,531 per observed channel (1% of
1,953,125).  The log-uniform 0.1%-1% range is the TRAINING distribution only.

Canonical invocation:
    --seed 0 --n-snapshots 50 --n-obs 19531 --cond-fields 0 2
(K=8 is irrelevant here: Senseiver is deterministic, so K=1 and CRPS = MAE.)
"""
from __future__ import annotations
from paths import DATA_DIR

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from methods.dmfgen.ensemble_eval import (check_canonical_fingerprint, ensemble_metrics, jhu_band_metrics,
                                      require_compute_node)
# CANONICAL sensor draw.  helpers.py:536 uses `torch.randint(..., size=(1,))`
# on the CPU generator; the helpers_baseline.py:1457 copy passes device=cuda,
# which advances the CUDA stream before `torch.randperm(n_pts, device=cuda)`
# and so produces a DIFFERENT permutation from the same seed.  ensemble_eval.py
# uses the helpers.py version, so evaluation must import it from there or the
# paired per-snapshot CIs and TOST tests are invalid.
from methods.dmfgen.helpers import build_sparse_condition
from methods.common.helpers_baseline import nearest_sensor_fill_nodes
from methods.dmfgen.obs_consistency import observation_consistency_metrics
import methods.common.config as bconfig
import methods.common.data as bdata
import methods.common.registry as bregistry


def parse_args():
    p = argparse.ArgumentParser("Senseiver ICLR protocol evaluation.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pt",
                   help="Checkpoint file inside --run-dir (best.pt | budget.pt | last.pt).")
    p.add_argument("--config", type=str, default=None,
                   help="Fallback YAML; run_config.yaml in --run-dir wins if present.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--n-obs", type=int, default=19531,
                   help="FIXED sensors per conditioned field (1%% of 1,953,125).")
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--band-metrics", action="store_true",
                   help="Also store per-snapshot band energy / correlation.")
    p.add_argument("--sweep", type=int, nargs="*",
                   default=[1953, 3906, 7812, 15625, 19531],
                   help="Sensor counts for the monotonicity gate.")
    p.add_argument("--sweep-snapshots", type=int, default=8,
                   help="Snapshots used for the sweep (the headline uses all).")
    p.add_argument("--seed", type=int, default=0,
                   help="Canonical protocol seed; ensemble_eval.py's default.")
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--skip-nn-floor", action="store_true")
    return p.parse_args()


def det_ensemble(pred_np):
    """Tile a deterministic prediction into a K=2 ensemble of identical members.

    `ensemble_eval.fair_crps` divides by K*(K-1), and `ensemble_metrics` uses
    std(ddof=1), so a literal K=1 array yields NaN for both.  With two identical
    members the fair estimator's pair term is exactly zero, so
    crps = mean|x - y| = MAE (the deterministic limit) and spread = 0 exactly.
    Every other key is unaffected.
    """
    return np.repeat(pred_np[None], 2, axis=0)


def draw_sensors(coords, fields, cond_fields, n_obs, seed, snap):
    """Byte-identical to ensemble_eval.py:300-304."""
    torch.manual_seed(seed * 777 + int(snap))
    n_obs_list = [n_obs] * len(cond_fields)
    return build_sparse_condition(
        coords_full=coords, fields_full=fields, cond_fields=cond_fields,
        n_obs_min=n_obs_list, n_obs_max=n_obs_list,
    )


@torch.no_grad()
def predict_full_field(model, coords, obs, chunk):
    """Chunked full-field reconstruction, mirroring upstream network_light.test()
    (which re-runs the whole forward, encoder included, per query chunk)."""
    obs_coords, obs_values, obs_mask, _, obs_field_ids = obs
    n_pts = coords.shape[1]
    out = torch.empty(coords.shape[0], n_pts, model.n_fields,
                      device=coords.device, dtype=coords.dtype)
    for s in range(0, n_pts, chunk):
        e = min(s + chunk, n_pts)
        out[:, s:e] = model(coords[:, s:e], obs_coords, obs_values, obs_mask, obs_field_ids)
    return out


def main():
    args = parse_args()
    require_compute_node()
    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    cfg_path = run_dir / "run_config.yaml"
    if cfg_path.exists():
        cfg = bconfig.load_yaml(cfg_path)
    elif args.config:
        cfg = bconfig.load_yaml(bconfig.ensure_absolute(args.config))
    else:
        raise FileNotFoundError(f"No run_config.yaml in {run_dir} and no --config given.")
    cfg = bconfig.validate_and_normalize_config(cfg)

    # Training may have staged the H5 to node-local NVMe and recorded that path
    # in run_config.yaml; that path does not exist on the evaluation node.
    # Fall back to the recorded shared copy, then to the fleet-wide dataset.
    _paths = cfg["shared"]["paths"]
    _dp = Path(_paths["data_path"])
    if not _dp.exists():
        _fallbacks = [_paths.get("data_path_shared"),
                      f"{DATA_DIR}/JHU_4cubes_stride100.h5"]
        for _c in _fallbacks:
            if _c and Path(_c).exists():
                print(f"[eval] staged data_path {_dp} absent; using {_c}", flush=True)
                _paths["data_path"] = str(_c)
                break
        else:
            raise FileNotFoundError(f"data_path {_dp} missing and no fallback exists")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = bdata.build_dataset(cfg, split=args.split, stats_path=run_dir / "dataset_stats.pt")
    field_names = list(dataset.field_names)[: dataset.num_fields]
    cond_fields = list(cfg["shared"]["conditioning"]["cond_fields"])

    adapter = bregistry.get_baseline_adapter(cfg["baseline_model"])
    bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=run_dir,
                                        train_set=dataset, val_set=dataset)
    checkpoint = bconfig.safe_torch_load(ckpt_path, map_location="cpu")
    adapter.load_checkpoint(bundle, checkpoint)
    model = bundle.model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    arch = bconfig.resolve_stage_config(cfg)["architecture"]
    bottleneck = int(arch["num_latents"]) * int(arch["latent_dim"])
    print(f"[eval] trainable_params={n_params}", flush=True)
    print(f"[eval] latent_bottleneck_scalars={bottleneck}", flush=True)
    print(f"[eval] checkpoint={ckpt_path} epoch={checkpoint.get('epoch')}", flush=True)

    # ensemble_eval.py:291-293 -- the canonical snapshot order.
    rng = np.random.default_rng(args.seed)
    snap_ids = rng.choice(len(dataset), size=min(args.n_snapshots, len(dataset)),
                          replace=False)

    results = {
        "run_dir": str(run_dir), "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split, "n_snapshots": int(len(snap_ids)),
        "snapshot_ids": [int(x) for x in snap_ids],
        "cond_fields": cond_fields, "n_obs_per_field": args.n_obs,
        "seed": args.seed, "field_names": field_names,
        "trainable_params": int(n_params),
        "latent_bottleneck_scalars": int(bottleneck),
        "deterministic": True, "K": 1,
        "note_K": "metrics computed with two identical members so the fair CRPS "
                  "estimator and ddof=1 spread are well defined; CRPS == MAE, spread == 0",
    }

    # ---------------- headline: every held-out snapshot -------------------
    per_snap, senconsis, nn_floor = [], [], []
    infer_times, infer_mems = [], []
    for si, snap in enumerate(snap_ids):
        item = dataset[int(snap)]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)
        obs = draw_sensors(coords, fields, cond_fields, args.n_obs, args.seed, int(snap))
        _om, _oi = obs[2], obs[3]
        _sel = _oi[_om > 0]
        print(f"[seedcheck] snap={int(snap)} sensors={int(_sel.numel())} "
              f"idx_sum={int(_sel.sum().item())}", flush=True)
        check_canonical_fingerprint(int(snap), int(_sel.numel()),
                                    int(_sel.sum().item()), args.seed,
                                    cond_fields, [args.n_obs] * len(cond_fields))

        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        recon = predict_full_field(model, coords, obs, args.chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            infer_mems.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
        infer_times.append(time.perf_counter() - t0)

        m = ensemble_metrics(det_ensemble(recon[0].float().cpu().numpy()),
                             fields[0].float().cpu().numpy(), field_names)
        m["snapshot"] = int(snap)
        if args.band_metrics:
            m["band"] = jhu_band_metrics(recon[0].float().cpu().numpy()[None],
                                         fields[0].float().cpu().numpy(), dataset.std.numpy())
        per_snap.append(m)

        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = obs
        senconsis.append(observation_consistency_metrics(
            recon, obs_values, obs_mask, obs_indices, obs_field_ids, field_names))

        if not args.skip_nn_floor:
            nn_val, _ = nearest_sensor_fill_nodes(
                coords, obs_coords, obs_values, obs_mask, obs_field_ids,
                n_fields=dataset.num_fields)
            nn_floor.append(ensemble_metrics(det_ensemble(nn_val[0].float().cpu().numpy()),
                                             fields[0].float().cpu().numpy(), field_names))
        del recon
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[eval] snapshot {si+1}/{len(snap_ids)} (id={int(snap)}) "
              f"rel_l2={per_snap[-1]['aggregate']['rel_l2_mean']:.5f} "
              f"crps(MAE)={per_snap[-1]['aggregate']['crps']:.5f} "
              f"senconsis={senconsis[-1]['obs_rel_l2_SenConsis']:.5f}", flush=True)

    def mean_of(dicts, path):
        vals = []
        for d in dicts:
            v = d
            for k in path:
                v = v[k]
            vals.append(float(v))
        return float(np.nanmean(vals))

    keys = ["rel_l2_mean", "rel_l2_single", "crps", "rmse"]
    results["aggregate"] = {k: mean_of(per_snap, ["aggregate", k]) for k in keys}
    results["per_field"] = {
        f: {k: mean_of(per_snap, ["per_field", f, k]) for k in keys} for f in field_names
    }
    results["snapshots"] = per_snap   # per-snapshot records for paired tests and TUNE/TEST splits
    if nn_floor:
        results["nearest_sensor_floor"] = {
            "aggregate": {k: mean_of(nn_floor, ["aggregate", k]) for k in keys},
            "per_field": {f: {k: mean_of(nn_floor, ["per_field", f, k]) for k in keys}
                          for f in field_names},
        }
    sc_keys = sorted({k for d in senconsis for k in d})
    results["sensor_consistency"] = {
        k: float(np.nanmean([float(d[k]) for d in senconsis if k in d])) for k in sc_keys
    }

    # ---------------- cost instrumentation --------------------------------
    results["cost"] = {
        "inference_wallclock_s_full_field_mean": float(np.mean(infer_times)),
        "inference_wallclock_s_full_field_median": float(np.median(infer_times)),
        "inference_peak_gpu_mem_mb": float(np.max(infer_mems)) if infer_mems else None,
        "field_points": int(dataset.num_points),
        "query_chunk": args.chunk,
    }
    print(f"[cost] inference_wallclock_s_full_field="
          f"{results['cost']['inference_wallclock_s_full_field_median']:.3f}", flush=True)
    print(f"[cost] inference_peak_gpu_mem_mb={results['cost']['inference_peak_gpu_mem_mb']}",
          flush=True)

    # ---------------- sensor-count sweep (monotonicity gate) --------------
    sweep = {}
    # Same seeding scheme, so the sensor sets are NESTED across sweep points
    # (build_sparse_condition takes randperm(...)[:m] from an identical stream),
    # which is what makes the monotonicity gate meaningful.
    n_sw = min(args.sweep_snapshots, len(snap_ids))
    for n_obs in args.sweep:
        rl2, sc = [], []
        for snap in snap_ids[:n_sw]:
            item = dataset[int(snap)]
            coords = item["coords"].unsqueeze(0).to(device)
            fields = item["fields"].unsqueeze(0).to(device)
            obs = draw_sensors(coords, fields, cond_fields, n_obs, args.seed, int(snap))
            recon = predict_full_field(model, coords, obs, args.chunk)
            rl2.append(ensemble_metrics(det_ensemble(recon[0].float().cpu().numpy()),
                                        fields[0].float().cpu().numpy(),
                                        field_names)["aggregate"]["rel_l2_mean"])
            sc.append(observation_consistency_metrics(
                recon, obs[1], obs[2], obs[3], obs[4], field_names
            )["obs_rel_l2_SenConsis"])
            del recon
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sweep[str(n_obs)] = {"rel_l2_mean": float(np.nanmean(rl2)),
                             "obs_rel_l2_SenConsis": float(np.nanmean(sc))}
        print(f"[sweep] n_obs={n_obs} rel_l2={sweep[str(n_obs)]['rel_l2_mean']:.5f} "
              f"senconsis={sweep[str(n_obs)]['obs_rel_l2_SenConsis']:.5f}", flush=True)
    results["sensor_sweep"] = sweep

    # ---------------- acceptance gate -------------------------------------
    order = [sweep[str(n)]["rel_l2_mean"] for n in args.sweep]
    monotone = all(order[i] >= order[i + 1] - 1e-9 for i in range(len(order) - 1))
    senc = results["sensor_consistency"].get("obs_rel_l2_SenConsis", float("nan"))
    beats_nn = (results["aggregate"]["rel_l2_mean"]
                < results.get("nearest_sensor_floor", {}).get("aggregate", {}).get(
                    "rel_l2_mean", float("inf")))
    results["acceptance_gate"] = {
        "sensor_consistency_below_0.1": bool(senc < 0.1),
        "sensor_consistency": float(senc),
        "monotone_in_sensor_count": bool(monotone),
        "beats_nearest_sensor_floor": bool(beats_nn),
        "passed": bool((senc < 0.1) and monotone and beats_nn),
    }
    print(f"[gate] {json.dumps(results['acceptance_gate'])}", flush=True)

    out = Path(args.out) if args.out else run_dir / "Evaluation" / "iclr_protocol_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as h:
        json.dump(results, h, indent=2)
    print(f"[eval] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
