"""DeepONet evaluation on the shared ICLR cross-cube JHU protocol.

Mirrors `eval_senseiver_iclr.py` exactly so the two baselines are scored by the
same code path, and reproduces `ensemble_eval.py`'s canonical driver so the
sensor layouts are byte-identical to every other method:

  rng = np.random.default_rng(seed)                                # :291
  snap_ids = rng.choice(len(dataset), n_snapshots, replace=False)  # :292-293
  for snap in snap_ids:
      torch.manual_seed(seed * 777 + int(snap))                    # :300
      build_sparse_condition(...)   # imported from helpers.py, NOT
                                    # helpers_baseline.py (different RNG stream)

Fingerprint gate: at --seed 0 --n-obs 19531 --cond-fields 0 2, snapshot 29 must
print `sensors=39062 idx_sum=37987162596`.  That value is only reproducible on a
COMPUTE node (H100 SXM); torch.randperm on CUDA is not portable across GPU SKU,
so this script must never be run on a login node.

DeepONet is deterministic: K=1, fair CRPS reduces exactly to MAE, and spread /
coverage are undefined and not reported.

Emits, in one JSON:
  * per-field and aggregate metrics from ensemble_eval.ensemble_metrics
  * sensor consistency (obs_consistency.observation_consistency_metrics)
  * the nearest-sensor (Voronoi) floor on the IDENTICAL sensor draw,
    non-periodic (torch.cdist, no wrap: the cutout spans 12.11% of the 2-pi
    domain and is NOT periodic)
  * observed-channel vs unobserved-channel split against that floor
  * a sensor-count sweep (a model that ignores its conditioning is flat)
  * inference wall-clock and peak GPU memory for one full 125^3 field
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from methods.dmfgen.ensemble_eval import check_canonical_fingerprint, ensemble_metrics, require_compute_node
from methods.dmfgen.helpers import build_sparse_condition
from methods.common.helpers_baseline import nearest_sensor_fill_nodes
from methods.dmfgen.obs_consistency import observation_consistency_metrics
from methods.deeponet.model import build_deeponet
import methods.common.config as bconfig
import methods.common.helpers_baseline as hb


def parse_args():
    p = argparse.ArgumentParser("DeepONet ICLR protocol evaluation.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pt")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--n-obs", type=int, default=19531,
                   help="FIXED sensors per conditioned field (1%% of 1,953,125).")
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--cond-fields", type=int, nargs="*", default=None)
    p.add_argument("--sweep", type=int, nargs="*",
                   default=[1953, 3906, 7812, 15625, 19531])
    p.add_argument("--sweep-snapshots", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--skip-nn-floor", action="store_true")
    p.add_argument("--skip-sweep", action="store_true")
    return p.parse_args()


def det_ensemble(pred_np):
    """Two identical members: ensemble_eval.fair_crps divides by K*(K-1) and
    ensemble_metrics uses std(ddof=1), so a literal K=1 array is NaN.  With two
    copies the pair term is exactly zero, so crps = mean|x-y| = MAE and
    spread = 0.  Every other key is unaffected."""
    return np.repeat(pred_np[None], 2, axis=0)


def draw_sensors(coords, fields, cond_fields, n_obs, seed, snap):
    """Byte-identical to ensemble_eval.py:300-304."""
    torch.manual_seed(seed * 777 + int(snap))
    n = [n_obs] * len(cond_fields)
    return build_sparse_condition(coords_full=coords, fields_full=fields,
                                  cond_fields=cond_fields, n_obs_min=n, n_obs_max=n)


@torch.no_grad()
def predict_full_field(model, coords, obs, chunk):
    """DeepONet evaluates the branch ONCE per input function (upstream
    DeepONetCartesianProd), then the trunk over query chunks."""
    oc, ov, om, _, ofid = obs
    coeff = model.encode(oc, ov, om, ofid)
    n_pts = coords.shape[1]
    out = torch.empty(coords.shape[0], n_pts, model.n_fields,
                      device=coords.device, dtype=coords.dtype)
    for s in range(0, n_pts, chunk):
        e = min(s + chunk, n_pts)
        out[:, s:e] = model.combine(coeff, model.trunk_forward(coords[:, s:e]))
    return out


def _subset_choice(ids, n, size=None, replace=False):
    k = int(size) if size is not None else len(ids)
    if k > len(ids):
        raise SystemExit(f"[subset] asked for {k} snapshots but the override lists {len(ids)}")
    return np.asarray(list(ids)[:k], dtype=np.int64)


def main(build_fn=build_deeponet, snapshot_ids=None):
    args = parse_args()
    require_compute_node()
    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    cfg = bconfig.load_yaml(run_dir / "run_config.yaml")
    shared = cfg["shared"]
    arch = cfg["deeponet_params"]["architecture"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = hb.TurbulentCombustionH5Dataset(
        str(bconfig.ensure_absolute(shared["paths"]["data_path"])), split=args.split,
        train_ratio=float(shared["data"]["train_ratio"]), seed=int(shared["seed"]),
        time_stride=int(shared["data"]["time_stride"]),
        field_names=tuple(shared["data"]["field_names"]),
        stats_path=str(run_dir / "dataset_stats.pt"))
    field_names = list(dataset.field_names)[: dataset.num_fields]
    cond_fields = args.cond_fields or list(shared["conditioning"]["cond_fields"])
    unobs = [f for i, f in enumerate(field_names) if i not in cond_fields]
    obsf = [field_names[i] for i in cond_fields]

    model = build_fn(arch).to(device)
    ck = bconfig.safe_torch_load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    model.eval()
    n_params = model.n_params()
    bottleneck = model.bottleneck_scalars()
    field_values = dataset.num_points * dataset.num_fields
    print(f"[eval] trainable_params={n_params}", flush=True)
    print(f"[eval] p={model.p} bottleneck_scalars={bottleneck} "
          f"compression={field_values/bottleneck:.1f}x", flush=True)
    print(f"[eval] checkpoint={ckpt_path} epoch={ck.get('epoch')}", flush=True)

    if snapshot_ids is None:
        rng = np.random.default_rng(args.seed)
        snap_ids = rng.choice(len(dataset), size=min(args.n_snapshots, len(dataset)),
                              replace=False)
    else:  # explicit subset (DeepONet++ TUNE/TEST split); same size contract as the seeded draw
        snap_ids = _subset_choice(snapshot_ids, len(dataset), size=min(args.n_snapshots, len(dataset)))

    results = {
        "model": "deeponet", "run_dir": str(run_dir), "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ck.get("epoch", -1)), "split": args.split,
        "n_snapshots": int(len(snap_ids)), "snapshot_ids": [int(x) for x in snap_ids],
        "cond_fields": cond_fields, "n_obs_per_field": args.n_obs, "seed": args.seed,
        "field_names": field_names, "trainable_params": int(n_params),
        "latent_p": int(model.p), "bottleneck_scalars": int(bottleneck),
        "compression_vs_field": float(field_values / bottleneck),
        "deterministic": True, "K": 1,
        "note_K": "two identical members so the fair-CRPS estimator and ddof=1 "
                  "spread are defined; CRPS == MAE, spread == 0 exactly",
        "note_periodicity": "nearest-sensor floor uses plain Euclidean cdist; the "
                            "cutout spans 12.11% of the 2-pi box and is NOT periodic",
    }

    per_snap, senconsis, nn_floor = [], [], []
    infer_times, infer_mems, fingerprints = [], [], []
    for si, snap in enumerate(snap_ids):
        item = dataset[int(snap)]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)
        obs = draw_sensors(coords, fields, cond_fields, args.n_obs, args.seed, int(snap))
        om, oi = obs[2], obs[3]
        sel = oi[om > 0]
        fp = {"snap": int(snap), "sensors": int(sel.numel()),
              "idx_sum": int(sel.sum().item())}
        fingerprints.append(fp)
        print(f"[seedcheck] snap={fp['snap']} sensors={fp['sensors']} "
              f"idx_sum={fp['idx_sum']}", flush=True)
        check_canonical_fingerprint(fp["snap"], fp["sensors"], fp["idx_sum"],
                                    args.seed, cond_fields,
                                    [args.n_obs] * len(cond_fields))

        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        recon = predict_full_field(model, coords, obs, args.chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            infer_mems.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
        infer_times.append(time.perf_counter() - t0)

        gt = fields[0].float().cpu().numpy()
        per_snap.append(ensemble_metrics(det_ensemble(recon[0].float().cpu().numpy()),
                                         gt, field_names))
        per_snap[-1]["snapshot"] = int(snap)
        senconsis.append(observation_consistency_metrics(
            recon, obs[1], obs[2], obs[3], obs[4], field_names))
        if not args.skip_nn_floor:
            nn_val, _ = nearest_sensor_fill_nodes(
                coords, obs[0], obs[1], obs[2], obs[4], n_fields=dataset.num_fields)
            nn_floor.append(ensemble_metrics(
                det_ensemble(nn_val[0].float().cpu().numpy()), gt, field_names))
            del nn_val
        del recon
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[eval] snapshot {si+1}/{len(snap_ids)} (id={int(snap)}) "
              f"rel_l2={per_snap[-1]['aggregate']['rel_l2_mean']:.5f} "
              f"crps(MAE)={per_snap[-1]['aggregate']['crps']:.5f} "
              f"senconsis={senconsis[-1]['obs_rel_l2_SenConsis']:.5f}", flush=True)

    results["fingerprints"] = fingerprints
    _fp29 = [f for f in fingerprints if f["snap"] == 29]
    if _fp29:
        ok = (_fp29[0]["sensors"] == 39062 and _fp29[0]["idx_sum"] == 37987162596)
        results["fingerprint_snap29_matches_canonical"] = bool(ok)
        print(f"[seedcheck] CANONICAL snap=29 match={ok} got={_fp29[0]}", flush=True)

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
    results["snapshots"] = per_snap   # per-snapshot records for paired tests and TUNE/TEST splits
    results["per_field"] = {f: {k: mean_of(per_snap, ["per_field", f, k]) for k in keys}
                            for f in field_names}
    if nn_floor:
        results["nearest_sensor_floor"] = {
            "aggregate": {k: mean_of(nn_floor, ["aggregate", k]) for k in keys},
            "per_field": {f: {k: mean_of(nn_floor, ["per_field", f, k]) for k in keys}
                          for f in field_names}}
        # The comparison that matters for a deterministic baseline: does it beat
        # trivial interpolation where it HAS data, and does it infer anything
        # where it does not?
        results["floor_split"] = {
            "observed_channels": obsf, "unobserved_channels": unobs,
            "model_observed_rel_l2": float(np.mean(
                [results["per_field"][f]["rel_l2_mean"] for f in obsf])),
            "floor_observed_rel_l2": float(np.mean(
                [results["nearest_sensor_floor"]["per_field"][f]["rel_l2_mean"]
                 for f in obsf])),
            "model_unobserved_rel_l2": float(np.mean(
                [results["per_field"][f]["rel_l2_mean"] for f in unobs])),
            "floor_unobserved_rel_l2": float(np.mean(
                [results["nearest_sensor_floor"]["per_field"][f]["rel_l2_mean"]
                 for f in unobs])),
        }
        fs = results["floor_split"]
        fs["beats_floor_on_observed"] = bool(
            fs["model_observed_rel_l2"] < fs["floor_observed_rel_l2"])
        fs["infers_unobserved_below_trivial"] = bool(fs["model_unobserved_rel_l2"] < 1.0)
        print(f"[floor] {json.dumps(fs)}", flush=True)

    sc_keys = sorted({k for d in senconsis for k in d})
    results["sensor_consistency"] = {
        k: float(np.nanmean([float(d[k]) for d in senconsis if k in d])) for k in sc_keys}

    results["cost"] = {
        "inference_wallclock_s_full_field_mean": float(np.mean(infer_times)),
        "inference_wallclock_s_full_field_median": float(np.median(infer_times)),
        "inference_peak_gpu_mem_mb": float(np.max(infer_mems)) if infer_mems else None,
        "field_points": int(dataset.num_points), "query_chunk": args.chunk}
    print(f"[cost] inference_wallclock_s_full_field="
          f"{results['cost']['inference_wallclock_s_full_field_median']:.3f} "
          f"inference_peak_gpu_mem_mb={results['cost']['inference_peak_gpu_mem_mb']}",
          flush=True)

    if not args.skip_sweep:
        sweep = {}
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
                    recon, obs[1], obs[2], obs[3], obs[4],
                    field_names)["obs_rel_l2_SenConsis"])
                del recon
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            sweep[str(n_obs)] = {"rel_l2_mean": float(np.nanmean(rl2)),
                                 "obs_rel_l2_SenConsis": float(np.nanmean(sc))}
            print(f"[sweep] n_obs={n_obs} rel_l2={sweep[str(n_obs)]['rel_l2_mean']:.5f} "
                  f"senconsis={sweep[str(n_obs)]['obs_rel_l2_SenConsis']:.5f}", flush=True)
        results["sensor_sweep"] = sweep
        order = [sweep[str(n)]["rel_l2_mean"] for n in args.sweep]
        monotone = all(order[i] >= order[i + 1] - 1e-9 for i in range(len(order) - 1))
        spread = float(max(order) - min(order))
        results["sensor_sweep_monotone"] = bool(monotone)
        results["sensor_sweep_spread_10x"] = spread
        senc = results["sensor_consistency"].get("obs_rel_l2_SenConsis", float("nan"))
        beats_nn = (results["aggregate"]["rel_l2_mean"] <
                    results.get("nearest_sensor_floor", {}).get(
                        "aggregate", {}).get("rel_l2_mean", float("inf")))
        results["acceptance_gate"] = {
            "sensor_consistency": float(senc),
            "sensor_consistency_below_0.1": bool(senc < 0.1),
            "monotone_in_sensor_count": bool(monotone),
            "sweep_spread_over_10x_sensors": spread,
            "responds_to_conditioning": bool(spread > 0.01),
            "beats_nearest_sensor_floor_aggregate": bool(beats_nn),
        }
        print(f"[gate] {json.dumps(results['acceptance_gate'])}", flush=True)

    out = Path(args.out) if args.out else run_dir / "Evaluation" / "iclr_protocol_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as h:
        json.dump(results, h, indent=2)
    print(f"[eval] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
