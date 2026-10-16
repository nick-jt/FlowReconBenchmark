"""Extended CoNFiLD canonical eval: TUNE-split sweeps and Stage-B fix arms.

Superset of src/confild_eval_unified.py (imported, not forked -- the operator,
loaders, figure and spectral code are the canonical ones). Additions:

  --snaps            explicit snapshot list (TUNE = odd cube-3 indices) instead
                     of the seeded 50-draw; sensor construction and the snap-29
                     fingerprint gate are UNCHANGED (sensors are still built for
                     every snapshot with the canonical per-snapshot seeding).
  --post-sensor-steps/--post-sensor-lr
                     Stage-B fix 3: after DPS sampling, fine-tune the window
                     latents against the observed sensor values for a few Adam
                     steps (sensor-consistency correction). Strength is tuned
                     on the TUNE split only.
  --window1          Stage-B fix 4: window=1 INFORMATION conditioning on the
                     existing window-32 checkpoints -- each scored snapshot is
                     reconstructed from a joint window sample in which ONLY the
                     target row carries sensors, so the temporal information
                     matches the single-frame baselines. (A true window=1
                     retrain is impossible with the published architecture:
                     channel_mult has 6 levels, so the UNet downsamples the
                     window axis by 32 and window_length must be a multiple of
                     32.)

Everything else (seeding, sensor counts, DDPM sampler, PS guidance, metrics,
figures, output naming via --tag) is byte-for-byte the canonical path.
"""
from paths import DATA_DIR, SRC_DIR

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


from methods.confild.eval_unified import SPECTRA_BANDS, WindowSensorOperator, load_stage1, load_stage2, spectral_bands, zslice_figure
from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import (  # noqa: E402
    get_conditioning_method,
)
from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_noise  # noqa: E402
from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import (  # noqa: E402
    create_sampler,
)
from methods.dmfgen.helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from methods.dmfgen.ensemble_eval import check_canonical_fingerprint, ensemble_metrics, require_compute_node


class SingleRowSensorOperator:
    """Decode ONE row of the latent window at that row's sensors.

    Used for the window=1-information arm: the joint window is still sampled
    (the checkpoint models 32-row images) but only the target row is observed,
    so no other snapshot's sensors inform the reconstruction.
    """

    def __init__(self, decoder, coords, field_ids, latent_min, latent_max, row):
        self.decoder = decoder
        self.coords = coords          # [1, M, 3]
        self.field_ids = field_ids    # [1, M]
        self.lo = latent_min
        self.hi = latent_max
        self.row = int(row)

    def unnorm(self, z_n):
        return (z_n + 1.0) * 0.5 * (self.hi - self.lo) + self.lo

    def forward(self, z_n, **kwargs):
        batch = z_n.shape[0]
        latents = self.unnorm(z_n)[:, 0, self.row, :]                  # [B, D]
        pred = self.decoder(self.coords.expand(batch, -1, -1),
                            latents.unsqueeze(1))                       # [B, M, F]
        return pred.gather(2, self.field_ids.expand(batch, -1)
                           .unsqueeze(-1)).squeeze(-1)                  # [B, M]

    def project(self, data, measurement, **kwargs):
        raise NotImplementedError


def sensor_finetune(op, z_n, y, steps, lr):
    """Latent fine-tune against the observed sensors, from the DPS solution."""
    pre = float(F.mse_loss(op.forward(z_n), y).detach())
    z = z_n.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    post = pre
    for _ in range(int(steps)):
        loss = F.mse_loss(op.forward(z), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        post = float(loss.detach())
    return z.detach(), pre, post


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--stage2-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data", default=f"{DATA_DIR}/JHU_4cubes_stride100.h5")
    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op-seed", type=int, default=1000)
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--snaps", type=int, nargs="+", default=None,
                   help="Explicit snapshot ids to score (overrides the seeded draw).")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0, 2])
    p.add_argument("--n-obs", type=int, nargs="+", default=[19531, 19531])
    p.add_argument("--dps-scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=262144)
    p.add_argument("--row-chunk", type=int, default=4)
    p.add_argument("--post-sensor-steps", type=int, default=0)
    p.add_argument("--post-sensor-lr", type=float, default=5.0e-3)
    p.add_argument("--window1", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", required=True)
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--sensor-indices-npz", default=None,
                   help="INJECT the recorded sensor set (sensor_indices, sensor_field_ids) for "
                        "the scored snapshots instead of their canonical draw")
    p.add_argument("--dump-npz", default=None,
                   help="write the first scored snapshot (truth, member 0, mean, std, sensors) "
                        "in physical units to this npz, the gallery dump schema")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("confild_eval_unified2 requires a CUDA compute node.")
    require_compute_node()
    infix = f"_{args.tag}"

    os.environ.setdefault("JHU_SPLIT_MODE", "block")
    os.environ.setdefault("JHU_SPLIT_GAP", "0")
    os.environ["JHU_AUGMENT"] = ""
    device = torch.device(args.device)
    out = Path(args.out_dir)
    (out / "Evaluation").mkdir(parents=True, exist_ok=True)

    ds_kwargs = dict(train_ratio=args.train_ratio, seed=42,
                     field_names=("Ux", "Uy", "Uz", "p"), time_stride=1,
                     stats_path=str(out / "dataset_stats.pt"))
    TurbulentCombustionH5Dataset(args.data, split="train", **ds_kwargs)
    dataset = TurbulentCombustionH5Dataset(args.data, split="val", **ds_kwargs)
    n_fields = int(dataset.num_fields)
    field_names = [str(n) for n in getattr(dataset, "field_names",
                                           [f"f{i}" for i in range(n_fields)])]

    decoder, stats, s1 = load_stage1(args.stage1_ckpt, device)
    denoiser, latent_min, latent_max, window = load_stage2(
        args.stage2_ckpt, device, int(s1["architecture"]["latent_dim"]), 32)
    latent_dim = int(s1["architecture"]["latent_dim"])

    coords_full = dataset[0]["coords"].to(device) * 2.0 - 1.0
    drift = float((coords_full.amin(0) + 1.0).abs().max()
                  + (coords_full.amax(0) - 1.0).abs().max())
    if drift > 1e-4:
        raise RuntimeError(f"coordinate convention mismatch (drift {drift:.3e})")

    sampler = create_sampler(sampler="ddpm", steps=args.steps, noise_schedule="cosine",
                             model_mean_type="epsilon", model_var_type="fixed_large",
                             dynamic_threshold=False, clip_denoised=True,
                             rescale_timesteps=False, timestep_respacing="")
    noiser = get_noise(sigma=0.0, name="gaussian")

    if args.snaps is not None:
        scored = sorted(set(int(s) for s in args.snaps))
        if any(s < 0 or s >= len(dataset) for s in scored):
            raise ValueError(f"--snaps out of range for val split of {len(dataset)}")
    else:
        rng = np.random.default_rng(args.seed)
        snap_ids = rng.choice(len(dataset), size=min(args.n_snapshots, len(dataset)),
                              replace=False)
        scored = sorted(int(s) for s in snap_ids)

    # --- sensors: canonical seeding, per snapshot, gate intact ---------------
    sensors = {}
    for snap in range(len(dataset)):
        item = dataset[int(snap)]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)
        torch.manual_seed(args.seed * 777 + int(snap))
        _oc, _ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=args.cond_fields, n_obs_min=args.n_obs, n_obs_max=args.n_obs,
        )
        valid = om[0] > 0
        print(f"[seedcheck] snap={int(snap)} sensors={int(om[0].sum())} "
              f"idx_sum={int(oi[0][valid].sum())}", flush=True)
        check_canonical_fingerprint(int(snap), int(om[0].sum()),
                                    int(oi[0][valid].sum()), args.seed,
                                    args.cond_fields, args.n_obs)
        sensors[int(snap)] = (oi[0, valid].long(), ofid[0, valid].long())

    if args.sensor_indices_npz:
        R = np.load(args.sensor_indices_npz, allow_pickle=False)
        ridx = torch.from_numpy(np.asarray(R["sensor_indices"], dtype=np.int64)).to(device)
        rfid = torch.from_numpy(np.asarray(R["sensor_field_ids"], dtype=np.int64)).to(device)
        for snap in scored:
            if ridx.numel() != sensors[snap][0].numel():
                raise RuntimeError(f"[inject] recorded {ridx.numel()} sensors, draw has {sensors[snap][0].numel()}")
            sensors[snap] = (ridx.clone(), rfid.clone())
        print(f"[inject] sensors of snaps {scored} taken from {args.sensor_indices_npz} "
              f"(idx_sum={int(ridx.sum())})", flush=True)

    widths = {int(v[0].numel()) for v in sensors.values()}
    if len(widths) != 1:
        raise RuntimeError(f"sensor counts differ across snapshots {sorted(widths)}")
    n_sensor = widths.pop()

    def measurement_rows(rows, s_idx, s_fld):
        meas = []
        for w, r in enumerate(rows):
            ids = s_idx[w]
            z_at = dataset[r]["fields"][ids.cpu()].to(device)
            raw_at = z_at * stats.benchmark_std + stats.benchmark_mean
            pm1 = stats.normalize(raw_at, ids)
            meas.append(pm1.gather(1, s_fld[w].unsqueeze(-1)).squeeze(-1))
        return torch.cat(meas).reshape(1, -1)

    def decode_latent(latent):
        pieces = []
        with torch.no_grad():
            for s in range(0, dataset.num_points, args.chunk):
                c = coords_full[s:s + args.chunk].unsqueeze(0)
                pieces.append(decoder(c, latent.view(1, 1, -1))[0])
        pm1 = torch.cat(pieces, 0)
        phys = stats.denormalize(pm1)
        return stats.benchmark_normalize(phys).float().cpu().numpy()

    results, band_records, correction_log = {}, [], []

    def score(snap, ens_slot, window_start):
        truth_np = dataset[snap]["fields"].float().numpy()
        m = ensemble_metrics(ens_slot, truth_np, field_names)
        if args.dump_npz and snap == scored[0]:
            mu, sd = dataset.mean.numpy().astype(np.float32), dataset.std.numpy().astype(np.float32)
            s_idx, s_fld = sensors[snap]
            meta = {"method": "confild", "stage1_ckpt": args.stage1_ckpt, "stage2_ckpt": args.stage2_ckpt,
                    "split": "val", "snapshot_index": int(snap),
                    "absolute_frame": int(dataset.indices[int(snap)]), "window_start": int(window_start),
                    "window_length": int(window), "dps_scale": args.dps_scale, "steps": args.steps,
                    "K": int(ens_slot.shape[0]), "cond_fields": args.cond_fields, "n_obs": args.n_obs,
                    "n_sensors": int(s_idx.numel()), "sensor_idx_sum": int(s_idx.sum()),
                    "sensor_source": (f"INJECTED from {Path(args.sensor_indices_npz).name}"
                                      if args.sensor_indices_npz else "canonical draw"),
                    "units": "physical (arr = standardized * norm_std + norm_mean)"}
            np.savez_compressed(
                args.dump_npz, truth=(truth_np * sd + mu).astype(np.float32),
                pred_sample=(ens_slot[0] * sd + mu).astype(np.float32),
                pred_mean=(ens_slot.mean(0) * sd + mu).astype(np.float32),
                pred_std=(ens_slot.std(0) * sd).astype(np.float32),
                coords=dataset.coords.numpy().astype(np.float32),
                coords_raw=dataset.coords_raw.numpy().astype(np.float32),
                sensor_indices=s_idx.cpu().numpy().astype(np.int64),
                sensor_field_ids=s_fld.cpu().numpy().astype(np.int64),
                names=np.array(field_names), norm_mean=mu, norm_std=sd, meta=np.array(json.dumps(meta)))
            print(f"[dump] wrote {args.dump_npz}", flush=True)
        m["spectral_bands_ensemble_mean"] = spectral_bands(
            ens_slot.mean(axis=0), truth_np, field_names)
        m["spectral_bands_member0"] = spectral_bands(ens_slot[0], truth_np, field_names)
        m["snapshot"] = int(snap)
        m["window_start"] = int(window_start)
        agg = m["aggregate"]
        print(f"[confild-eval2] snap {snap} " +
              " ".join(f"{a}={b:.5f}" for a, b in agg.items()), flush=True)
        json.dump(m, open(out / "Evaluation" / f"crps{infix}_snap{snap}.json", "w"),
                  indent=1)
        if not args.no_figs:
            try:
                zslice_figure(out / "Evaluation" / f"zslice{infix}_snap{snap:03d}.png",
                              ens_slot, truth_np, field_names, snap, agg)
            except Exception as exc:
                print(f"[confild-eval2] figure failed for snap {snap}: {exc}", flush=True)
        band_records.append(m)
        results[int(snap)] = agg

    if args.window1:
        # ---- window=1 INFORMATION arm ------------------------------------
        for snap in scored:
            start = min(max(0, snap - window // 2), len(dataset) - window)
            row = snap - start
            s_idx, s_fld = sensors[snap]
            op = SingleRowSensorOperator(
                decoder, coords_full[s_idx].unsqueeze(0), s_fld.unsqueeze(0),
                latent_min, latent_max, row)
            y = measurement_rows([snap], s_idx.unsqueeze(0), s_fld.unsqueeze(0))
            cond = get_conditioning_method(name="ps", operator=op, noiser=noiser,
                                           scale=args.dps_scale)
            sample_fn = partial(sampler.p_sample_loop, model=denoiser,
                                measurement_cond_fn=partial(cond.conditioning),
                                record=False, save_root=None)
            ens = np.zeros((args.K, 1, dataset.num_points, n_fields), dtype=np.float32)
            for k in range(args.K):
                torch.manual_seed(args.op_seed + 100003 * k + 7919 * snap)
                z0 = torch.randn(1, 1, window, latent_dim, device=device)
                z_n = sample_fn(x_start=z0, measurement=y)
                if args.post_sensor_steps:
                    z_n, pre, post = sensor_finetune(op, z_n, y,
                                                     args.post_sensor_steps,
                                                     args.post_sensor_lr)
                    correction_log.append({"snap": snap, "k": k,
                                           "sensor_mse_pre": pre,
                                           "sensor_mse_post": post})
                latent = op.unnorm(z_n.detach())[0, 0, row]
                ens[k, 0] = decode_latent(latent)
                print(f"[confild-eval2] window1 snap {snap} member {k+1}/{args.K}",
                      flush=True)
            score(snap, ens[:, 0], start)
    else:
        # ---- canonical window-joint path (+ optional correction) ---------
        starts, cursor = [], 0
        while cursor < len(dataset):
            start = min(cursor, max(0, len(dataset) - window))
            starts.append(start)
            if start + window >= len(dataset):
                break
            cursor = start + window
        print(f"[confild-eval2] windows {starts} (length {window})", flush=True)
        done = set()
        for start in starts:
            rows = list(range(start, min(start + window, len(dataset))))
            needed = [w for w, r in enumerate(rows) if r in scored and r not in done]
            if not needed:
                continue
            s_idx = torch.stack([sensors[r][0] for r in rows])
            s_fld = torch.stack([sensors[r][1] for r in rows])
            s_coords = coords_full[s_idx]
            y = measurement_rows(rows, s_idx, s_fld)
            op = WindowSensorOperator(decoder, s_coords, s_fld, latent_min,
                                      latent_max, args.row_chunk)
            cond = get_conditioning_method(name="ps", operator=op, noiser=noiser,
                                           scale=args.dps_scale)
            sample_fn = partial(sampler.p_sample_loop, model=denoiser,
                                measurement_cond_fn=partial(cond.conditioning),
                                record=False, save_root=None)
            ens = np.zeros((args.K, len(needed), dataset.num_points, n_fields),
                           dtype=np.float32)
            for k in range(args.K):
                torch.manual_seed(args.op_seed + 100003 * k + start)
                z0 = torch.randn(1, 1, len(rows), latent_dim, device=device)
                z_n = sample_fn(x_start=z0, measurement=y)
                if args.post_sensor_steps:
                    z_n, pre, post = sensor_finetune(op, z_n, y,
                                                     args.post_sensor_steps,
                                                     args.post_sensor_lr)
                    correction_log.append({"window": start, "k": k,
                                           "sensor_mse_pre": pre,
                                           "sensor_mse_post": post})
                latents = op.unnorm(z_n.detach()).reshape(len(rows), latent_dim)
                for slot, w in enumerate(needed):
                    ens[k, slot] = decode_latent(latents[w])
                print(f"[confild-eval2] window {start} member {k+1}/{args.K}",
                      flush=True)
            for slot, w in enumerate(needed):
                done.add(rows[w])
                score(rows[w], ens[:, slot], start)

    keys = sorted({k for v in results.values() for k in v})
    mean = {k: float(np.mean([v[k] for v in results.values() if k in v])) for k in keys}
    band_mean = {}
    for which in ("spectral_bands_ensemble_mean", "spectral_bands_member0"):
        band_mean[which] = {
            f"{name}.{band}": float(np.mean([bands[which][name][band]
                                             for bands in band_records]))
            for name in field_names for band in SPECTRA_BANDS
        } if band_records else {}
    summary = {
        "per_snapshot": results, "mean": mean, "spectral_bands_mean": band_mean,
        "n_scored": len(results), "n_sensor_total": int(n_sensor),
        "window_length": int(window), "K": args.K, "seed": args.seed,
        "op_seed": args.op_seed, "dps_scale": args.dps_scale,
        "post_sensor_steps": args.post_sensor_steps,
        "post_sensor_lr": args.post_sensor_lr, "window1": bool(args.window1),
        "snaps": scored, "stage1_ckpt": args.stage1_ckpt,
        "stage2_ckpt": args.stage2_ckpt, "tag": args.tag,
        "correction_log": correction_log,
    }
    json.dump(summary, open(out / "Evaluation" / f"crps{infix}_summary.json", "w"),
              indent=1)
    print("[confild-eval2] mean " + " ".join(f"{k}={v:.5f}" for k, v in mean.items()),
          flush=True)


if __name__ == "__main__":
    main()
