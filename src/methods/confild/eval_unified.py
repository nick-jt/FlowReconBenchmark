"""CoNFiLD stage 3: sensor-guided DPS through the frozen CNF decoder, scored
with the canonical seeded ensemble estimator.

Replaces confild_conditional.py, which was wired to the legacy MLP denoiser in
confild_stage2.py. This version consumes the UNIFIED checkpoints written by
confild_upstream_training.py (upstream UNet + global scalar latent range,
matching upstream's own Case4Operator._unnorm at measurements.py:219).

Guidance is upstream verbatim: PosteriorSampling ('ps') at scale 1.0
(condition_methods.py:79-87) driving a DDPM sampler (cosine, 1000 steps,
epsilon, fixed_large, clip_denoised) -- the Case-4 random-sensor recipe.

WINDOW-JOINT RECONSTRUCTION
    Stage 2 models a latent IMAGE of `window_length` consecutive snapshots, so
    a single p_sample_loop reconstructs a whole window at once. Every row of the
    window is conditioned on its OWN sensors, which is upstream's Case-4
    arrangement and 32x cheaper than sampling a window per snapshot and
    discarding 31 rows. Cube 3's 50 held-out snapshots are covered by windows
    [0:32] and [18:50]; the 14 snapshots in both are scored from the first.

SEEDING -- canonical, identical to ensemble_eval.py:291-304
    rng = np.random.default_rng(seed); snap_ids = rng.choice(len(dataset), n, replace=False)
    torch.manual_seed(seed * 777 + snap) immediately before build_sparse_condition
Evaluation sensor count is FIXED at 1% per observed channel (19,531). The
log-uniform 0.1-1% range is the TRAINING distribution only.
"""
from paths import DATA_DIR, THIRD_PARTY_DIR

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

# The Delta path stays the default; the env var lets other sites (MIT Engaging)
# point at their own checkout of the upstream repo without editing this file.
CONFILD_ROOT = os.environ.get(
    "CONFILD_ROOT", f"{THIRD_PARTY_DIR}/CoNFiLD"
)
sys.path.insert(0, CONFILD_ROOT)

from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import (  # noqa: E402
    get_conditioning_method,
)
from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_noise  # noqa: E402
from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import (  # noqa: E402
    create_sampler,
)
from ConditionalNeuralField.cnf.nf_networks import SIRENAutodecoder_film  # noqa: E402
from UnconditionalDiffusionTraining_and_Generation.src.script_util import create_model  # noqa: E402

# CANONICAL sensor path. helpers.py's build_sparse_condition draws the count
# with a CPU generator; helpers_baseline.py's uses `device=device`, which
# advances CUDA RNG state before torch.randperm and yields a DIFFERENT sensor
# layout for the same seed (measured overlap 2.0% = chance). ensemble_eval.py:29
# imports from helpers, so every reported number must too. Same reason for the
# dataset class.
from methods.dmfgen.helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from methods.confild.upstream_core import FieldStatistics
from methods.dmfgen.ensemble_eval import check_canonical_fingerprint, ensemble_metrics, require_compute_node


class WindowSensorOperator:
    """latent window -> decoded sensor values, upstream's Case-4 construction.

    forward(z_n): [B, 1, W, D] normalised latent image -> [B, W*M] sensor values
    in the decoder's [-1, 1] field units. Row w is decoded at row w's own
    sensors. Activation memory is bounded by checkpointing over row chunks --
    32 rows x 39,062 sensors through a 17-layer SIREN would otherwise hold
    ~40 GB of activations for the DPS backward.
    """

    def __init__(self, decoder, coords, field_ids, latent_min, latent_max, row_chunk):
        self.decoder = decoder
        self.coords = coords            # [W, M, 3] in [-1, 1]
        self.field_ids = field_ids      # [W, M] long
        self.lo = latent_min
        self.hi = latent_max
        self.row_chunk = int(row_chunk)

    def unnorm(self, z_n):
        return (z_n + 1.0) * 0.5 * (self.hi - self.lo) + self.lo

    def _rows(self, latents, coords, field_ids):
        pred = self.decoder(coords, latents.unsqueeze(1))          # [w, M, F]
        return pred.gather(2, field_ids.unsqueeze(-1)).squeeze(-1)  # [w, M]

    def forward(self, z_n, **kwargs):
        batch, _, window, _ = z_n.shape
        latents = self.unnorm(z_n).reshape(batch * window, -1)
        coords = self.coords.repeat(batch, 1, 1)
        field_ids = self.field_ids.repeat(batch, 1)
        out = []
        for start in range(0, latents.shape[0], self.row_chunk):
            stop = min(start + self.row_chunk, latents.shape[0])
            args = (latents[start:stop], coords[start:stop], field_ids[start:stop])
            if torch.is_grad_enabled() and latents.requires_grad:
                out.append(checkpoint(self._rows, *args, use_reentrant=False))
            else:
                out.append(self._rows(*args))
        return torch.cat(out, dim=0).reshape(batch, -1)

    def project(self, data, measurement, **kwargs):
        raise NotImplementedError


# Spectral bands, identical to compare_spectra.py:16-26 and :112-118 so the
# CoNFiLD numbers drop straight into the same table as the other baselines.
SPECTRA_GRID = 125
SPECTRA_KMAX = SPECTRA_GRID // 2          # 62
SPECTRA_BANDS = {"inertial_k8_31": (8, 31), "dissipation_k32_62": (32, 62)}


def shell_spectrum(volume):
    f = np.abs(np.fft.fftn(volume)) ** 2
    k = np.fft.fftfreq(SPECTRA_GRID) * SPECTRA_GRID
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kb = np.round(np.sqrt(KX**2 + KY**2 + KZ**2)).astype(int)
    return np.bincount(kb.ravel(), weights=f.ravel(), minlength=SPECTRA_KMAX + 1)


def spectral_bands(pred, truth, field_names):
    """Band-integrated energy ratio pred/truth per channel.

    Reported for the ensemble MEAN and for member 0 separately: the mean is
    smoother than any member by construction, so quoting only the mean would
    understate small-scale energy for every method equally and hide the effect
    the bottleneck actually has. The dissipation band is the claim that has to
    survive scrutiny, so both are recorded.
    """
    out = {}
    for c, name in enumerate(field_names):
        st = shell_spectrum(truth[:, c].reshape(SPECTRA_GRID, SPECTRA_GRID, SPECTRA_GRID))
        sp = shell_spectrum(pred[:, c].reshape(SPECTRA_GRID, SPECTRA_GRID, SPECTRA_GRID))
        entry = {}
        for band, (lo, hi) in SPECTRA_BANDS.items():
            denom = float(st[lo:hi + 1].sum())
            entry[band] = float(sp[lo:hi + 1].sum() / denom) if denom > 0 else float("nan")
        entry["spectrum_pred"] = sp.tolist()
        entry["spectrum_truth"] = st.tolist()
        out[name] = entry
    return out


def zslice_figure(path, ens, truth, field_names, snap, agg):
    """z-midplane truth / ensemble mean / |error| / ensemble std per channel.

    The project's standing rule is that visual inspection outranks aggregate
    statistics: every failure found so far was visible in a figure and subtle
    in the metrics. All arrays are [.., N, C] in benchmark z-scored units.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = SPECTRA_GRID
    mid = g // 2
    mean = ens.mean(axis=0)
    std = ens.std(axis=0)
    rows = [("truth", truth, "coolwarm"), ("ens mean", mean, "coolwarm"),
            ("|error|", np.abs(mean - truth), "magma"), ("ens std", std, "magma")]
    ncol = len(field_names)
    fig, axs = plt.subplots(len(rows), ncol,
                            figsize=(3.05 * ncol, 3.0 * len(rows)), squeeze=False)
    for c in range(ncol):
        ref = truth[:, c].reshape(g, g, g)[:, :, mid]
        lo, hi = np.percentile(ref, [1, 99])
        for r, (label, arr, cmap) in enumerate(rows):
            plane = arr[:, c].reshape(g, g, g)[:, :, mid]
            ax = axs[r][c]
            if cmap == "coolwarm":
                im = ax.imshow(plane, cmap=cmap, vmin=lo, vmax=hi)
            else:
                im = ax.imshow(plane, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(field_names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(f"CoNFiLD eval - snapshot {snap} (z-midplane, z-scored units)   "
                 f"rel_l2_mean {agg.get('rel_l2_mean', float('nan')):.4f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def load_stage1(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if int(ck.get("format_version", 0)) < 2 or ck.get("training_stage") != 1:
        raise RuntimeError(
            f"{path} is not a unified stage-1 checkpoint (format_version 2). "
            "The legacy cnf_last.pt files are not supported; retrain via "
            "train_confild_unified.slurm."
        )
    arch = ck["architecture"]
    stats = FieldStatistics.from_state_dict(ck["field_statistics"])
    n_fields = int(stats.benchmark_mean.numel())
    decoder = SIRENAutodecoder_film(
        in_coord_features=3, in_latent_features=int(arch["latent_dim"]),
        out_features=n_fields, num_hidden_layers=int(arch["layers"]),
        hidden_features=int(arch["hidden_features"]),
    ).to(device)
    decoder.load_state_dict(ck["decoder"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder, stats.to(device), ck


def load_stage2(path, device, latent_dim, smoke_window):
    if path:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if int(ck.get("format_version", 0)) < 2 or ck.get("training_stage") != 2:
            raise RuntimeError(f"{path} is not a unified stage-2 checkpoint.")
        arch = ck["architecture"]
        model = create_model(
            image_size=int(arch["model_image_size"]), num_channels=int(arch["num_channels"]),
            num_res_blocks=int(arch["num_res_blocks"]), num_heads=int(arch["num_heads"]),
            num_head_channels=int(arch["num_head_channels"]),
            attention_resolutions=str(arch["attention_resolutions"]),
            channel_mult=str(arch["channel_mult"]),
        ).to(device)
        model.load_state_dict(ck["ema"])
        model.eval()
        return model, ck["latent_min"].to(device), ck["latent_max"].to(device), int(arch["window_length"])
    # --smoke: random-init denoiser so every code path runs before stage 2 lands.
    print("[confild-eval] SMOKE: no stage-2 checkpoint, using a random-init UNet. "
          "Metrics from this run are MEANINGLESS.", flush=True)
    model = create_model(
        image_size=latent_dim, num_channels=32, num_res_blocks=1, num_heads=4,
        num_head_channels=-1, attention_resolutions="32,16,8", channel_mult="1,1,1,1,2,2",
    ).to(device).eval()
    one = torch.ones(1, device=device)
    return model, -one, one, smoke_window


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--stage2-ckpt", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data", default=f"{DATA_DIR}/JHU_4cubes_stride100.h5")
    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op-seed", type=int, default=1000)
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0, 2])
    p.add_argument("--n-obs", type=int, nargs="+", default=[19531, 19531])
    p.add_argument("--dps-scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=262144)
    p.add_argument("--row-chunk", type=int, default=4)
    p.add_argument("--smoke-window", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="",
                   help="Suffix inserted into output filenames, e.g. 'canonical_best' "
                        "-> crps_canonical_best_summary.json, so multiple checkpoint "
                        "selections can share one Evaluation/ directory.")
    p.add_argument("--no-figs", action="store_true",
                   help="Disable per-snapshot z-midplane figures.")
    args = p.parse_args()

    # Compute-node guard: this evaluation streams the 5.9 GB HDF5 and runs
    # 1000-step guided DDPM sampling; it must never run on a login node.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "confild_eval_unified requires a CUDA compute node "
            f"(no CUDA device visible on {os.uname().nodename})."
        )
    # A login node WITH a GPU (H100 PCIe) passes the check above but draws a
    # different sensor layout; require a SLURM job on the SXM SKU as well.
    require_compute_node()
    infix = f"_{args.tag}" if args.tag else ""

    os.environ.setdefault("JHU_SPLIT_MODE", "block")
    os.environ.setdefault("JHU_SPLIT_GAP", "0")
    os.environ["JHU_AUGMENT"] = ""      # never augment at evaluation
    device = torch.device(args.device)
    out = Path(args.out_dir)
    (out / "Evaluation").mkdir(parents=True, exist_ok=True)

    # Constructed exactly as ensemble_eval.load_run does.
    ds_kwargs = dict(train_ratio=args.train_ratio, seed=42,
                     field_names=("Ux", "Uy", "Uz", "p"), time_stride=1,
                     stats_path=str(out / "dataset_stats.pt"))
    TurbulentCombustionH5Dataset(args.data, split="train", **ds_kwargs)  # writes TRAIN stats
    dataset = TurbulentCombustionH5Dataset(args.data, split="val", **ds_kwargs)
    n_fields = int(dataset.num_fields)
    field_names = [str(n) for n in getattr(dataset, "field_names",
                                           [f"f{i}" for i in range(n_fields)])]

    decoder, stats, s1 = load_stage1(args.stage1_ckpt, device)
    denoiser, latent_min, latent_max, window = load_stage2(
        args.stage2_ckpt, device, int(s1["architecture"]["latent_dim"]), args.smoke_window)
    latent_dim = int(s1["architecture"]["latent_dim"])

    coords_full = dataset[0]["coords"].to(device) * 2.0 - 1.0
    # The unified trainer normalised coordinates with its own per-axis min/max.
    # If the two disagree the decoder is being queried off-manifold, so fail loud.
    drift = float((coords_full.amin(0) + 1.0).abs().max() + (coords_full.amax(0) - 1.0).abs().max())
    if drift > 1e-4:
        raise RuntimeError(f"coordinate convention mismatch (drift {drift:.3e})")

    sampler = create_sampler(sampler="ddpm", steps=args.steps, noise_schedule="cosine",
                             model_mean_type="epsilon", model_var_type="fixed_large",
                             dynamic_threshold=False, clip_denoised=True,
                             rescale_timesteps=False, timestep_respacing="")
    noiser = get_noise(sigma=0.0, name="gaussian")

    rng = np.random.default_rng(args.seed)
    snap_ids = rng.choice(len(dataset), size=min(args.n_snapshots, len(dataset)),
                          replace=False)
    scored = sorted(int(s) for s in snap_ids)

    # Windows of `window` CONSECUTIVE snapshots covering every scored snapshot.
    starts, cursor = [], 0
    while cursor < len(dataset):
        start = min(cursor, max(0, len(dataset) - window))
        starts.append(start)
        if start + window >= len(dataset):
            break
        cursor = start + window
    print(f"[confild-eval] windows {starts} (length {window}) covering "
          f"{len(scored)} scored snapshots of {len(dataset)}", flush=True)

    # --- sensors: canonical seeding, drawn per snapshot, never per window -----
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
        # Cross-baseline sensor-layout fingerprint. Must match the canonical
        # reference; a mismatch means a different inverse problem, not just a
        # different sample, because DPS is driven by these values.
        print(f"[seedcheck] snap={int(snap)} sensors={int(om[0].sum())} "
              f"idx_sum={int(oi[0][valid].sum())}", flush=True)
        check_canonical_fingerprint(int(snap), int(om[0].sum()),
                                    int(oi[0][valid].sum()), args.seed,
                                    args.cond_fields, args.n_obs)
        sensors[int(snap)] = (oi[0, valid].long(), ofid[0, valid].long())

    widths = {int(v[0].numel()) for v in sensors.values()}
    if len(widths) != 1:
        raise RuntimeError(f"sensor counts differ across snapshots {sorted(widths)}; "
                           "window-joint DPS requires a fixed count (1% per channel).")
    n_sensor = widths.pop()

    results, done, band_records = {}, set(), []
    for start in starts:
        rows = list(range(start, min(start + window, len(dataset))))
        s_idx = torch.stack([sensors[r][0] for r in rows])          # [W, M]
        s_fld = torch.stack([sensors[r][1] for r in rows])          # [W, M]
        s_coords = coords_full[s_idx]                               # [W, M, 3]

        # Measurements at sensor points only. dataset fields are z-scored, the
        # decoder works in stats' [-1, 1] units, so undo the z-score first.
        meas = []
        for w, r in enumerate(rows):
            ids = s_idx[w]
            z_at = dataset[r]["fields"][ids.cpu()].to(device)        # [M, F]
            raw_at = z_at * stats.benchmark_std + stats.benchmark_mean
            pm1 = stats.normalize(raw_at, ids)                       # [M, F]
            meas.append(pm1.gather(1, s_fld[w].unsqueeze(-1)).squeeze(-1))
        y = torch.cat(meas).reshape(1, -1)                           # [1, W*M]

        op = WindowSensorOperator(decoder, s_coords, s_fld, latent_min, latent_max,
                                  args.row_chunk)
        cond = get_conditioning_method(name="ps", operator=op, noiser=noiser,
                                       scale=args.dps_scale)
        sample_fn = partial(sampler.p_sample_loop, model=denoiser,
                            measurement_cond_fn=partial(cond.conditioning),
                            record=False, save_root=None)

        # Only decode rows this window is responsible for scoring: the second
        # window overlaps the first by `window - stride`, and a full-field
        # decode is 1.95M points.
        needed = [w for w, r in enumerate(rows) if r in scored and r not in done]
        if not needed:
            continue
        ens = np.zeros((args.K, len(needed), dataset.num_points, n_fields), dtype=np.float32)
        for k in range(args.K):
            torch.manual_seed(args.op_seed + 100003 * k + start)
            z0 = torch.randn(1, 1, len(rows), latent_dim, device=device)
            z_n = sample_fn(x_start=z0, measurement=y)
            latents = op.unnorm(z_n.detach()).reshape(len(rows), latent_dim)
            with torch.no_grad():
                for slot, w in enumerate(needed):
                    pieces = []
                    for s in range(0, dataset.num_points, args.chunk):
                        c = coords_full[s:s + args.chunk].unsqueeze(0)
                        pieces.append(decoder(c, latents[w].view(1, 1, -1))[0])
                    pm1 = torch.cat(pieces, 0)
                    phys = stats.denormalize(pm1)
                    ens[k, slot] = stats.benchmark_normalize(phys).float().cpu().numpy()
            print(f"[confild-eval] window {start} member {k+1}/{args.K}", flush=True)

        for slot, w in enumerate(needed):
            snap = rows[w]
            done.add(snap)
            truth_np = dataset[snap]["fields"].float().numpy()
            m = ensemble_metrics(ens[:, slot], truth_np, field_names)
            m["spectral_bands_ensemble_mean"] = spectral_bands(
                ens[:, slot].mean(axis=0), truth_np, field_names)
            m["spectral_bands_member0"] = spectral_bands(
                ens[0, slot], truth_np, field_names)
            m["snapshot"] = int(snap)
            m["window_start"] = int(start)
            agg = m["aggregate"]
            print(f"[confild-eval] snap {snap} " +
                  " ".join(f"{a}={b:.5f}" for a, b in agg.items()), flush=True)
            json.dump(m, open(out / "Evaluation" / f"crps{infix}_snap{snap}.json", "w"), indent=1)
            if not args.no_figs:
                try:
                    zslice_figure(out / "Evaluation" / f"zslice{infix}_snap{snap:03d}.png",
                                  ens[:, slot], truth_np, field_names, snap, agg)
                except Exception as exc:  # figures must never kill the metrics
                    print(f"[confild-eval] figure failed for snap {snap}: {exc}",
                          flush=True)
            band_records.append(m)
            results[int(snap)] = agg

    keys = sorted({k for v in results.values() for k in v})
    mean = {k: float(np.mean([v[k] for v in results.values() if k in v])) for k in keys}
    band_mean = {}
    for which in ("spectral_bands_ensemble_mean", "spectral_bands_member0"):
        band_mean[which] = {
            f"{name}.{band}": float(np.mean([bands[which][name][band]
                                             for bands in band_records]))
            for name in field_names for band in SPECTRA_BANDS
        } if band_records else {}
    for which, table in band_mean.items():
        print(f"[confild-eval] {which} " +
              " ".join(f"{k}={v:.4f}" for k, v in sorted(table.items())), flush=True)
    summary = {"per_snapshot": results, "mean": mean,
               "spectral_bands_mean": band_mean, "n_scored": len(results),
               "n_sensor_total": int(n_sensor), "window_length": int(window),
               "windows": starts, "K": args.K, "seed": args.seed, "op_seed": args.op_seed,
               "smoke": not bool(args.stage2_ckpt)}
    json.dump(summary, open(out / "Evaluation" / f"crps{infix}_summary.json", "w"), indent=1)
    print("[confild-eval] mean " + " ".join(f"{k}={v:.5f}" for k, v in mean.items()), flush=True)


if __name__ == "__main__":
    main()
