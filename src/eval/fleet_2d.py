#!/usr/bin/env python3
"""Matched-protocol posterior-ensemble evaluation for the 2D Kolmogorov fleet.

One driver for all three trained models (DMF-Gen point-cloud FFM, latent-FM
stage 2, SiT -- patch-tokenizer on the grid, or the point-tokenizer "SiT-point"
of the 3D row, sampled on the run's own discretization with the per-step
re-chunked point sampler), following the post-audit canonical pattern of
eval_latentfm_ensemble.py / eval_sit_ensemble.py rather than the env-gated
ENSEMBLE_K hook in model_baseline.py, because for the 2D fleet the hook path
is unusable for a matched protocol:

  * the hook draws sensors through helpers_baseline.build_sparse_condition,
    which is NOT RNG-equivalent to the canonical helpers.py draw (the baseline
    variant burns a CUDA randint before the randperm; see
    eval_sit_ensemble.py:25-32) -- layouts would differ from ours' eval;
  * nothing seeds the draw per snapshot in the visualize path, so layouts
    would not even be reproducible;
  * for the SiT *patch* tokenizer the hook never fires at all
    (model_baseline.py gates it on `_draw_point_sample is not None`, which is
    only set for the pointnet tokenizer).

This driver instead draws sensors ONCE per (snapshot, n_obs) via the
canonical path -- torch.manual_seed(seed * 777 + snap) immediately before
helpers.build_sparse_condition -- and feeds the identical sensor set to
whichever model it is scoring, so sensor layouts are bit-identical across all
three models (same job, same GPU SKU). Ensemble noise follows the canonical
convention: per-snapshot base = seed * 131 + si (si = position in the frame
list), torch.manual_seed(base * 10_000 + k) before sample k. Metrics are
ensemble_eval.ensemble_metrics, verbatim.

Frames are FIXED and evenly spaced over the TEST half of the held-out block
(block mode, gap 0; even held-out indices, protocol):
    snap_i = (i * len(test)) // n_frames,  i = 0..n_frames-1
(with len(test) = 320 this is i*6.4 floored: 0, 6, 12, 19, 25, ...).
This replaces ensemble_eval.main()'s rng.choice snapshot selection -- the
random-selection convention is deliberately NOT used here so every model and
every sensor density scores the same frames; everything downstream of the
frame choice keeps the canonical seeding contract.

2D adaptations (vs the 3D/JHU drivers):
  * no 3D grid helpers -- build_obs_grid_mask / grid_to_pointcloud (2D);
  * the JHU canonical fingerprint gate does not apply (different protocol
    tuple); we print + store [seedcheck] sensors/idx_sum per snapshot as the
    2D fingerprint instead;
  * no spectra / clamped z-slab extras; the diagnostic spread figure
    (save_ensemble_figure) already handles 2D and is kept, every --fig-every.

Cost fields: k==0 sample wall-clock (CUDA-synchronized) and peak GPU memory
are recorded per snapshot and summarized in every JSON. For DMF-Gen the K
samples are drawn in one sample_ensemble call, so we record total/K as
seconds per field and note the difference.

Datasets: originally 2D Kolmogorov; generalized to the cylinder2d fleet.
Each run's own config supplies its data path, so mesh-trained models
(dmfgen / senseiver / mlp_rbf / point-SiT / the `deform` Geo-FNO on
Cylinder2D_mesh.h5, 23,800 points) and grid-trained models (patch-SiT / s3gm /
the `fno` Geo-FNO / latent_fm on Cylinder2D_grid.h5, 400x200) each evaluate on their training discretization; sensor layouts are
bit-identical WITHIN each discretization family (same n_pts, same seeds) and
count-matched (same n_obs) across families -- recorded per-JSON via
data_path / num_points. --stratify-blocks B splits the val block into B
equal sub-blocks with even spacing inside each (cylinder: B=2 for the two
held-out Re trajectories). --cond-fields generalizes the observed-channel
set (cylinder: [0, 1] = Ux, Uy observed; p unobserved = the 2D
identifiability probe, reported per-field).

Deterministic models (--model senseiver | mlp_rbf | geofno): one forward per
snapshot,
scored the way the JHU canonical eval scores deterministic rows
(eval_senseiver_iclr.det_ensemble): the prediction is tiled into TWO identical
ensemble members so ensemble_eval's fair CRPS estimator and ddof=1 spread are
well defined -- the pair term vanishes exactly, so crps == MAE and
spread == 0 -- then the dispersion fields (spread, spread_error_ratio,
coverage_50/90, rank_hist) are set to null in the output because they carry
no information for a point predictor. K is reported as 1.

S3GM (--model s3gm): its own predictor-corrector DPS sampler
(model_baseline.dps_sample) with the RUN CONFIG's sampling params
(sampling_N / snr / n_corrector_steps / alpha_obs -- i.e. the 2D run's own
values, never 3D-tuned defaults; they are recorded in the JSON). ONE
documented deviation from the per-k seeding convention: the K samples are
drawn as a single batched PC chain (shape_5d[0] = K) seeded once with
torch.manual_seed(base * 10_000), because K sequential 1200-net-eval DPS
chains per snapshot would not fit any backfill wall. dps_sample broadcasts
the (1, ...) observation grids over the K-batch, so members differ only in
their noise draws, exactly as sequential chains would. --s3gm-sampler selects
two alternatives with the same seeding: the same chain plus a final Tweedie
step (pc_dps_denoised), or the upstream sampler (upstream).

--resume skips any snapshot whose crps_snapN.json already exists (baselines
only), so a timed-out job can be resubmitted and continues where it stopped.

Upstream GL_rbf_CQ (--model dmfgen_cq): a run of the upstream
PhyCoFlow-MultiFieldReconstruction trainer, sampled through eval.upstream_cq on
the frames, sensors and scoring of this driver. The K members are one batched call
seeded at base, as for DMF-Gen; the run selects its own (EMA) evaluation weights.

Touches no shared module.
"""
from __future__ import annotations
from paths import REPO_ROOT

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from paths import RUNS_DIR
import torch

# Write-time identity guard: refuses to let one measurement overwrite another's
# JSON, independent of whether the filename was built correctly. See the module.
from eval.artifact_guard import exit_if_conflicts, safe_write_json
import methods.common.config as bconfig
import methods.common.data as bdata
import methods.common.registry as bregistry
import methods.s3gm.adapter as s3gm_adapter
import methods.sit.adapter as sit_adapter
from protocol import PROTOCOL_2D, count_parameters, frame_list

# Sensor values are never written back into any method's prediction (plan 0.5);
# the flag is stamped into every JSON so a row that says otherwise is visible.
CLAMP_HARD = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Matched 2D Kolmogorov fleet ensemble eval")
    p.add_argument("--model", required=True,
                   choices=["dmfgen", "dmfgen_cq", "latent_fm", "sit", "senseiver",
                            "mlp_rbf", "mlprbf", "geofno", "s3gm"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best")
    p.add_argument("--split", default="test", choices=["test", "tune", "val"],
                   help="Held-out half to score: test (reported numbers), tune "
                        "(selection / sweeps), val (whole held-out block).")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--nfe", type=int, default=None,
                   help="ODE steps. Default: 4 for dmfgen/latent_fm; the run "
                        "config's sampling_N for sit/s3gm (their configured "
                        "benchmark step counts); 1 for deterministic models.")
    p.add_argument("--n-obs-list", type=int, nargs="+", default=[655],
                   help="Sensor counts to evaluate (per conditioned field). "
                        "More than one entry = a density sweep; supported for "
                        "every model (one JSON per density).")
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0],
                   help="Observed field ids (kolm: [0]; cylinder: [0, 1]).")
    p.add_argument("--cond-source", default="points", choices=["points", "surface"],
                   help="points: canonical draw over the whole point set. surface: "
                        "sensors restricted to the data file's surface_indices pool "
                        "(cylinder surface-to-field task); the n-obs sweep is then "
                        "allowed for every model and counts are capped at the pool.")
    p.add_argument("--n-frames", type=int, default=50)
    p.add_argument("--stratify-blocks", type=int, default=1,
                   help="Split the val block into B equal sub-blocks and "
                        "space frames evenly inside each (cylinder: 2, one "
                        "per held-out Re).")
    p.add_argument("--out-prefix", default="kolm_fleet",
                   help="Combined-JSON prefix (kolm_fleet / cyl_fleet).")
    p.add_argument("--resume", action="store_true",
                   help="Skip snapshots whose crps_snapN.json already exists "
                        "(baselines only); their stored metrics are reused.")
    p.add_argument("--expect-val-len", type=int, default=320,
                   help="Abort if the split length differs (protocol guard; "
                        "Kolmogorov TEST 320, cylinder TEST 300).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sensor-noise", type=float, default=0.0,
                   help="Gaussian noise added to sensor VALUES, in standardized "
                        "(z-score) units, drawn from the op_seed generator so "
                        "every method sees bit-identical corrupted readings.")
    p.add_argument("--sensor-occlusion", type=float, default=0.0,
                   help="fraction of sensors removed as one contiguous spatial "
                        "slab (0.25 = the protocol's 25%% slab occlusion). "
                        "Models a blocked or failed instrument cluster, which "
                        "is a different failure from i.i.d. sensor loss.")
    p.add_argument("--sensor-dropout", type=int, nargs="*", default=None,
                   help="cond-field ids whose sensors are removed entirely "
                        "(channel dropout). Single-channel regimes cannot use "
                        "this; the guard below refuses it rather than emitting "
                        "an empty condition.")
    p.add_argument("--op-seed", type=int, default=1000,
                   help="Protocol symmetry only; a no-op here (noise=0, no ops).")
    p.add_argument("--raw-weights", action="store_true",
                   help="Score the raw (non-EMA) weights of a baseline that keeps an "
                        "EMA copy; the choice is made on TUNE and stamped as `weights`.")
    p.add_argument("--sit-chunked", action="store_true",
                   help="SiT-point only: sample each token chunk as an independent "
                        "ODE instead of the canonical per-step sampler, which "
                        "integrates all points as one state and redraws the chunk "
                        "partition at every drift evaluation. Stamped as "
                        "`sit_sampler` and in the protocol string.")
    p.add_argument("--band-metrics", action="store_true",
                   help="Kolmogorov only: per-frame band energy ratio of single "
                        "members and of the ensemble mean vs truth, and band "
                        "correlation (spectral_utils.band_metrics, bands of "
                        "fig_spectra_kolmogorov.py). Stamped in the protocol "
                        "string.")
    p.add_argument("--s3gm-sampler", default="pc_dps",
                   choices=["pc_dps", "pc_dps_denoised", "upstream"],
                   help="S3GM only. pc_dps: adapter.dps_sample with the run config's "
                        "settings (the reported rows). pc_dps_denoised: the same chain "
                        "plus one Tweedie step at its terminal level sigma(eps). "
                        "upstream: S3GM's own guided sampler (s3gm3d.s3gm_reconstruct, "
                        "bit-identical to upstream on its native case) on a one-frame "
                        "window: predictor only on the discrete VE ladder of --nfe "
                        "steps, ending at the denoised mean, guidance weight "
                        "alpha_case/sqrt(observed fraction). Stamped in the protocol.")
    p.add_argument("--s3gm-alpha-case", type=float, default=0.5,
                   help="alpha_case of --s3gm-sampler upstream (published value 0.5).")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Score every S-th frame of the frame list, keeping each "
                        "frame's canonical sensor and noise seeds (budgeted sweeps). "
                        "Stamped in the protocol.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <run-dir>/Evaluation).")
    p.add_argument("--fig-every", type=int, default=10)
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--dry-run", action="store_true",
                   help="CPU-safe: resolve config + dataset + frame list, "
                        "print the plan, touch no GPU, build no model.")
    # ---- field-dump mode (additive; no behavior change without the flags) ----
    p.add_argument("--sensor-indices-npz", default=None,
                   help="dump mode only: INJECT the sensor set recorded in this "
                        "npz instead of drawing. CUDA randperm is not portable "
                        "across GPU SKUs at every size, so this is how a dump on "
                        "this machine reproduces the draw another machine scored.")
    p.add_argument("--dump-frame", type=int, default=None,
                   help="Dump mode: index INTO THE SPLIT of the single frame "
                        "to reconstruct and save to --dump-npz. The canonical "
                        "sensor draw (torch.manual_seed(seed*777+snap)) and, "
                        "when the frame is one of the canonical evenly-spaced "
                        "eval frames, the canonical ensemble-noise seeding "
                        "(base = seed*131 + si) are reused, so the dumped "
                        "sample/mean/std cross-reference the eval JSONs "
                        "exactly. Skips the 50-frame protocol loop entirely.")
    p.add_argument("--dump-npz", default=None,
                   help="Dump mode: output .npz path (required with "
                        "--dump-frame).")
    return p.parse_args()


def require_block_split() -> None:
    if os.environ.get("JHU_SPLIT_MODE") != "block" or \
       os.environ.get("JHU_SPLIT_GAP") != "0":
        raise SystemExit(
            "[guard] JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 must be exported "
            "(protocol: trajectory-holdout val block). Refusing to run.")


DISPERSION_KEYS = ("spread", "spread_error_ratio", "coverage_50", "coverage_90", "coverage_minmax")

# Deterministic point predictors: single forward, det-tiled scoring,
# dispersion fields nulled, K reported as 1.
DET_MODELS = {"senseiver", "mlp_rbf", "geofno"}

# dmfgen_cq is scored on the FRB dataset of the DMF-Gen row (data file, split, statistics).
CQ_DATA_CONFIG = REPO_ROOT / "configs/kolmogorov2d/config_bench_kolm_ffm.yaml"

# --band-metrics: the large- and small-scale bands of fig_spectra_kolmogorov.py
# plus the band between them. The vorticity field is the full periodic box, so
# no window, and k_power=-2 turns |omega_hat|^2 into the kinetic energy
# spectrum that figure plots.
KOLM_BANDS = {"large": (1, 8), "mid": (16, 31), "small": (32, 128)}


def mean_std(per_snap: list[dict], path: list[str]):
    vals = []
    for m in per_snap:
        d = m
        for k in path:
            d = d[k]
        if d is None:            # nulled dispersion field (deterministic model)
            return None, None
        vals.append(float(d))
    return float(np.mean(vals)), float(np.std(vals))


def null_dispersion(m: dict) -> dict:
    """Deterministic row: dispersion metrics carry no information -- null them
    (mirrors how the JHU canonical deterministic evals report only error keys)."""
    for d in [m["aggregate"], *m["per_field"].values()]:
        for k in DISPERSION_KEYS:
            d[k] = None
    m["rank_hist"] = None
    m["deterministic"] = True
    return m


def summarize(per_snap: list[dict], field_names: list[str]) -> dict:
    agg_keys = list(per_snap[0]["aggregate"].keys())
    fld_keys = list(per_snap[0]["per_field"][field_names[0]].keys())
    return {
        "aggregate": {k: mean_std(per_snap, ["aggregate", k])[0] for k in agg_keys},
        "aggregate_std": {k: mean_std(per_snap, ["aggregate", k])[1] for k in agg_keys},
        "per_field": {f: {k: mean_std(per_snap, ["per_field", f, k])[0]
                          for k in fld_keys} for f in field_names},
    }


# ---------------------------------------------------------------------------
# Model loading (per family)
# ---------------------------------------------------------------------------

def load_dmfgen(run_dir: Path, ckpt: str, device: str, split: str):
    from methods.dmfgen.ensemble_eval import load_run
    ckpt_name = ckpt if ckpt.endswith(".pt") else f"{ckpt}.pt"
    model, dataset, cfg = load_run(str(run_dir), ckpt_name, device, split=split)
    epoch = int(torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=False).get("epoch", -1))
    return model, dataset, cfg, epoch


def load_dmfgen_cq(run_dir: Path, ckpt: str, device: str, nfe: int, split: str, K: int,
                   raw_weights: bool = False):
    """(UpstreamRun, FRB dataset, data config, batch, members) for an upstream GL_rbf_CQ
    run. batch(snap, fields, ov, oi, ofid, om) holds K copies of split frame `snap`
    conditioned on the normalized sensor tensors of build_sparse_condition;
    members(phys) maps sampled physical members [K, N, C] to normalized numpy."""
    import yaml
    from methods.dmfgen.helpers import TurbulentCombustionH5Dataset
    from eval.upstream_cq import UpstreamRun
    up = UpstreamRun(run_dir, ckpt, device, nfe, weights="live" if raw_weights else "configured")
    cfg = yaml.safe_load(open(CQ_DATA_CONFIG))
    dataset = TurbulentCombustionH5Dataset(
        str(REPO_ROOT / cfg["data"]), split=split, train_ratio=cfg["train_ratio"],
        seed=cfg["seed"], field_names=tuple(cfg["field_names"]), time_stride=cfg["time_stride"])

    def batch(snap, fields, ov, oi, ofid, om):
        valid = om[0].bool()
        fid = ofid[0, valid].cpu()
        truth = fields[0].cpu() * dataset.std + dataset.mean
        readings = ov[0, valid, 0].cpu() * dataset.std[fid] + dataset.mean[fid]
        return up.batch(int(dataset.indices[snap]), truth, oi[0, valid].cpu(), fid, readings, K)

    def members(phys):
        return ((phys.cpu() - dataset.mean) / dataset.std).float().numpy()

    return up, dataset, cfg, batch, members


def load_baseline(run_dir: Path, ckpt: str, model_name: str, split: str):
    cfg = bconfig.validate_and_normalize_config(bconfig.load_yaml(run_dir / "run_config.yaml"))
    cfg["baseline_model"] = model_name
    cfg["training_stage"] = 2 if model_name == "latent_fm" else 1
    checkpoint = bconfig.safe_torch_load(run_dir / f"{ckpt}.pt", map_location="cpu")
    if model_name == "latent_fm" and checkpoint.get("ae_checkpoint"):
        ae = str(checkpoint["ae_checkpoint"])
        if not Path(ae).exists() and "Save_TrainedModel/" in ae:
            # Checkpoints carry the ABSOLUTE stage-1 path of the machine that
            # trained them, inside the .pt (not in any config), so a transferred
            # run points at a directory that does not exist here. Relocate it
            # under this checkout by its Save_TrainedModel-relative tail; the
            # weights are unchanged, only the path is rewritten, and we fail
            # loudly rather than silently picking a different checkpoint.
            tail = ae.split("Save_TrainedModel/", 1)[1]
            local = RUNS_DIR / tail
            if local.exists():
                print(f"[eval] stage1 ckpt relocated\n         from {ae}\n           to {local}",
                      flush=True)
                ae = str(local)
            else:
                raise SystemExit(
                    f"[eval] stage-1 checkpoint recorded in {run_dir.name}/{ckpt}.pt does not "
                    f"exist here and no local counterpart was found:\n  recorded: {ae}\n"
                    f"  looked for: {local}")
        cfg["latent_fm_params"]["stage2"]["stage1_checkpoint"] = ae
        print(f"[eval] stage1 ckpt {ae}", flush=True)
    device = bconfig.infer_device(None, cfg["shared"]["device_ids"])
    dataset = bdata.build_dataset(cfg, split=split, stats_path=run_dir / "dataset_stats.pt")
    adapter = bregistry.get_baseline_adapter(model_name)
    bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=run_dir,
                                        train_set=dataset, val_set=dataset)
    adapter.load_checkpoint(bundle, checkpoint)
    print(f"[eval] loaded {run_dir.name}/{ckpt}.pt "
          f"(epoch {checkpoint.get('epoch')})", flush=True)
    return adapter, bundle, dataset, cfg, device, int(checkpoint.get("epoch", -1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.model == "mlprbf":          # accept both spellings
        args.model = "mlp_rbf"
    require_block_split()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "Evaluation"
    if args.s3gm_sampler != "pc_dps" and args.model != "s3gm":
        raise SystemExit("[guard] --s3gm-sampler applies to --model s3gm only.")

    # The sweep used to be dmfgen-only because only that leg looped over
    # densities; the baseline branch now loops too (one JSON per density,
    # <prefix>_<model>_n<N>.json), so any model may sweep. Deterministic rows
    # still resume per-density via crps_n<N>_snap<M>.json.
    if False:
        raise SystemExit("[guard] unreachable")
    if args.sensor_indices_npz and args.dump_frame is None:
        raise SystemExit("[guard] --sensor-indices-npz is a dump-mode option.")
    if (args.dump_frame is None) != (args.dump_npz is None):
        raise SystemExit("[guard] --dump-frame and --dump-npz go together.")
    if args.dump_frame is not None and len(args.n_obs_list) != 1:
        raise SystemExit("[guard] dump mode takes exactly one --n-obs-list "
                         "value (broadcast per conditioned field).")

    # ---------------- dry run: CPU-safe plan check -------------------------
    if args.dry_run:
        if args.model == "dmfgen_cq":
            raise SystemExit("[dry-run] not available for dmfgen_cq (the upstream run loads on GPU)")
        if args.model == "dmfgen":
            from methods.dmfgen.evaluate import _normalize_eval_config
            from methods.dmfgen.helpers import TurbulentCombustionH5Dataset
            cfg = _normalize_eval_config(json.load(open(run_dir / "args.json")))
            dataset = TurbulentCombustionH5Dataset(
                cfg["data"], split=args.split,
                train_ratio=cfg.get("train_ratio", 0.9),
                field_names=cfg.get("field_names"),
                seed=cfg.get("seed", 42),
                time_stride=cfg.get("time_stride", 1),
                stats_path=str(run_dir / "dataset_stats.pt"))
            ckpt_file = run_dir / (args.ckpt if args.ckpt.endswith(".pt")
                                   else f"{args.ckpt}.pt")
            nfe = args.nfe if args.nfe is not None else 4
            data_path = cfg["data"]
        else:
            cfg = bconfig.validate_and_normalize_config(
                bconfig.load_yaml(run_dir / "run_config.yaml"))
            cfg["training_stage"] = 2 if args.model == "latent_fm" else 1
            dataset = bdata.build_dataset(cfg, split=args.split,
                                       stats_path=run_dir / "dataset_stats.pt")
            ckpt_file = run_dir / f"{args.ckpt}.pt"
            data_path = cfg["shared"]["paths"]["data_path"]
            if args.model in DET_MODELS:
                nfe = 1  # deterministic: single forward, no ODE
            elif args.nfe is not None:
                nfe = args.nfe
            elif args.model in ("sit", "s3gm"):
                nfe = int(bconfig.resolve_stage_config(cfg)["sampling"]["sampling_N"])
            else:
                nfe = 4
        n = len(dataset)
        frames = frame_list(n, args.n_frames, args.stratify_blocks)
        print(f"[dry-run] model={args.model} run={run_dir}")
        print(f"[dry-run] ckpt={ckpt_file} exists={ckpt_file.is_file()}")
        print(f"[dry-run] data={data_path}")
        print(f"[dry-run] split={args.split} len={n} "
              f"(expected {args.expect_val_len}) "
              f"fields={list(dataset.field_names)} points={dataset.num_points}")
        print(f"[dry-run] K={args.K} nfe={nfe} n_obs_list={args.n_obs_list} "
              f"cond_fields={args.cond_fields} seed={args.seed} "
              f"stratify_blocks={args.stratify_blocks} prefix={args.out_prefix}")
        print(f"[dry-run] frames ({len(frames)}): {frames}")
        print(f"[dry-run] out_dir={out_dir}")
        if n != args.expect_val_len:
            raise SystemExit(f"[dry-run] SPLIT LENGTH MISMATCH: {n} != "
                             f"{args.expect_val_len}")
        print("[dry-run] OK")
        return

    # ---------------- real run --------------------------------------------
    from methods.dmfgen.ensemble_eval import ensemble_metrics, require_compute_node, save_ensemble_figure
    from methods.dmfgen.helpers import build_sparse_condition
    require_compute_node()
    gpu_name = torch.cuda.get_device_name(0)
    sit_sampler = "chunked" if args.sit_chunked else "rechunk_per_step"
    sit_point = False
    out_dir.mkdir(parents=True, exist_ok=True)

    s3gm_params = None
    if args.model == "dmfgen":
        device = "cuda:0"
        model, dataset, cfg, ckpt_epoch = load_dmfgen(run_dir, args.ckpt, device, args.split)
        nfe = args.nfe if args.nfe is not None else 4
        solver = "euler"
        data_path = cfg["data"]
        n_params = count_parameters(model)["total"]
        train_range = [list(cfg["n_obs_min_list"]), list(cfg["n_obs_max_list"])]
    elif args.model == "dmfgen_cq":
        device = "cuda:0"
        nfe = args.nfe if args.nfe is not None else 4
        solver = "euler"
        up, dataset, cfg, cq_batch, cq_members = load_dmfgen_cq(
            run_dir, args.ckpt, device, nfe, args.split, args.K, args.raw_weights)
        data_path = cfg["data"]
        ckpt_epoch, train_range = up.epoch, up.train_range
        n_params = count_parameters(up.model)["total"]
    else:
        adapter, bundle, dataset, cfg, device, ckpt_epoch = load_baseline(
            run_dir, args.ckpt, args.model, args.split)
        n_params = count_parameters(bundle.model)["total"]
        _tc = cfg["shared"]["conditioning"]
        train_range = [list(_tc["n_obs_min_list"]), list(_tc["n_obs_max_list"])]
        import methods.common.helpers_baseline as HB
        data_path = cfg["shared"]["paths"]["data_path"]
        if args.model in DET_MODELS:
            solver = "none"
            nfe = 1  # deterministic: single forward, no ODE
        elif args.model == "s3gm":
            sampling_cfg = bconfig.resolve_stage_config(cfg)["sampling"]
            nfe = int(args.nfe if args.nfe is not None
                      else sampling_cfg["sampling_N"])
            solver = "pc_dps"
            s3gm_params = {
                "snr": float(sampling_cfg["snr"]),
                "n_corrector_steps": int(sampling_cfg["n_corrector_steps"]),
                "alpha_obs": float(sampling_cfg["alpha_obs"]),
                "note": ("guidance/sampling params taken from this run's own "
                         "config (2D values), not any 3D-tuned defaults; K "
                         "samples drawn as one batched PC chain seeded with "
                         "torch.manual_seed(base*10000) -- documented "
                         "deviation from per-k seeding"),
            }
            if args.s3gm_sampler == "pc_dps_denoised":
                solver = "pc_dps_denoised"
                s3gm_params["sampler"] = "pc_dps plus one Tweedie step at sigma(eps=1e-3)"
            elif args.s3gm_sampler == "upstream":
                solver = "s3gm_upstream"
                s3gm_params = {
                    "sampler": ("s3gm3d.s3gm_reconstruct, one-frame window: reverse-diffusion "
                                "predictor on the discrete VE ladder of nfe levels, no corrector, "
                                "returns the last predictor mean"),
                    "alpha_case": args.s3gm_alpha_case,
                    "alpha": "alpha_case / sqrt(observed points / (num_points * n_cond_fields))",
                    "note": ("K samples drawn as one batched chain seeded with "
                             "torch.manual_seed(base*10000)"),
                }
        else:
            sampling_cfg = bconfig.resolve_stage_config(cfg)["sampling"]
            solver = str(sampling_cfg["ode_solver"])
            if args.nfe is not None:
                nfe = args.nfe
            elif args.model == "sit":
                nfe = int(sampling_cfg["sampling_N"])
            else:
                nfe = 4
        num_x = int(cfg["shared"]["data"]["num_x"])
        num_y = int(cfg["shared"]["data"]["num_y"])
        n_fields = dataset.num_fields
        n_pts = dataset.num_points
        # Geo-FNO `deform` variant: the mesh-native model (sensors in at their own
        # coordinates, prediction at every mesh point), no grid pass.
        geofno_mesh = args.model == "geofno" and str(bundle.components["variant"]) == "deform"
        if geofno_mesh:
            from methods.geofno.adapter import predict_geofno_mesh
            print(f"[protocol] geofno variant=deform: mesh-native forward on {n_pts} points", flush=True)
        sit_point = args.model == "sit" and str(bundle.components["tokenizer"]) == "pointnet"
        if sit_point:
            _sit_chunk = int(bundle.components.get("node_subsample") or 0) or int(n_pts)
            _sit_sigma = float(bundle.components.get("cond_fill_sigma", 0.05))
            print(f"[protocol] sit tokenizer=pointnet: {sit_sampler} point sampler, "
                  f"chunk={_sit_chunk} of {n_pts} points, fill sigma={_sit_sigma}",
                  flush=True)
    if args.sit_chunked and not sit_point:
        raise SystemExit("[guard] --sit-chunked applies to the point-token SiT only.")
    if args.band_metrics:
        kolm_side = math.isqrt(int(dataset.num_points))
        if kolm_side ** 2 != int(dataset.num_points):
            raise SystemExit("[guard] --band-metrics needs the periodic square "
                             "Kolmogorov grid.")
        from methods.dmfgen.spectral_utils import band_metrics

        def kolm_grid(a):
            """[K, N, C] in point order (the grid raster) -> [K, C, side, side]."""
            return a.transpose(0, 2, 1).reshape(a.shape[0], a.shape[2], kolm_side, kolm_side)

    def _s3gm_draw(gv, gm):
        """K S3GM members as ONE batched chain under the caller's seed.
        gv, gm: [1, n_fields, H_pad, W_pad]. Returns (grid, before): the members
        [K, n_fields, H_pad, W_pad] and, for pc_dps_denoised only, the same chain
        before the Tweedie step (else None)."""
        sde = bundle.components["sde"]
        with torch.enable_grad():          # DPS needs the guidance gradient
            if args.s3gm_sampler == "upstream":
                # The 2D prior is a one-frame video model (frame_indices [0]), so a
                # snapshot is upstream's sampler with one window of one frame: no
                # overlap-consistency term, beta has no effect.
                from methods.s3gm import s3gm3d
                from methods.s3gm.sde import VESDE
                obs_frac = float(gm.sum()) / float(n_pts * len(args.cond_fields))
                ladder = VESDE(config=None, sigma_min=sde.sigma_min,
                               sigma_max=sde.sigma_max, N=nfe)
                out = s3gm3d.s3gm_reconstruct(
                    bundle.model, ladder, gv.unsqueeze(1), gm.unsqueeze(1), Nz=1,
                    K=args.K, t_win=1, overlap=0,
                    alpha=args.s3gm_alpha_case / math.sqrt(obs_frac), beta=0.0,
                    n_corrector_steps=0, device=device, progress=False)
                return out[:, 0], None
            grid = s3gm_adapter.dps_sample(
                net=bundle.model, sde=sde,
                shape_5d=(args.K, 1, n_fields,
                          int(bundle.components["H_pad"]),
                          int(bundle.components["W_pad"])),
                obs_value_grid=gv, obs_mask_grid=gm, device=device,
                N_steps=nfe, snr=s3gm_params["snr"],
                n_corrector_steps=s3gm_params["n_corrector_steps"],
                alpha_obs=s3gm_params["alpha_obs"])
        if args.s3gm_sampler != "pc_dps_denoised":
            return grid, None
        # dps_sample stops at sigma(eps=1e-3) ~ sigma_min and returns that noisy
        # iterate; one Tweedie step removes the remaining noise.
        x5 = grid.detach().unsqueeze(1)
        t_end = torch.full((args.K,), 1e-3, device=device)
        sig = sde.marginal_prob(x5, t_end)[1].view(-1, 1, 1, 1, 1)
        return (x5 + sig ** 2 * s3gm_adapter._score_fn_nograd(
            bundle.model, sde, x5, t_end)).squeeze(1), grid

    def _sit_point_draw(oc, ov, om, ofid, coords):
        """One posterior sample of the point-token SiT (SiT-point) for the
        given canonical sensor set: the adapter's point sampler with the
        training-time nearest-sensor fill, per-step re-chunked unless
        --sit-chunked. Returns a closure producing [1, N, n_fields] on the
        query points (the run's own discretization: grid or mesh)."""
        def _draw():
            return sit_adapter.sit_conditional_sample_points_chunked(
                net=bundle.model, transport=bundle.components["transport"],
                coords=coords, obs_coords=oc, obs_values=ov, obs_mask=om,
                obs_field_ids=ofid, n_fields=n_fields, device=device,
                n_steps=nfe, sampler_type=solver, chunk=_sit_chunk,
                sigma=_sit_sigma, rechunk=not args.sit_chunked)
        return _draw

    # Surface-to-field task: one fixed sensor pool per data file, drawn from
    # under the same seeding contract (helpers.build_sparse_condition valid_mask).
    surface_mask = None
    pool_size = None
    if args.cond_source == "surface":
        import h5py
        with h5py.File(data_path, "r") as _f:
            if "surface_indices" in _f:
                _pool = np.asarray(_f["surface_indices"][:], dtype=np.int64)
            else:
                _side = Path(data_path).with_suffix(".surface_indices.npy")
                if not _side.exists():
                    raise SystemExit(f"[surface] no surface_indices in {data_path} "
                                     f"and no sidecar {_side}")
                _pool = np.load(_side).astype(np.int64)
        surface_mask = torch.zeros(int(dataset.num_points), dtype=torch.bool)
        surface_mask[torch.from_numpy(_pool)] = True
        pool_size = int(surface_mask.sum())
        surface_mask = surface_mask.unsqueeze(0).to(device)
        trained_pool = getattr(dataset, "sensor_pool", None)
        print(f"[surface] pool={pool_size} of {dataset.num_points} points "
              f"(run trained with sensor_pool={trained_pool!r}); n_obs capped at pool",
              flush=True)
        if trained_pool != "surface":
            print("[surface] WARNING: this run was NOT trained with surface-only "
                  "conditioning; scoring it here is a train/test protocol mismatch",
                  flush=True)

    n_total = len(dataset)
    if n_total != args.expect_val_len:
        raise SystemExit(f"[guard] {args.split} split has {n_total} frames, "
                         f"expected {args.expect_val_len}. Wrong split env?")
    frames = frame_list(n_total, args.n_frames, args.stratify_blocks)
    # (si, snap) pairs to score; si stays the canonical position, which seeds the noise
    scored = list(enumerate(frames))[::args.frame_stride]
    scored_frames = [snap for _, snap in scored]
    field_names = list(dataset.field_names)
    print(f"[protocol] model={args.model} split={args.split} n={n_total} "
          f"data={data_path} frames={frames}", flush=True)
    print(f"[protocol] K={args.K} nfe={nfe} solver={solver} seed={args.seed} "
          f"op_seed={args.op_seed} n_obs_list={args.n_obs_list} "
          f"cond_fields={args.cond_fields} gpu={gpu_name}", flush=True)

    seeding_note = (
        "sensors: helpers.build_sparse_condition (canonical CPU-randint variant, "
        "NOT helpers_baseline's CUDA-randint variant) under "
        "torch.manual_seed(seed*777+snap); bit-identical across every model "
        "evaluated on the same data file / point count (same GPU SKU), and "
        "count-matched across discretizations. ensemble noise: "
        "base=seed*131+si, torch.manual_seed(base*10000+k) before sample k "
        "(s3gm: one batched chain seeded at base*10000 -- see s3gm_sampling). "
        "Frames are fixed evenly-spaced val indices "
        "(i*len(val)//n_frames, stratified across --stratify-blocks equal "
        "sub-blocks), not ensemble_eval.main()'s rng.choice.")

    # --- measurement-operator identity ---------------------------------------
    # The operator is part of WHAT IS MEASURED, so it has to appear in every key
    # derived from the measurement: the per-snapshot resume cache, the output
    # filename, and the protocol stamp. Two prior incidents this session came
    # from a sweep dimension that changed the result but not its key
    # (DELTA_STATUS sec.20, sec.21); this is the same hazard and is closed here
    # rather than after the fact.
    _ops = []
    if args.sensor_noise > 0:
        _ops.append(f"noise{args.sensor_noise:g}")
    if args.sensor_occlusion > 0:
        _ops.append(f"occl{args.sensor_occlusion:g}")
    if args.sensor_dropout:
        _ops.append("drop" + "x".join(str(int(f)) for f in args.sensor_dropout))
    op_tag = "_".join(_ops)                      # "" when the operator is clean
    op_suffix = f"_{op_tag}" if op_tag else ""
    # A corrupted-sensor run is NOT on the clean protocol and must never be
    # picked up by anything filtering for it (the density figure does exactly
    # that). Give it its own protocol string.
    PROTOCOL = (PROTOCOL_2D if not op_tag
                else f"{PROTOCOL_2D}_op_{op_tag}")
    # The chunked SiT sampler and the band diagnostics re-score canonical
    # settings. Their own stamp keeps every payload-based reader (fleet_select,
    # the density figure, the operator table) from taking them for a second
    # canonical row, and keys their per-snapshot records apart.
    variant_tag = ("_sitchunked" if args.sit_chunked else "") + ("_band" if args.band_metrics else "")
    variant_tag += "_s3gmdenoised" if args.s3gm_sampler == "pc_dps_denoised" else ""
    if args.frame_stride > 1:
        variant_tag += f"_stride{args.frame_stride}"
    PROTOCOL += variant_tag
    # The upstream S3GM sampler is a tuning-half sampler choice like NFE or SiT's
    # weights: it stays on the protocol, and its stamp `s3gm_sampler` (an
    # artifact_guard identity key) and record names keep it apart from pc_dps.
    s3gm_sampler = (f"upstream_a{args.s3gm_alpha_case:g}"
                    if args.s3gm_sampler == "upstream" else None)
    sampler_tag = f"_s3gm{s3gm_sampler}" if s3gm_sampler else ""
    if op_tag:
        print(f"[operator] {op_tag}: noise={args.sensor_noise} "
              f"occlusion={args.sensor_occlusion} dropout={args.sensor_dropout} "
              f"op_seed={args.op_seed}; protocol stamped '{PROTOCOL}'", flush=True)

    weights_tag = "raw" if args.raw_weights else "ema_if_kept"
    # Per-snapshot cache records of the TEST protocol keep the canonical name; a
    # tuning-half sweep varies checkpoint / NFE / weights per run and keys its
    # records by them. SiT-point records of the canonical per-step sampler are
    # tagged too, so they never collide with the untagged records that older
    # runs of the chunked sampler wrote.
    cache_tag = ("" if args.split == "test"
                 else f"_{args.split}_{args.ckpt}_nfe{nfe}_{weights_tag}") + variant_tag + sampler_tag
    if sit_point and not args.sit_chunked:
        cache_tag += "_sitstep"

    def run_protocol(n_obs: int):
        """Evaluate all frames at one sensor count. Returns (per_snap, cost)."""
        per_snap, timings, mems = [], [], []
        fig_dir = out_dir / f"figs_{args.model}_K{args.K}_nfe{nfe}_n{n_obs}{op_suffix}{variant_tag}{sampler_tag}"
        for si, snap in scored:
            # The per-snapshot cache key must identify the WORK (snapshot x
            # density), not the shape of the invocation. Read both spellings;
            # always write the density-qualified one.
            crps_path = out_dir / f"crps{cache_tag}_n{n_obs}{op_suffix}_snap{int(snap)}.json"
            cache_path = crps_path
            if not cache_path.exists() and not op_tag and not cache_tag:
                # legacy spelling only ever described CLEAN sensors, so it may
                # never satisfy an operator run
                legacy = out_dir / f"crps_snap{int(snap)}.json"
                if legacy.exists():
                    cache_path = legacy

            cached = None
            if (args.resume and args.model != "dmfgen"
                    and cache_path.exists() and cache_path.stat().st_size > 0):
                m = json.loads(cache_path.read_text())
                # NEVER trust the filename for the density. The legacy spelling
                # carries no density at all, so resuming a 6554-sensor sweep
                # from it silently republished the canonical 655-sensor numbers
                # under an n_obs=6554 label (caught 2026-09-10; the aggregate
                # came back bit-identical to the canonical row, which is what
                # gave it away). Every cached record stores the density it was
                # computed at -- check it, and recompute on mismatch or absence.
                raw = m.get("n_obs")
                if isinstance(raw, (list, tuple)) and raw:
                    raw = raw[0]
                cached_n = int(raw) if isinstance(raw, (int, float)) else None
                cached_proto = m.get("protocol", "")
                same_setting = (m.get("split") == args.split and int(m.get("nfe", -1)) == int(nfe)
                                and m.get("weights", "ema_if_kept") == weights_tag
                                and m.get("ckpt", "best") == args.ckpt
                                and int(m.get("K", -1)) == (1 if args.model in DET_MODELS else args.K))
                if cached_n == int(n_obs) and cached_proto == PROTOCOL and same_setting:
                    cached = m
                else:
                    print(f"[resume] REJECT {cache_path.name}: recorded "
                          f"n_obs={cached_n} protocol={cached_proto} split={m.get('split')} "
                          f"nfe={m.get('nfe')} weights={m.get('weights')} ckpt={m.get('ckpt')} K={m.get('K')} "
                          f"!= requested n_obs={int(n_obs)} protocol={PROTOCOL} split={args.split} nfe={nfe} "
                          f"weights={weights_tag} ckpt={args.ckpt}; recomputing this snapshot", flush=True)

            if cached is not None:
                per_snap.append(cached)
                if cached.get("sample_seconds") is not None:
                    timings.append(float(cached["sample_seconds"]))
                if cached.get("peak_gpu_gb") is not None:
                    mems.append(float(cached["peak_gpu_gb"]))
                print(f"[resume] snap={snap} loaded {cache_path.name}", flush=True)
                continue

            item = dataset[int(snap)]
            coords = item["coords"].unsqueeze(0).to(device)
            fields = item["fields"].unsqueeze(0).to(device)

            n_cf = len(args.cond_fields)
            torch.manual_seed(args.seed * 777 + int(snap))
            oc, ov, om, oi, ofid = build_sparse_condition(
                coords_full=coords, fields_full=fields,
                cond_fields=list(args.cond_fields),
                n_obs_min=[n_obs] * n_cf, n_obs_max=[n_obs] * n_cf,
                valid_mask=surface_mask)
            # --- measurement operator ----------------------------------
            # Corruption happens HERE, once, on the assembled sensor set, so
            # every method downstream consumes bit-identical corrupted
            # observations -- that is the whole point of the operator axis and
            # it is only true if no model re-draws anything itself.
            opgen = torch.Generator(device=ov.device).manual_seed(
                args.op_seed + int(snap))
            if args.sensor_noise > 0:
                # sensor values are already z-scored, so sigma is in z-units
                ov = ov + args.sensor_noise * torch.randn(
                    ov.shape, generator=opgen, device=ov.device, dtype=ov.dtype)
            if args.sensor_occlusion > 0:
                # A CONTIGUOUS slab, not i.i.d. dropout: a blocked instrument
                # cluster removes a region, and losing one region is a strictly
                # harder problem than losing the same count at random, because
                # it creates a gap with no nearby data rather than thinning an
                # otherwise-covered field.
                live = om[0] > 0
                if int(live.sum()) > 0:
                    axis = int(torch.randint(oc.shape[-1], (1,),
                                             generator=opgen,
                                             device=ov.device).item())
                    pos = oc[0][:, axis]
                    order = torch.argsort(torch.where(live, pos,
                                                      torch.full_like(pos, float("inf"))))
                    n_live = int(live.sum())
                    n_cut = int(round(args.sensor_occlusion * n_live))
                    if n_cut > 0:
                        # slab start drawn uniformly, wrapping, so the removed
                        # band is not always at a domain edge
                        start = int(torch.randint(max(1, n_live), (1,),
                                                  generator=opgen,
                                                  device=ov.device).item())
                        cut = order[[(start + j) % n_live for j in range(n_cut)]]
                        om[0][cut] = 0
            if args.sensor_dropout:
                for f in args.sensor_dropout:
                    om[0][ofid[0] == int(f)] = 0
                if int(om.sum()) == 0:
                    raise SystemExit(
                        "[guard] --sensor-dropout removed every sensor. Channel "
                        "dropout needs more than one observed channel; this "
                        "regime has "
                        f"{len(args.cond_fields)}.")
            sensors = int(om.sum())
            idx_sum = int(oi[om.bool()].sum())
            print(f"[seedcheck] snap={snap} n_obs={n_obs} sensors={sensors} "
                  f"idx_sum={idx_sum}", flush=True)

            base = args.seed * 131 + si
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            if args.model == "dmfgen":
                from methods.dmfgen.ensemble_eval import sample_ensemble
                obs = {"coords": oc, "values": ov, "mask": om,
                       "indices": oi, "field_ids": ofid}
                t0 = time.perf_counter()
                ens = sample_ensemble(model, coords, obs, K=args.K,
                                      n_steps=nfe, chunk=args.chunk,
                                      clamp_hard=CLAMP_HARD, seed=base)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) / args.K)
                ens = ens.numpy()
            elif args.model == "dmfgen_cq":
                batch = cq_batch(snap, fields, ov, oi, ofid, om)
                t0 = time.perf_counter()
                phys = up.sample(batch, base)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) / args.K)
                ens = cq_members(phys)
            elif args.model in DET_MODELS:
                # Deterministic: one forward (pointwise chunked for
                # senseiver / mlp_rbf, mirroring
                # eval_senseiver_iclr.predict_full_field; dense grid pass for
                # geofno, mirroring visualize_reconstruction_deterministic),
                # scored as two identical members so fair CRPS == MAE and
                # spread == 0.
                with torch.no_grad():
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if args.model == "geofno" and geofno_mesh:
                        pred = predict_geofno_mesh(bundle.model, coords, oc, ov, om, ofid)
                    elif args.model == "geofno":
                        gv, gm = HB.build_obs_grid_mask(
                            ov, om, ofid, oi, n_fields, n_pts,
                            num_y, num_x, num_y, num_x,
                            point_to_grid=bundle.components.get("point_to_grid"))
                        pred_grid = bundle.model(gv, gm)
                        pred = HB.grid_to_pointcloud(
                            pred_grid, num_y, num_x,
                            point_to_grid=bundle.components.get("point_to_grid"))
                    else:  # senseiver / mlp_rbf: pointwise
                        n_q = coords.shape[1]
                        pred = torch.empty(1, n_q, n_fields,
                                           device=coords.device,
                                           dtype=coords.dtype)
                        for s in range(0, n_q, args.chunk):
                            e = min(s + args.chunk, n_q)
                            pred[:, s:e] = bundle.model(coords[:, s:e], oc, ov,
                                                        om, ofid)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - t0)
                ens = np.repeat(pred[0].detach().float().cpu().numpy()[None],
                                2, axis=0)
            elif args.model == "s3gm":
                # K samples as ONE batched predictor-corrector DPS chain
                # (documented deviation from per-k seeding; see module
                # docstring). dps_sample broadcasts the (1, ...) obs grids
                # over the K-batch and needs autograd for the DPS gradient.
                gv, gm = HB.build_obs_grid_mask(
                    ov, om, ofid, oi, n_fields, n_pts, num_y, num_x,
                    int(bundle.components["H_pad"]),
                    int(bundle.components["W_pad"]),
                    point_to_grid=bundle.components.get("point_to_grid"))
                torch.manual_seed(base * 10_000)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                grid, before = _s3gm_draw(gv, gm)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) / args.K)
                to_np = lambda g: HB.grid_to_pointcloud(
                    g, num_y, num_x,
                    point_to_grid=bundle.components.get("point_to_grid")).detach().float().cpu().numpy()
                ens = to_np(grid)
                ens_before = to_np(before) if before is not None else None
            else:
                if args.model == "latent_fm":
                    gv, gm = HB.build_obs_grid_mask(
                        ov, om, ofid, oi, n_fields, n_pts,
                        num_y, num_x, num_y, num_x)
                    cond = {"obs_value_grid": gv, "obs_mask_grid": gm}

                    def _draw():
                        grid = bundle.model.sample(cond, n_steps=nfe,
                                                   ode_solver=solver)
                        return HB.grid_to_pointcloud(grid, num_y, num_x)
                elif str(bundle.components["tokenizer"]) == "pointnet":
                    # SiT-point (the 3D JHTDB configuration, run here on the
                    # 2D regimes): the run's own point sampler with the same
                    # nearest-sensor fill as training, fed the canonical
                    # sensor set drawn above.
                    _draw = _sit_point_draw(oc, ov, om, ofid, coords)
                else:  # sit (patch tokenizer, grid path)
                    h_pad = int(bundle.components["H_pad"])
                    w_pad = int(bundle.components["W_pad"])
                    p2g = bundle.components.get("point_to_grid")
                    gv, gm = HB.build_obs_grid_mask(
                        ov, om, ofid, oi, n_fields, n_pts,
                        num_y, num_x, h_pad, w_pad, point_to_grid=p2g)
                    if str(bundle.components["cond_mode"]) == "interp":
                        gv = HB.nearest_fill_grid(gv, gm)

                    def _draw():
                        grid = sit_adapter.sit_conditional_sample(
                            net=bundle.model,
                            transport=bundle.components["transport"],
                            shape=(1, n_fields, h_pad, w_pad),
                            obs_value_grid=gv, obs_mask_grid=gm,
                            device=device, n_steps=nfe, sampler_type=solver)
                        return HB.grid_to_pointcloud(grid, num_y, num_x,
                                                     point_to_grid=p2g)

                ens = []
                with torch.no_grad():
                    for k in range(args.K):
                        torch.manual_seed(base * 10_000 + k)
                        if k == 0:
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                        r = _draw()
                        if k == 0:
                            torch.cuda.synchronize()
                            timings.append(time.perf_counter() - t0)
                        ens.append(r[0].detach().float().cpu().numpy())
                ens = np.stack(ens, axis=0)

            mems.append(torch.cuda.max_memory_allocated() / 1024 ** 3)
            truth_np = fields[0].detach().float().cpu().numpy()
            m = ensemble_metrics(ens, truth_np, field_names)
            if args.model in DET_MODELS:
                m = null_dispersion(m)
            if args.band_metrics:
                m["band"] = band_metrics(kolm_grid(ens), kolm_grid(truth_np[None])[0],
                                         KOLM_BANDS, window=False, k_power=-2)
            if args.model == "s3gm" and ens_before is not None:
                m["before_denoise"] = {
                    "aggregate": ensemble_metrics(ens_before, truth_np, field_names)["aggregate"],
                    **({"band": band_metrics(kolm_grid(ens_before), kolm_grid(truth_np[None])[0],
                                             KOLM_BANDS, window=False, k_power=-2)}
                       if args.band_metrics else {})}
            if sit_point:
                m["sit_sampler"] = sit_sampler
            if s3gm_sampler:
                # divergence check: a diverged member runs to |x| ~ 1e6-1e8 (z-units)
                m["s3gm_sampler"] = s3gm_sampler
                m["member_max_abs"] = [float(v) for v in np.abs(ens).reshape(len(ens), -1).max(axis=1)]
                m["truth_max_abs"] = float(np.abs(truth_np).max())
            m["snapshot"] = int(snap)
            m["si"] = si
            m["sensors"] = sensors
            m["idx_sum"] = idx_sum
            m["K"] = 1 if args.model in DET_MODELS else args.K
            m["nfe"] = nfe
            m["n_obs"] = n_obs
            m["protocol"] = PROTOCOL
            m["split"] = args.split
            m["ckpt"] = args.ckpt
            m["weights"] = weights_tag
            m["clamp_hard"] = CLAMP_HARD
            m["cond_source"] = args.cond_source
            m["sample_seconds"] = timings[-1]
            m["peak_gpu_gb"] = mems[-1]
            per_snap.append(m)

            if args.model != "dmfgen":
                safe_write_json(crps_path, m)

            if (not args.no_figs) and (si % max(1, args.fig_every) == 0):
                fig_dir.mkdir(parents=True, exist_ok=True)
                try:
                    cr = item.get("coords_raw")
                    cr = cr.cpu().numpy() if cr is not None else coords[0].cpu().numpy()
                    save_ensemble_figure(
                        ens, truth_np, cr, field_names,
                        fig_dir / f"spread_snap{int(snap)}.png",
                        tag=(f"{args.model} {args.ckpt}  snap {snap}  "
                             f"K={args.K} NFE={nfe} n_obs={n_obs}  "
                             f"relL2={m['aggregate']['rel_l2_mean']:.4f}"))
                except Exception as exc:  # a plot must never kill an eval
                    print(f"  [warn] figure snap {snap} failed: {exc}", flush=True)

            agg = m["aggregate"]
            print(f"[ensemble] snap={snap} n_obs={n_obs} K={args.K} " + " ".join(
                f"{k}={v:.5f}" for k, v in agg.items() if v is not None),
                flush=True)

        cost = {
            "inference_seconds_per_field_mean": float(np.mean(timings)),
            "inference_seconds_per_field_std": float(np.std(timings)),
            "inference_peak_gpu_gb": float(np.max(mems)),
            "timing_note": ("dmfgen and s3gm (batched chain): total sampling "
                            "wall-clock / K; other generative baselines: "
                            "k==0 draw wall-clock; deterministic models: the "
                            "single forward. CUDA-synchronized; diagnostic "
                            "figures excluded."),
        }
        return per_snap, cost

    # ------------------------------------------------------------------
    # dump mode: one frame -> npz (truth / sample / mean / std / sensors)
    # ------------------------------------------------------------------
    def dump_one(snap: int, n_obs: int, cond_fields: list[int],
                 out_path: Path) -> None:
        """Reconstruct ONE split frame and save the fields to npz.

        Mirrors the per-snapshot body of run_protocol exactly (same canonical
        sensor draw, same ensemble-noise convention) so the dumped fields are
        the very samples the eval JSONs scored -- provided (seed, K, nfe,
        n_obs, cond_fields) match and `snap` is one of the canonical
        evenly-spaced frames (then si = frames.index(snap)).
        """
        snap = int(snap)
        canonical = snap in frames
        si = frames.index(snap) if canonical else snap
        if not canonical:
            print(f"[dump] WARNING: frame {snap} is not in the canonical "
                  f"{len(frames)}-frame list; ensemble noise base uses "
                  f"si={si} (=snap) and does NOT cross-reference the eval "
                  "JSONs.", flush=True)
        item = dataset[snap]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)

        torch.manual_seed(args.seed * 777 + snap)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=list(cond_fields),
            n_obs_min=[n_obs] * len(cond_fields),
            n_obs_max=[n_obs] * len(cond_fields),
            valid_mask=surface_mask)          # the same pool the scored frames used
        if args.sensor_indices_npz:
            R = np.load(args.sensor_indices_npz, allow_pickle=False)
            ridx = torch.from_numpy(np.asarray(R["sensor_indices"], dtype=np.int64)).to(coords.device)
            rfid = torch.from_numpy(np.asarray(R["sensor_field_ids"], dtype=np.int64)).to(coords.device)
            if ridx.numel() != oi.shape[1]:
                raise SystemExit(f"[inject] recorded {ridx.numel()} sensors, draw has {oi.shape[1]}")
            # overwrite the draw's own tensors: keeps dtype/shape/padding exactly
            oi[0] = ridx.to(oi.dtype)
            ofid[0] = rfid.to(ofid.dtype)
            om[0] = 1
            oc[0] = coords[0, ridx]
            ov[0, :, 0] = fields[0, ridx, rfid]
            print(f"[inject] sensors taken from {args.sensor_indices_npz}", flush=True)
        valid = om[0].bool()
        sensors = int(om.sum())
        idx_sum = int(oi[om.bool()].sum())
        print(f"[seedcheck] snap={snap} n_obs={n_obs} "
              f"cond_fields={list(cond_fields)} sensors={sensors} "
              f"idx_sum={idx_sum}", flush=True)

        base = args.seed * 131 + si
        t0 = time.perf_counter()
        if args.model == "dmfgen":
            from methods.dmfgen.ensemble_eval import sample_ensemble
            obs = {"coords": oc, "values": ov, "mask": om,
                   "indices": oi, "field_ids": ofid}
            ens = sample_ensemble(model, coords, obs, K=args.K,
                                  n_steps=nfe, chunk=args.chunk,
                                  clamp_hard=CLAMP_HARD, seed=base).numpy()
        elif args.model == "dmfgen_cq":
            ens = cq_members(up.sample(cq_batch(snap, fields, ov, oi, ofid, om), base))
        elif args.model in ("senseiver", "mlp_rbf"):
            n_q = coords.shape[1]
            with torch.no_grad():
                pred = torch.empty(1, n_q, n_fields,
                                   device=coords.device, dtype=coords.dtype)
                for s in range(0, n_q, args.chunk):
                    e = min(s + args.chunk, n_q)
                    pred[:, s:e] = bundle.model(coords[:, s:e], oc, ov,
                                                om, ofid)
            ens = pred[0].detach().float().cpu().numpy()[None]
        elif args.model == "geofno" and geofno_mesh:
            with torch.no_grad():
                pred = predict_geofno_mesh(bundle.model, coords, oc, ov, om, ofid)
            ens = pred[0].detach().float().cpu().numpy()[None]
        elif args.model == "geofno":
            # Same dense grid pass run_protocol uses for this model; it was only
            # missing here, which kept the fleet's second-best surface-task row
            # out of the galleries.
            with torch.no_grad():
                gv, gm = HB.build_obs_grid_mask(
                    ov, om, ofid, oi, n_fields, n_pts,
                    num_y, num_x, num_y, num_x,
                    point_to_grid=bundle.components.get("point_to_grid"))
                pred_grid = bundle.model(gv, gm)
                pred = HB.grid_to_pointcloud(
                    pred_grid, num_y, num_x,
                    point_to_grid=bundle.components.get("point_to_grid"))
            ens = pred[0].detach().float().cpu().numpy()[None]
        elif args.model == "s3gm":
            # The batched predictor-corrector DPS chain run_protocol uses: K
            # members as ONE chain seeded at base*10000 (the documented
            # deviation from per-k seeding), so sample k=0 here is the one the
            # eval scored. Dump mode was guarded off only because this leg was
            # never written, which kept S3GM out of every gallery.
            h_pad = int(bundle.components["H_pad"])
            w_pad = int(bundle.components["W_pad"])
            p2g = bundle.components.get("point_to_grid")
            gv, gm = HB.build_obs_grid_mask(
                ov, om, ofid, oi, n_fields, n_pts, num_y, num_x,
                h_pad, w_pad, point_to_grid=p2g)
            torch.manual_seed(base * 10_000)
            grid, _ = _s3gm_draw(gv, gm)
            ens = HB.grid_to_pointcloud(
                grid, num_y, num_x,
                point_to_grid=p2g).detach().float().cpu().numpy()
        else:
            if args.model == "latent_fm":
                gv, gm = HB.build_obs_grid_mask(
                    ov, om, ofid, oi, n_fields, n_pts,
                    num_y, num_x, num_y, num_x)
                cond = {"obs_value_grid": gv, "obs_mask_grid": gm}

                def _draw():
                    grid = bundle.model.sample(cond, n_steps=nfe,
                                               ode_solver=solver)
                    return HB.grid_to_pointcloud(grid, num_y, num_x)
            elif str(bundle.components["tokenizer"]) == "pointnet":
                _draw = _sit_point_draw(oc, ov, om, ofid, coords)   # SiT-point
            else:  # sit (patch tokenizer, grid path)
                h_pad = int(bundle.components["H_pad"])
                w_pad = int(bundle.components["W_pad"])
                p2g = bundle.components.get("point_to_grid")
                gv, gm = HB.build_obs_grid_mask(
                    ov, om, ofid, oi, n_fields, n_pts,
                    num_y, num_x, h_pad, w_pad, point_to_grid=p2g)
                if str(bundle.components["cond_mode"]) == "interp":
                    gv = HB.nearest_fill_grid(gv, gm)

                def _draw():
                    grid = sit_adapter.sit_conditional_sample(
                        net=bundle.model,
                        transport=bundle.components["transport"],
                        shape=(1, n_fields, h_pad, w_pad),
                        obs_value_grid=gv, obs_mask_grid=gm,
                        device=device, n_steps=nfe, sampler_type=solver)
                    return HB.grid_to_pointcloud(grid, num_y, num_x,
                                                 point_to_grid=p2g)

            ens = []
            with torch.no_grad():
                for k in range(args.K):
                    torch.manual_seed(base * 10_000 + k)
                    ens.append(_draw()[0].detach().float().cpu().numpy())
                    print(f"[dump] member {k + 1}/{args.K} "
                          f"({time.perf_counter() - t0:.1f}s)", flush=True)
            ens = np.stack(ens, axis=0)
        K_eff = ens.shape[0]
        print(f"[dump] ensemble {ens.shape} in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)

        truth_np = fields[0].detach().float().cpu().numpy()
        # scored exactly like the eval (senseiver: 2 identical members)
        m = ensemble_metrics(np.repeat(ens, 2, axis=0) if K_eff == 1 else ens,
                             truth_np, field_names)
        if args.model in DET_MODELS:
            m = null_dispersion(m)
        print(f"[dump] snap={snap} " + " ".join(
            f"{k}={v:.5f}" for k, v in m["aggregate"].items()
            if isinstance(v, float)), flush=True)

        pred_mean = ens.mean(axis=0)
        pred_std = (ens.std(axis=0, ddof=1) if K_eff > 1
                    else np.zeros_like(ens[0]))

        ds_mean = getattr(dataset, "mean", None)
        ds_std = getattr(dataset, "std", None)
        try:
            abs_frame = int(np.asarray(dataset.indices)[snap])
        except Exception:
            abs_frame = None
        meta = {
            "protocol": f"{PROTOCOL_2D}_dump",
            "model": args.model,
            "sensor_source": (f"INJECTED from {Path(args.sensor_indices_npz).name}"
                              if args.sensor_indices_npz else "drawn on this GPU"),
            "run_dir": str(run_dir),
            "ckpt": args.ckpt,
            "split": args.split,
            "split_env": {k: os.environ.get(k)
                          for k in ("JHU_SPLIT_MODE", "JHU_SPLIT_GAP")},
            "split_len": n_total,
            "snapshot_index": snap,
            "absolute_frame": abs_frame,
            "canonical_frame": canonical,
            "si": si,
            "seed": args.seed,
            "sensor_seed_formula": f"torch.manual_seed({args.seed}*777+{snap})",
            "sensor_device": str(coords.device),
            "cond_fields": list(cond_fields),
            "n_obs": n_obs,
            "n_sensors": sensors,
            "sensor_idx_sum": idx_sum,
            "K": K_eff,
            "nfe": nfe,
            "ode_solver": solver,
            "weights": weights_tag,
            **({"sit_sampler": sit_sampler} if sit_point else {}),
            **({"s3gm_sampler": s3gm_sampler} if s3gm_sampler else {}),
            "noise_seeding": (f"base={args.seed}*131+{si}; "
                              "torch.manual_seed(base*10000+k) per member"
                              if args.model not in ("dmfgen", "dmfgen_cq") else
                              f"one batched call seeded at {base}"),
            "deterministic": args.model in DET_MODELS,
            "clamp_hard": CLAMP_HARD,
            "units": "STANDARDIZED (z-score with this run's train stats; "
                     "physical = arr * norm_std + norm_mean)",
            "pred_std_ddof": 1,
            "gpu": gpu_name,
            "metrics": m,
            "field_names": field_names,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save = dict(
            truth=truth_np.astype(np.float32),
            pred_mean=pred_mean.astype(np.float32),
            pred_sample=ens[0].astype(np.float32),
            pred_std=pred_std.astype(np.float32),
            coords=item["coords"].numpy().astype(np.float32),
            sensor_indices=oi[0, valid].long().cpu().numpy(),
            sensor_field_ids=ofid[0, valid].long().cpu().numpy(),
            names=np.array(field_names),
            meta=np.array(json.dumps(meta, indent=1)),
        )
        if ds_mean is not None and ds_std is not None:
            save["norm_mean"] = ds_mean.cpu().numpy().ravel().astype(np.float32)
            save["norm_std"] = ds_std.cpu().numpy().ravel().astype(np.float32)
        cr = item.get("coords_raw")
        if cr is not None:
            save["coords_raw"] = cr.numpy().astype(np.float32)
        np.savez_compressed(out_path, **save)
        print(f"[dump] wrote {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    import contextlib
    own_weights = args.model in ("dmfgen", "dmfgen_cq")   # no baseline bundle
    weights_ctx = (contextlib.nullcontext() if args.raw_weights or own_weights
                   else adapter.evaluation_weights(bundle))
    if args.dump_frame is not None:
        cf = args.cond_fields if args.cond_fields is not None else [0]
        with weights_ctx:
            if not own_weights:
                bundle.model.eval()
            dump_one(args.dump_frame, args.n_obs_list[0], cf, Path(args.dump_npz))
        return

    deterministic = args.model in DET_MODELS
    observed = [field_names[c] for c in args.cond_fields if c < len(field_names)]
    unobserved = [f for f in field_names if f not in observed]
    if len(field_names) == 1:
        field_note = ("single-field dataset: aggregate == per-field values")
    else:
        field_note = (f"multi-field dataset: observed={observed} "
                      f"unobserved={unobserved}; the unobserved per_field "
                      "columns are the 2D identifiability probe -- report "
                      "them alongside the aggregate")

    def payload_common(n_obs: int, per_snap: list[dict], cost: dict) -> dict:
        return {
            "protocol": PROTOCOL,
            "operator": {"noise_sigma_z": args.sensor_noise,
                         "occlusion_frac": args.sensor_occlusion,
                         "channel_dropout": args.sensor_dropout,
                         "op_seed": args.op_seed,
                         "tag": op_tag or "clean"},
            "model": args.model,
            "run_dir": str(run_dir),
            "ckpt": args.ckpt,
            "ckpt_epoch": ckpt_epoch,
            "weights": weights_tag,
            "clamp_hard": CLAMP_HARD,
            "n_params": n_params,
            "train_n_obs_range": train_range,
            "split": args.split,
            "split_env": {"JHU_SPLIT_MODE": "block", "JHU_SPLIT_GAP": "0"},
            "val_len": n_total,
            "n_frames": len(scored_frames),
            "frame_indices": scored_frames,
            "K": 1 if deterministic else args.K,
            "deterministic": deterministic,
            "coverage_ideal": per_snap[0].get("coverage_ideal"),
            **({"note_K": "deterministic model; metrics computed with two "
                          "identical members so the fair CRPS estimator is "
                          "well defined: CRPS == MAE exactly, "
                          "rel_l2_single == rel_l2_mean; dispersion fields "
                          "(spread, spread_error_ratio, coverage, rank_hist) "
                          "are null"} if deterministic else {}),
            "nfe": nfe,
            "ode_solver": solver,
            "seed": args.seed,
            "op_seed": args.op_seed,
            "cond_fields": list(args.cond_fields),
            "n_obs": [n_obs] * len(args.cond_fields),
            "cond_source": args.cond_source,
            **({"sensor_pool_size": pool_size,
                "surface_note": ("sensors restricted to the data file's "
                                 "surface_indices pool (mesh: 360 wall-adjacent "
                                 "cells; grid: their unique nearest fluid cells); "
                                 "per-field count capped at the pool size, so "
                                 "`sensors` in each snapshot is the effective "
                                 "count")} if args.cond_source == "surface" else {}),
            "stratify_blocks": args.stratify_blocks,
            "data_path": str(data_path),
            "num_points": int(dataset.num_points),
            "field_names": field_names,
            "observed_fields": observed,
            "unobserved_fields": unobserved,
            "gpu": gpu_name,
            "seeding": seeding_note,
            "field_note": field_note,
            **({"s3gm_sampling": s3gm_params} if s3gm_params else {}),
            **({"s3gm_sampler": s3gm_sampler} if s3gm_sampler else {}),
            **({"sit_sampler": sit_sampler} if sit_point else {}),
            **({"band_bands": KOLM_BANDS,
                "band_summary": {b: {k: float(np.mean([r["band"][b][k] for r in per_snap]))
                                     for k in per_snap[0]["band"][b]} for b in KOLM_BANDS},
                "band_note": ("per snapshot `band`: spectral_utils.band_metrics on the "
                              "periodic square vorticity grid, no window, kinetic energy "
                              "spectrum (k_power=-2); band_summary = mean over frames")}
               if args.band_metrics else {}),
            **cost,
            "summary": summarize(per_snap, field_names),
            "snapshots": per_snap,
        }

    if args.model == "dmfgen" and len(args.n_obs_list) == 1:
        # Single-density protocol run (e.g. the cylinder fleet): main JSON only.
        n_obs = args.n_obs_list[0]
        per_snap, cost = run_protocol(n_obs)
        payload = payload_common(n_obs, per_snap, cost)
        main_path = out_dir / f"{args.out_prefix}{op_suffix}_dmfgen_K{args.K}_nfe{nfe}.json"
        main_path = safe_write_json(main_path, payload)
        s = payload["summary"]["aggregate"]
        print(f"[RESULT] dmfgen relL2={s['rel_l2_mean']:.5f} "
              f"crps={s['crps']:.5f} "
              f"infer={cost['inference_seconds_per_field_mean']:.3f}s "
              f"peak={cost['inference_peak_gpu_gb']:.2f}GB", flush=True)
        print(f"[out] wrote {main_path}", flush=True)
    elif args.model == "dmfgen":
        rel_l2_by_n = {}
        cost_by_n = {}
        for n_obs in args.n_obs_list:
            per_snap, cost = run_protocol(n_obs)
            payload = payload_common(n_obs, per_snap, cost)
            # The historical name for the points sweep is hardcoded, which is
            # exactly how an operator run destroyed the clean DMF-Gen density
            # sweep (2026-09-10): it ignored out_prefix AND op_suffix, so three
            # corrupted-sensor runs wrote straight over the clean files. The
            # legacy name is kept ONLY for the clean protocol; anything with an
            # operator gets a qualified name like every other row.
            sweep_path = (out_dir / f"sensor_sweep_dmfgen_n{n_obs}.json"
                          if (args.cond_source == "points" and not op_tag) else
                          out_dir / f"{args.out_prefix}{op_suffix}_dmfgen_n{n_obs}.json")
            sweep_path = safe_write_json(sweep_path, payload)
            print(f"[out] wrote {sweep_path}", flush=True)
            if n_obs == (655 if args.cond_source == "points" else max(args.n_obs_list)):
                main_path = out_dir / f"{args.out_prefix}{op_suffix}_dmfgen_K{args.K}_nfe{nfe}.json"
                main_path = safe_write_json(main_path, payload)
                print(f"[out] wrote {main_path}", flush=True)
            rel_l2_by_n[str(n_obs)] = payload["summary"]["aggregate"]["rel_l2_mean"]
            cost_by_n[str(n_obs)] = {
                "inference_seconds_per_field_mean":
                    cost["inference_seconds_per_field_mean"],
                "inference_peak_gpu_gb": cost["inference_peak_gpu_gb"],
            }
        combined = {
            # was hardcoded clean, so an operator run produced a file claiming
            # the clean protocol while holding corrupted-sensor numbers -- the
            # one failure mode the protocol stamp exists to prevent
            "protocol": PROTOCOL,
            "operator": {"noise_sigma_z": args.sensor_noise,
                         "occlusion_frac": args.sensor_occlusion,
                         "channel_dropout": args.sensor_dropout,
                         "tag": op_tag or "clean"},
            "model": "dmfgen",
            "run_dir": str(run_dir),
            "ckpt": args.ckpt,
            "ckpt_epoch": ckpt_epoch,
            "clamp_hard": CLAMP_HARD,
            "split": args.split,
            "K": args.K,
            "nfe": nfe,
            "n_frames": len(scored_frames),
            "frame_indices": scored_frames,
            "seed": args.seed,
            "seeding": seeding_note,
            "metric": "rel_l2_mean (ensemble-mean relative L2, aggregate == "
                      "vorticity, mean over frames)",
            "rel_l2_by_n": rel_l2_by_n,
            "cost_by_n": cost_by_n,
        }
        comb_path = (out_dir / "sensor_sweep_dmfgen.json"
                     if (args.cond_source == "points" and not op_tag)
                     else out_dir / f"{args.out_prefix}{op_suffix}_dmfgen_sweep.json")
        comb_path = safe_write_json(comb_path, combined)
        print(f"[out] wrote {comb_path}", flush=True)
        print(f"[RESULT] dmfgen rel_l2_by_n={rel_l2_by_n}", flush=True)
    else:
        tag = {"latent_fm": "latentfm", "mlp_rbf": "mlprbf", "dmfgen_cq": "dmfgencq"}.get(
            args.model, args.model)
        if deterministic:
            main_path = out_dir / f"{args.out_prefix}{op_suffix}_{tag}_K1.json"
        else:
            main_path = out_dir / f"{args.out_prefix}{op_suffix}_{tag}_K{args.K}_nfe{nfe}.json"
        rel_l2_by_n = {}
        with weights_ctx:
            if not own_weights:
                bundle.model.eval()
            for n_obs in args.n_obs_list:
                per_snap, cost = run_protocol(n_obs)
                payload = payload_common(n_obs, per_snap, cost)
                rel_l2_by_n[str(n_obs)] = payload["summary"]["aggregate"]["rel_l2_mean"]
                if len(args.n_obs_list) > 1:
                    # surface sweep: one JSON per density (largest = main)
                    sweep_path = out_dir / f"{args.out_prefix}{op_suffix}_{tag}_n{n_obs}.json"
                    sweep_path = safe_write_json(sweep_path, payload)
                    print(f"[out] wrote {sweep_path}", flush=True)
        if len(args.n_obs_list) > 1:
            print(f"[RESULT] {args.model} rel_l2_by_n={rel_l2_by_n}", flush=True)
        main_path = safe_write_json(main_path, payload)
        s = payload["summary"]["aggregate"]
        disp = ("deterministic (dispersion null)" if deterministic else
                f"spread/err={s['spread_error_ratio']:.3f} "
                f"cov90={s['coverage_90']:.3f}")
        print(f"[RESULT] {args.model} relL2={s['rel_l2_mean']:.5f} "
              f"crps={s['crps']:.5f} {disp} "
              f"infer={cost['inference_seconds_per_field_mean']:.3f}s "
              f"peak={cost['inference_peak_gpu_gb']:.2f}GB", flush=True)
        print(f"[out] wrote {main_path} and "
              f"{len(per_snap)} crps_snapN.json files", flush=True)


if __name__ == "__main__":
    main()
    exit_if_conflicts()
