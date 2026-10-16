"""Standalone trainer for the DeepONet++ (structured branch) arm.

PROVENANCE: this file is a COPY of src/train_deeponet.py's control flow (the
write-guard forbids editing the shared tree, and the fixed-mask validation +
early stopping below change main()'s control flow, so importing the vanilla
trainer and monkey-patching was not viable).  Deltas vs train_deeponet.py:

  1. builds `deeponetpp.build_deeponetpp` instead of the vanilla model
     (same encode/trunk_forward/combine interface);
  2. VALIDATION FOR SELECTION uses the FIXED canonical masks (seed-0 draw,
     torch.manual_seed(0*777+snap), 19,531 sensors/channel on cond_fields
     [0,2]) on the TUNE split -- the odd-indexed cube-3 snapshots
     {1,3,...,49} -- per PLAN_IMPROVE_2026-08-30 fleet hygiene.  best.pt is
     selected on this metric (mean standardised per-channel rel_l2, full
     field).  The random-draw val MSE of the vanilla trainer is still logged
     for the loss curve, but selects nothing.
  3. EARLY STOPPING at the fixed-validation optimum: patience_evals fixed-val
     evaluations without improvement stop the run; BASELINE_MAX_HOURS stays
     as the wall backstop.
  4. run-dir prefix `deeponetpp_`, save_root Save_TrainedModel/JHU/
     baseline_deeponetpp (distinct from the vanilla row).

Everything else -- budget stop, tail checkpoint archive, cost instrumentation,
TrainingHistoryLogger, canonical helpers.build_sparse_condition sensor draw,
Adam lr 1e-3 constant, batch 20 -- is unchanged line for line.
"""
from __future__ import annotations
from paths import SRC_DIR

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CANONICAL sensor draw (helpers.py, NOT helpers_baseline.py -- different RNG
# stream; see train_deeponet.py header for the measured 2% overlap).
from methods.dmfgen.helpers import build_sparse_condition
from methods.deeponet.modelpp import build_deeponetpp
from methods.common.loss_plot import LossTracker
import methods.common.adapters as badapters
import methods.common.config as bconfig
import methods.common.data as bdata
import methods.common.helpers_baseline as hb

FIELD_NAMES = ["Ux", "Uy", "Uz", "p"]


def parse_args():
    ap = argparse.ArgumentParser("DeepONet++ (structured branch) trainer.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Smoke-test override of the configured epoch count.")
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix appended to the run directory name.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# fixed-mask TUNE validation (DELTA 2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def fixed_mask_validation(model, dataset, device, tune_ids, cond_fields,
                          n_obs, eval_seed=0, chunk=262_144):
    """Mean (over TUNE snapshots) of the per-snapshot aggregate standardised
    rel_l2, under the CANONICAL fixed masks: torch.manual_seed(seed*777+snap)
    then helpers.build_sparse_condition at n_obs sensors per conditioned
    channel -- byte-identical to eval_deeponet_iclr.draw_sensors on the same
    GPU SKU.  Global RNG state is saved and restored."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    try:
        model.eval()
        aggs = []
        for snap in tune_ids:
            item = dataset[int(snap)]
            coords = item["coords"].unsqueeze(0).to(device)
            truth = item["fields"].unsqueeze(0).to(device)
            torch.manual_seed(eval_seed * 777 + int(snap))
            n = [n_obs] * len(cond_fields)
            oc, ov, om, _, ofid = build_sparse_condition(
                coords_full=coords, fields_full=truth,
                cond_fields=cond_fields, n_obs_min=n, n_obs_max=n)
            coeff = model.encode(oc, ov, om, ofid)
            n_pts = coords.shape[1]
            pred = torch.empty_like(truth)
            for s in range(0, n_pts, chunk):
                e = min(s + chunk, n_pts)
                pred[:, s:e] = model.combine(coeff, model.trunk_forward(coords[:, s:e]))
            pn = pred[0].float(); tn = truth[0].float()
            rl2 = [(torch.linalg.norm(pn[:, c] - tn[:, c])
                    / torch.linalg.norm(tn[:, c]).clamp_min(1e-12)).item()
                   for c in range(pn.shape[1])]
            aggs.append(float(np.mean(rl2)))
        return float(np.mean(aggs))
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.train(was_training)


# ---------------------------------------------------------------------------
# figures (copied from train_deeponet.py; only the labels changed)
# ---------------------------------------------------------------------------
def save_reconstruction_figure(model, dataset, device, save_dir: Path, epoch: int,
                               cond_fields, n_obs_list, snapshot_index: int,
                               fig_seed: int = 12345, chunk: int = 262_144):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval()
        item = dataset[snapshot_index]
        coords = item["coords"].unsqueeze(0).to(device)
        truth = item["fields"].unsqueeze(0).to(device)
        torch.manual_seed(fig_seed)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=truth,
            cond_fields=cond_fields, n_obs_min=n_obs_list, n_obs_max=n_obs_list,
        )
        with torch.no_grad():
            coeff = model.encode(oc, ov, om, ofid)
            n_pts = coords.shape[1]
            pred = torch.empty_like(truth)
            for s in range(0, n_pts, chunk):
                e = min(s + chunk, n_pts)
                pred[:, s:e] = model.combine(coeff, model.trunk_forward(coords[:, s:e]))

        mean = dataset.mean.to(device).view(1, 1, -1)
        std = dataset.std.to(device).view(1, 1, -1)
        pr = (pred * std + mean)[0].float().cpu().numpy()
        gt = (truth * std + mean)[0].float().cpu().numpy()

        side = int(round(dataset.num_points ** (1 / 3)))
        zmid = side // 2
        # h5 coordinates are C-ordered (x, y, z); verified against the file.
        pr3 = pr.reshape(side, side, side, -1)[:, :, zmid, :]
        gt3 = gt.reshape(side, side, side, -1)[:, :, zmid, :]

        valid = om[0].bool()
        idx = oi[0, valid].cpu().numpy()
        fid = ofid[0, valid].cpu().numpy()
        ijk = np.stack(np.unravel_index(idx, (side, side, side)), axis=1)
        on_slice = ijk[:, 2] == zmid

        n_ch = pr3.shape[-1]
        fig, axes = plt.subplots(n_ch, 3, figsize=(12, 3.4 * n_ch))
        for c in range(n_ch):
            err = np.abs(pr3[..., c] - gt3[..., c])
            vmin, vmax = float(gt3[..., c].min()), float(gt3[..., c].max())
            sel = on_slice & (fid == c)
            for j, (img, title, kw) in enumerate([
                (gt3[..., c], f"{FIELD_NAMES[c]} truth", dict(vmin=vmin, vmax=vmax, cmap="RdBu_r")),
                (pr3[..., c], f"{FIELD_NAMES[c]} DeepONet++", dict(vmin=vmin, vmax=vmax, cmap="RdBu_r")),
                (err, f"{FIELD_NAMES[c]} |error|", dict(cmap="magma")),
            ]):
                ax = axes[c, j]
                im = ax.imshow(img.T, origin="lower", **kw)
                ax.set_title(title, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046)
                if j == 0 and sel.any():
                    ax.scatter(ijk[sel, 0], ijk[sel, 1], s=1.5, c="k", alpha=0.6,
                               linewidths=0)
                    ax.set_xlabel(f"{int(sel.sum())} sensors on plane", fontsize=7)
        rel = float(np.linalg.norm(pr - gt) / (np.linalg.norm(gt) + 1e-12))
        fig.suptitle(f"DeepONet++  epoch {epoch}  snapshot {snapshot_index}  "
                     f"rel_l2(phys)={rel:.4f}", fontsize=11)
        fig.tight_layout()
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"recon_epoch{epoch:07d}.png", dpi=110)
        plt.close(fig)

        pn = pred[0].float().cpu().numpy(); tn = truth[0].float().cpu().numpy()
        metrics = {FIELD_NAMES[c]: float(np.linalg.norm(pn[:, c] - tn[:, c])
                                         / (np.linalg.norm(tn[:, c]) + 1e-12))
                   for c in range(n_ch)}
        metrics["aggregate"] = float(np.mean(list(metrics.values())))
        return metrics
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.train()


# ---------------------------------------------------------------------------
# epoch loop (copied verbatim from train_deeponet.py)
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, cond, n_query, training,
              grad_norm_log_every=0, gnorms=None):
    """Returns (mean_loss, compute_seconds, loader_wait_seconds, n_steps)."""
    model.train(training)
    total, count = 0.0, 0
    compute_s, loader_s = 0.0, 0.0
    n_steps = 0
    it = iter(loader)
    while True:
        t_l = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            loader_s += time.perf_counter() - t_l
            break
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loader_s += time.perf_counter() - t_l

        t_c = time.perf_counter()
        coords = batch["coords"].to(device, non_blocking=True)
        fields = batch["fields"].to(device, non_blocking=True)
        bsz, n_pts, _ = coords.shape

        oc, ov, om, _, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=cond["cond_fields"],
            n_obs_min=cond["n_obs_min_list"], n_obs_max=cond["n_obs_max_list"],
        )
        if 0 < n_query < n_pts:
            qi = torch.randperm(n_pts, device=device)[:n_query]
            qc, qf = coords[:, qi], fields[:, qi]
        else:
            qc, qf = coords, fields

        with torch.set_grad_enabled(training):
            pred = model(qc, oc, ov, om, ofid)
            loss = F.mse_loss(pred, qf)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Upstream deepxde applies NO gradient clipping; measurement only.
            if grad_norm_log_every and n_steps % grad_norm_log_every == 0:
                gnorms.append(float(nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf"))))
            optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compute_s += time.perf_counter() - t_c

        total += float(loss.detach()) * bsz
        count += bsz
        n_steps += 1
    return total / max(count, 1), compute_s, loader_s, n_steps


def main():
    args = parse_args()
    cfg_path = bconfig.ensure_absolute(args.config)
    cfg = bconfig.load_yaml(cfg_path)
    shared = cfg["shared"]
    tcfg = cfg["deeponet_params"]["training"]
    arch = cfg["deeponet_params"]["architecture"]

    bconfig.set_seed(int(shared["seed"]))
    device = bconfig.infer_device(args.device, shared["device_ids"])
    save_root = bconfig.ensure_absolute(shared["paths"]["save_root"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_root / (f"deeponetpp_iclr_jhu_xcube_DemoN{shared['demo_num']}_"
                           f"{timestamp}{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "Evaluation").mkdir(parents=True, exist_ok=True)
    fig_dir = run_dir / "Evaluation" / "figures"
    bconfig.save_yaml(run_dir / "run_config.yaml", cfg)

    stats_path = run_dir / "dataset_stats.pt"
    ds_kw = dict(train_ratio=float(shared["data"]["train_ratio"]),
                 seed=int(shared["seed"]),
                 time_stride=int(shared["data"]["time_stride"]),
                 field_names=tuple(shared["data"]["field_names"]),
                 stats_path=str(stats_path))
    data_path = str(bconfig.ensure_absolute(shared["paths"]["data_path"]))
    train_set = hb.TurbulentCombustionH5Dataset(data_path, split="train", **ds_kw)
    val_set = hb.TurbulentCombustionH5Dataset(data_path, split="val", **ds_kw)

    bs = int(tcfg["batch_size"])
    nw = int(shared["data"]["num_workers"])
    train_loader = bdata.build_dataloader(train_set, batch_size=bs, num_workers=nw, shuffle=True)
    val_loader = bdata.build_dataloader(val_set, batch_size=bs, num_workers=nw, shuffle=False)

    model = build_deeponetpp(arch).to(device)
    n_params = model.n_params()
    bottleneck = model.bottleneck_scalars()
    field_values = train_set.num_points * train_set.num_fields
    # Upstream deeponet_pde.py:259/:168 -- plain Adam, constant lr, no wd.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]))

    # DELTA 2/3: fixed-mask TUNE validation config
    fv = tcfg.get("fixed_val", {}) or {}
    fv_seed = int(fv.get("seed", 0))
    fv_n_obs = int(fv.get("n_obs", 19531))
    fv_cond = [int(c) for c in fv.get("cond_fields", [0, 2])]
    # TUNE split: odd-indexed cube-3 (val) snapshots {1,3,...,49}
    tune_ids = [i for i in range(len(val_set)) if i % 2 == 1]
    patience_evals = int(tcfg.get("patience_evals", 60))

    print(f"[run] dir={run_dir}", flush=True)
    print(f"[run] device={device} train={len(train_set)} val={len(val_set)}", flush=True)
    print(f"[params] trainable={n_params} target=6506253 "
          f"delta={100*(n_params-6506253)/6506253:+.3f}%", flush=True)
    print(f"[bottleneck] p={model.p} n_fields*p={bottleneck} scalars "
          f"field_values={field_values} compression={field_values/bottleneck:.1f}x",
          flush=True)
    print(f"[branch] pooled_scalars={model.pooled_scalars()} (vanilla: 736) "
          f"bins={list(model.bin_resolutions)} feat_dims={list(model.bin_feat_dims)} "
          f"fourier_modes={model.fourier_kmax**3} merge_width={model.merge_width}",
          flush=True)
    print(f"[tune] fixed-val: seed={fv_seed} n_obs={fv_n_obs}/channel "
          f"cond_fields={fv_cond} tune_ids(odd cube-3)={tune_ids} "
          f"patience_evals={patience_evals}", flush=True)

    cond = shared["conditioning"]
    n_query = int(tcfg["n_query_points"])
    eval_every = int(tcfg["eval_every"])
    total_epochs = int(args.max_epochs or tcfg["epochs"])
    gnorm_every = int(tcfg.get("grad_norm_log_every", 0) or 0)

    logger = badapters.TrainingHistoryLogger(run_dir)
    tracker = LossTracker(run_dir, name="deeponetpp")

    budget_h = float(os.environ.get("BASELINE_MAX_HOURS", "0") or 0.0)
    archive_n = int(os.environ.get("BASELINE_ARCHIVE_N", "10") or 0)
    archive_frac = float(os.environ.get("BASELINE_ARCHIVE_TAIL_FRAC", "0.05"))
    archive_every_s = ((budget_h * 3600.0 * archive_frac / archive_n)
                       if (budget_h > 0 and archive_n > 0) else 0.0)
    archive_start_s = budget_h * 3600.0 * (1.0 - archive_frac)
    archive_last, archive_i = 0.0, 0
    n_figs = int(shared["logging"].get("n_figures", 30))
    fig_every_s = (budget_h * 3600.0 / n_figs) if (budget_h > 0 and n_figs > 0) else 0.0
    fig_last = -1e9

    steps_per_epoch = max(1, len(train_loader))
    run_t0 = time.perf_counter()
    total_steps = 0
    cumul_compute = 0.0
    cumul_loader = 0.0
    best_fixed = float("inf")
    best_epoch = -1
    evals_since_best = 0
    gnorms: list[float] = []
    cost = {
        "trainable_params": int(n_params),
        "bottleneck_scalars": int(bottleneck),
        "pooled_scalars": int(model.pooled_scalars()),
        "steps_per_epoch": steps_per_epoch,
        "batch_size": bs,
        "budget_hours": budget_h,
        "budget_matched_epoch": None,
        "early_stopped_epoch": None,
        "selection": ("fixed-mask TUNE validation: canonical seed-%d masks, "
                      "%d sensors/channel, odd cube-3 snapshots" % (fv_seed, fv_n_obs)),
    }

    def write_cost(epoch, tr_time, tr_mem, step_time):
        w = time.perf_counter() - run_t0
        cost.update({
            "epochs_completed": int(epoch),
            "total_optimizer_steps": int(total_steps),
            "train_step_time_s_excl_val": round(step_time, 4),
            "train_epoch_time_s_excl_val": round(tr_time, 3),
            "train_peak_gpu_mem_mb": round(float(tr_mem), 1),
            "cumul_train_compute_s": round(cumul_compute, 3),
            "cumul_train_compute_hours": round(cumul_compute / 3600.0, 4),
            "cumul_loader_wait_s": round(cumul_loader, 3),
            "wall_elapsed_s": round(w, 3),
            "wall_elapsed_hours": round(w / 3600.0, 4),
            "duty_cycle_compute_over_wall": round(cumul_compute / w, 4) if w > 0 else None,
            "loader_fraction_of_wall": round(cumul_loader / w, 4) if w > 0 else None,
        })
        with open(run_dir / "cost_train.json", "w", encoding="utf-8") as h:
            json.dump(cost, h, indent=2)

    def snapshot(epoch, train_loss, val_loss, fixed_val=None):
        return {"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": int(epoch), "train_loss": float(train_loss),
                "val_loss": float(val_loss) if val_loss is not None else float("nan"),
                "fixed_val_rel_l2": (float(fixed_val) if fixed_val is not None
                                     else float("nan")),
                "config": cfg, "run_dir": str(run_dir), "run_name": run_dir.name,
                "trainable_params": int(n_params),
                "bottleneck_scalars": int(bottleneck)}

    stop_reason = None
    for epoch in range(1, total_epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        train_loss, c_s, l_s, n_st = run_epoch(
            model, train_loader, optimizer, device, cond, n_query, True,
            gnorm_every, gnorms)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            tr_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            tr_mem = 0.0
        tr_time = time.perf_counter() - t0
        cumul_compute += c_s
        cumul_loader += l_s
        total_steps += n_st
        step_time = tr_time / max(n_st, 1)
        print(f"[cost] epoch={epoch:05d} train_step_time_s_excl_val={step_time:.4f}"
              f" steps_per_epoch={n_st}"
              f" train_epoch_time_s_excl_val={tr_time:.3f}"
              f" train_peak_gpu_mem_mb={tr_mem:.1f}"
              f" compute_s={c_s:.3f} loader_wait_s={l_s:.3f}", flush=True)
        write_cost(epoch, tr_time, tr_mem, step_time)
        print(f"[cost] epoch={epoch:05d} total_optimizer_steps={total_steps}"
              f" cumul_compute_s={cumul_compute:.1f}"
              f" cumul_compute_hours={cumul_compute/3600.0:.4f}"
              f" cumul_loader_wait_s={cumul_loader:.1f}"
              f" duty_cycle={cost['duty_cycle_compute_over_wall']}"
              f" wall_elapsed_s={cost['wall_elapsed_s']:.1f}", flush=True)

        val_loss, fixed_val = None, None
        if epoch % eval_every == 0 or epoch == 1:
            # random-draw val MSE (curve only, selects nothing)
            val_loss, _, _, _ = run_epoch(model, val_loader, None, device, cond,
                                          n_query, False)
            # DELTA 2: fixed-mask TUNE validation -- the selection metric
            t_fv = time.perf_counter()
            fixed_val = fixed_mask_validation(model, val_set, device, tune_ids,
                                              fv_cond, fv_n_obs, fv_seed)
            print(f"[fixedval] epoch={epoch:05d} rel_l2={fixed_val:.6f} "
                  f"best={min(best_fixed, fixed_val):.6f} "
                  f"time={time.perf_counter()-t_fv:.1f}s", flush=True)
            ck = snapshot(epoch, train_loss, val_loss, fixed_val)
            torch.save(ck, run_dir / "last.pt")
            if float(fixed_val) < best_fixed:
                best_fixed = float(fixed_val)
                best_epoch = epoch
                evals_since_best = 0
                torch.save(ck, run_dir / "best.pt")
            else:
                evals_since_best += 1

        if gnorm_every and gnorms and epoch % 50 == 0:
            a = np.asarray(gnorms)
            print(f"[gradnorm] epoch={epoch:05d} n={a.size} grad_clip=None "
                  f"min={a.min():.4f} median={np.median(a):.4f} max={a.max():.4f}",
                  flush=True)
            (run_dir / "grad_norm_history.json").write_text(json.dumps(
                {"epoch": int(epoch), "n": int(a.size), "grad_clip": None,
                 "min": float(a.min()), "median": float(np.median(a)),
                 "p95": float(np.percentile(a, 95)), "max": float(a.max())}, indent=2))
            gnorms.clear()

        tracker.log(step=epoch, train_loss=train_loss, val_loss=val_loss)
        tracker.plot()
        logger.append(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                      epoch_time_s=tr_time, peak_gpu_mem_mb=tr_mem)
        msg = (f"[train] epoch={epoch:05d} loss={train_loss:.6e} "
               f"time={tr_time:.1f}s peak_mem={tr_mem:.0f}MB")
        if val_loss is not None:
            msg += f" val={val_loss:.6e} fixed_val_rel_l2={fixed_val:.6f}"
        print(msg, flush=True)

        elapsed = time.perf_counter() - run_t0
        if fig_every_s > 0 and (elapsed - fig_last) >= fig_every_s:
            m = save_reconstruction_figure(
                model, val_set, device, fig_dir, epoch,
                cond["vis_cond_fields"], cond["vis_n_obs_list"], snapshot_index=0)
            print(f"[fig] epoch={epoch:05d} " +
                  " ".join(f"{k}:{v:.4f}" for k, v in m.items()), flush=True)
            fig_last = elapsed

        if (archive_every_s > 0 and archive_i < archive_n
                and elapsed >= archive_start_s
                and (elapsed - archive_last) >= archive_every_s):
            adir = run_dir / "archive"; adir.mkdir(parents=True, exist_ok=True)
            torch.save(snapshot(epoch, train_loss, val_loss, fixed_val),
                       adir / f"ckpt_{archive_i:02d}_epoch{epoch:06d}.pt")
            print(f"[archive] {archive_i+1}/{archive_n} epoch={epoch}", flush=True)
            archive_i += 1
            archive_last = elapsed

        # DELTA 3: early stop at the fixed-validation optimum
        if patience_evals > 0 and evals_since_best >= patience_evals:
            torch.save(snapshot(epoch, train_loss, val_loss, fixed_val),
                       run_dir / "budget.pt")
            cost["early_stopped_epoch"] = int(epoch)
            with open(run_dir / "cost_train.json", "w", encoding="utf-8") as h:
                json.dump(cost, h, indent=2)
            stop_reason = (f"early-stop: {evals_since_best} fixed-val evals "
                           f"without improvement (best={best_fixed:.6f} "
                           f"at epoch {best_epoch})")
            print(f"[earlystop] {stop_reason}; wrote budget.pt.", flush=True)
            break

        if budget_h > 0 and elapsed >= budget_h * 3600.0:
            torch.save(snapshot(epoch, train_loss, val_loss, fixed_val),
                       run_dir / "budget.pt")
            cost["budget_matched_epoch"] = int(epoch)
            with open(run_dir / "cost_train.json", "w", encoding="utf-8") as h:
                json.dump(cost, h, indent=2)
            stop_reason = f"wall backstop {budget_h}h"
            print(f"[budget] {budget_h}h reached at epoch {epoch}; wrote budget.pt.",
                  flush=True)
            break

    print(f"[done] best_fixed_val_rel_l2={best_fixed:.6f} best_epoch={best_epoch} "
          f"stop={stop_reason or 'epoch limit'} run_dir={run_dir}", flush=True)
    print(f"[cost] FINAL total_optimizer_steps={cost['total_optimizer_steps']}"
          f" epochs={cost.get('epochs_completed')}"
          f" cumul_train_compute_s={cost['cumul_train_compute_s']}"
          f" cumul_train_compute_hours={cost['cumul_train_compute_hours']}"
          f" cumul_loader_wait_s={cost['cumul_loader_wait_s']}"
          f" wall_elapsed_s={cost['wall_elapsed_s']}"
          f" wall_elapsed_hours={cost['wall_elapsed_hours']}"
          f" duty_cycle={cost['duty_cycle_compute_over_wall']}", flush=True)


if __name__ == "__main__":
    main()
