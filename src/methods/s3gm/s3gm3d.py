"""S3GM on the JHU 125^3 cross-cube protocol (upstream-faithful 3-D adaptation).

Upstream: https://github.com/lzy12301/S3GM @ 2343293bf627e7b4afc0d63b11ad37ece0919653
(Li et al., "Learning spatiotemporal dynamics with a pretrained generative
model", Nature Machine Intelligence 6:1566-1579, 2024).

WHY A NEW MODULE
----------------
The S3GM port already inside ``model_baseline.py`` is 2-D only: its adapter
calls ``validate_regular_grid_compatibility(train_set, num_x, num_y)``, which
raises on JHU because 125*125 != 125^3.  Nothing in this file edits the shared
model file; ``train_s3gm3d.py`` swaps this adapter into the registry at import
time.

THE ADAPTATION (z -> upstream's frame axis)
-------------------------------------------
Upstream S3GM is a *spatiotemporal video* score model.  ``UNetVideoModel``
consumes ``(B, T, C, H, W)``, applies 2-D convolutions in (H, W), and couples
frames only inside ``FactorizedAttentionBlock`` (temporal attention with a
relative-position network over ``frame_indices``, plus spatial attention).

We map the volume's z axis onto upstream's frame axis.  A training sample is a
*slab*: ``T_WIN`` consecutive z planes of the 125x125 (x, y) grid, exactly
analogous to upstream drawing a ``num_frames``-long window from a trajectory.
Full-volume reconstruction then uses upstream's own ``generate_parallel_2d``
mechanism: ``b`` overlapping slabs denoised jointly, tied together by the
overlap-consistency loss.  Consequences:

  * ZERO changes to upstream model code are required.
  * Upstream's window/overlap sampler -- the part a naive "T=1, whole volume"
    port throws away -- is preserved and used.
  * Per-step noise-level diversity is preserved: upstream draws one ``t`` per
    *sample*, so slabs (not whole volumes) are what makes a batch of 32
    independent ``t`` values affordable.
  * The adaptation is anisotropic: z is treated as "time".  ``JHU_AUGMENT=
    octahedral`` (48 signed axis permutations) means every axis takes the
    slab role over training, which is the mitigation; it is not a proof of
    isotropy.  Reported as a fairness caveat.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# Canonical sensor draw. helpers_baseline's copy consumes a different RNG
# stream (CUDA randint before randperm) and yields an independent layout.
from methods.dmfgen.helpers import build_sparse_condition
from methods.common.helpers_baseline import build_obs_grid_mask3d, compute_pad_size
import methods.common.adapters as badapters
import methods.common.config as bconfig
import methods.common.helpers_baseline as hb
import methods.s3gm.ema as s3gm_ema
from methods.s3gm.adapter import _net_to_score
import methods.s3gm.sde as s3gm_sde
import methods.s3gm.unet_video as s3gm_unet

UNetVideoModel = s3gm_unet.UNetVideoModel
VESDE = s3gm_sde.VESDE
ExponentialMovingAverage = s3gm_ema.ExponentialMovingAverage
BaselineBundle = badapters.BaselineBundle
BaseBaselineAdapter = badapters.BaseBaselineAdapter
pointcloud_to_grid3d = hb.pointcloud_to_grid3d
midplane_slice = hb.midplane_slice
_save_single_field_plot = hb._save_single_field_plot


# ---------------------------------------------------------------------------
# Volume <-> slab plumbing
# ---------------------------------------------------------------------------
def volume_to_video(fields_pc: torch.Tensor, Nz: int, Ny: int, Nx: int,
                    H_pad: int, W_pad: int) -> torch.Tensor:
    """[B, N, C] point cloud -> [B, T=Nz, C, H_pad, W_pad] "video"."""
    g = pointcloud_to_grid3d(fields_pc, Nz, Ny, Nx)          # [B, C, Z, Y, X]
    g = g.permute(0, 2, 1, 3, 4).contiguous()                # [B, Z, C, Y, X]
    if H_pad > Ny or W_pad > Nx:
        g = F.pad(g, (0, W_pad - Nx, 0, H_pad - Ny), value=0.0)
    return g


def video_to_pointcloud(v: torch.Tensor, Nz: int, Ny: int, Nx: int) -> torch.Tensor:
    """[B, T, C, H_pad, W_pad] -> [B, N, C], cropping the (x, y) padding."""
    v = v[:, :Nz, :, :Ny, :Nx]
    B, T, C, H, W = v.shape
    return v.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C).contiguous()


def draw_slabs(video: torch.Tensor, t_win: int, n_slabs: int,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """[B, Z, C, H, W] -> [B * n_slabs, t_win, C, H, W] at random z offsets."""
    B, Z = video.shape[0], video.shape[1]
    hi = Z - t_win + 1
    starts = torch.randint(0, hi, (B * n_slabs,), device="cpu", generator=generator)
    out = []
    for k in range(B * n_slabs):
        b = k % B
        s = int(starts[k])
        out.append(video[b, s:s + t_win])
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# Training objective -- byte-for-byte upstream trainer/loss.py::loss_fn_video
# ---------------------------------------------------------------------------
def s3gm3d_loss(net, sde, slabs: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """slabs: [B, T, C, H, W].  Upstream loss_fn_video, verbatim."""
    B, T = slabs.shape[0], slabs.shape[1]
    device = slabs.device
    t = torch.rand(B, device=device) * (sde.T - eps) + eps
    z = torch.randn_like(slabs)
    mean, std = sde.marginal_prob(slabs, t)
    std_e = std[:, None, None, None, None]
    perturbed = mean + std_e * z

    # Upstream trainer/datasets.py: latent_mask all ones, obs_mask all zeros,
    # frame_indices = arange(num_frames)  (relative, per window).
    obs_mask = torch.zeros(B, T, 1, 1, 1, device=device)
    latent_mask = torch.ones(B, T, 1, 1, 1, device=device)
    frame_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

    out, _ = net(perturbed, x0=slabs, timesteps=std,
                 frame_indices=frame_indices,
                 obs_mask=obs_mask, latent_mask=latent_mask)
    score = _net_to_score(net, out, std)
    losses = torch.square(score * std_e + z)
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    return torch.mean(losses)


def _net_fn(net, sde, x, t):
    """Upstream trainer/loss.py::predict_fn with continuous=True, plus the
    fixed masks the published sampling notebooks use."""
    B, T = x.shape[0], x.shape[1]
    labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
    obs_mask = torch.zeros(B, T, 1, 1, 1, device=x.device)
    latent_mask = torch.ones(B, T, 1, 1, 1, device=x.device)
    frame_indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    out = net(x, x0=x, timesteps=labels, frame_indices=frame_indices,
              obs_mask=obs_mask, latent_mask=latent_mask)[0]
    return _net_to_score(net, out, labels)


# ---------------------------------------------------------------------------
# Sampler -- upstream sampler/utils.py::generate_parallel_2d
# ---------------------------------------------------------------------------
@torch.no_grad()
def _langevin_corrector(net, sde, x, vec_t, snr, n_steps):
    """Upstream LangevinCorrector.  The three published sampling notebooks all
    pass ``corrector=NoneCorrector``, so this runs only when explicitly asked
    for (n_steps > 0) and is off by default."""
    for _ in range(n_steps):
        grad = _net_fn(net, sde, x, vec_t)
        noise = torch.randn_like(x)
        gn = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        nn_ = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step = (snr * nn_ / gn) ** 2 * 2
        x = x + step * grad + torch.sqrt(step * 2) * noise
    return x


def s3gm_reconstruct(
    net,
    sde,
    obs_video: torch.Tensor,       # [1, Z, C, H_pad, W_pad] observed values
    obs_mask_video: torch.Tensor,  # [1, Z, C, H_pad, W_pad] 1 where observed
    Nz: int,
    K: int = 1,
    t_win: int = 10,
    overlap: int = 1,
    alpha: float = 5.0,
    beta: float = 0.4,
    snr: float = 0.128,
    n_corrector_steps: int = 0,
    eps: float = 1e-12,
    device: Optional[torch.device] = None,
    progress: bool = True,
):
    """Upstream ``generate_parallel_2d`` with the KSE/Kolmogorov/ERA5 notebook
    settings, generalized to K posterior draws batched together.

    ``sde.N`` is the number of denoising steps (upstream sets
    ``VESDE(..., N=outer_loop)`` where ``outer_loop`` is exactly the sampling
    step count, so the discrete sigma ladder always matches the step budget).

    Returns [K, Z, C, H_pad, W_pad].
    """
    device = device or obs_video.device
    C, H, W = obs_video.shape[2], obs_video.shape[3], obs_video.shape[4]
    ol = int(overlap)
    b = int(np.ceil((Nz - ol) / (t_win - ol)))
    ns_real = b * (t_win - ol) + ol
    shape = (K, b, t_win, C, H, W)

    y = obs_video.to(device)
    m = obs_mask_video.to(device)

    def x_to_sample(xx):
        """b overlapping slabs -> one [K, ns_real, C, H, W] volume."""
        out = xx.new_zeros(K, ns_real, C, H, W)
        for i in range(b):
            i_inv = b - i - 1
            out[:, i_inv * (t_win - ol): i_inv * (t_win - ol) + t_win] = xx[:, i_inv]
        return out

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    x = sde.prior_sampling(list(shape)).to(device).float()
    x_mean = torch.zeros_like(x)

    n_clamped, dx_max = 0, 0.0
    it = tqdm(range(sde.N), desc="S3GM sampling", leave=False) if progress else range(sde.N)
    for i in it:
        t = timesteps[i]
        vec_t = torch.ones(K * b, device=device).float() * t
        xb = x.reshape(K * b, t_win, C, H, W)

        if n_corrector_steps > 0:
            xb = _langevin_corrector(net, sde, xb, vec_t, snr, n_corrector_steps)

        z = torch.randn_like(x)
        zb = z.reshape(K * b, t_win, C, H, W)

        with torch.enable_grad():
            inp = xb.detach().clone().requires_grad_(True)
            score = _net_fn(net, sde, inp, vec_t)
            with torch.no_grad():
                f, G = sde.discretize(inp.detach(), vec_t)
                rev_f = f - G[:, None, None, None, None] ** 2 * score.detach()
                temp_mean = inp.detach() - rev_f
                temp_u = temp_mean + G[:, None, None, None, None] * zb

            _, std = sde.marginal_prob(inp, vec_t)
            x0_hat = (std[:, None, None, None, None] ** 2 * score + inp)
            x0_hat = x0_hat.reshape(K, b, t_win, C, H, W)
            vol = x_to_sample(x0_hat)[:, :Nz]

            # DPS term. Upstream uses an *unnormalised sum of squares* over the
            # observed entries (sampler/utils.py:706) with no 1/||r|| factor;
            # the sensor-count dependence lives entirely in `alpha`, which the
            # notebooks set to alpha_case / sqrt(observed fraction).
            resid = (y - vol) * m
            loss_dps = torch.sum((resid ** 2).reshape(K, -1), dim=-1).sum()

            # Overlap consistency between neighbouring slabs (upstream
            # loss_consis).  Summed over the K members so each draw sees the
            # same gradient it would see if run alone.
            if b > 1:
                a_ = x0_hat[:, :-1, (t_win - ol):t_win].detach()
                c_ = x0_hat[:, 1:, :ol]
                loss_consis = torch.sum(((a_ - c_) ** 2).reshape(K, b - 1, -1), dim=-1)
                loss_consis = torch.sum(loss_consis)
            else:
                loss_consis = x0_hat.sum() * 0.0

            loss = alpha * loss_dps + beta * loss_consis
            dx = torch.autograd.grad(loss, inp)[0]
            # Upstream's only guard (sampler/utils.py:727). If it binds, the
            # guidance has run away -- tracked and reported below rather than
            # silently absorbed.
            n_clamped += int((dx.abs() > 1e8).sum())
            dx = torch.clamp(dx, min=-1e8, max=1e8)
            dx_max = max(dx_max, float(dx.abs().max()))
            xb = (temp_u - dx).detach()

        x = xb.reshape(K, b, t_win, C, H, W)
        x_mean = temp_mean.detach().reshape(K, b, t_win, C, H, W)
        if progress:
            it.set_postfix_str(f"obs={float(alpha * loss_dps):.3e} consis={float(beta * loss_consis):.3e}")

    out = x_to_sample(x_mean)[:, :Nz]
    # Divergence diagnostic. Upstream's DPS step subtracts the raw gradient with
    # no step size (sampler/utils.py:728), so the sampler is only stable once
    # the score is accurate. `obs_rmse_z` is in z-score units: O(1) is healthy,
    # >>1 means the sample ran away from its own observations.
    with torch.no_grad():
        nobs = float(m.sum()) * K
        obs_rmse = float(torch.sqrt((((out - y) * m) ** 2).sum() / max(nobs, 1.0)))
        amp = float(out.abs().max())
    print(f"[sampler] N={sde.N} K={K} alpha={alpha:.3f} obs_rmse_z={obs_rmse:.4e} "
          f"max_abs={amp:.4e} dx_max={dx_max:.3e} clamped_elems={n_clamped} "
          f"{'DIVERGED' if (obs_rmse > 10.0 or not np.isfinite(obs_rmse)) else 'ok'}",
          flush=True)
    return out


# ---------------------------------------------------------------------------
# Observation plumbing
# ---------------------------------------------------------------------------
def s3gm_reconstruct_ensemble(net, sde, obs_video, obs_mask_video, Nz, K=1,
                              chunk=1, **kw):
    """K posterior draws, in chunks of `chunk` members.

    Measured on an H100 80GB (nf=32, ch_mult (1,2,3,4), 2 res blocks, 14
    windows of 10 planes): chunk=1 peaks at 32.0 GB, chunk=2 at 63.9 GB,
    chunk=4 OOMs.  chunk=1 is therefore the default -- the inference memory
    wall sits between 2 and 4 simultaneous draws.
    """
    outs = []
    done = 0
    while done < K:
        k = min(chunk, K - done)
        outs.append(s3gm_reconstruct(net, sde, obs_video, obs_mask_video, Nz,
                                     K=k, **kw).detach())
        done += k
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return torch.cat(outs, dim=0)


def obs_to_video(obs_values, obs_mask, obs_field_ids, obs_indices,
                 n_fields, n_pts, Nz, Ny, Nx, H_pad, W_pad):
    """Sparse sensors -> dense [1, Z, C, H_pad, W_pad] value/mask videos."""
    vg, mg = build_obs_grid_mask3d(
        obs_values, obs_mask, obs_field_ids, obs_indices,
        n_fields, n_pts, Nz, Ny, Nx, Nz, H_pad, W_pad,
    )                                                    # [B, C, Z, H_pad, W_pad]
    return vg.permute(0, 2, 1, 3, 4).contiguous(), mg.permute(0, 2, 1, 3, 4).contiguous()


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------
def run_epoch_s3gm3d(bundle: BaselineBundle, loader: DataLoader,
                     training: bool, epoch: int) -> float:
    net = bundle.model
    opt = bundle.optimizer if training else None
    sde = bundle.components["sde"]
    cmp = bundle.components
    Nz, Ny, Nx = cmp["Nz"], cmp["Ny"], cmp["Nx"]
    H_pad, W_pad = cmp["H_pad"], cmp["W_pad"]
    t_win = cmp["t_win"]
    n_slabs = cmp["slabs_per_snapshot"]
    ema = bundle.ema
    params_fn = cmp["all_params_fn"]
    opt_steps = [int(cmp.get("total_opt_steps", 0))]

    net.train(training)
    total, count = 0.0, 0
    # Separate loader wait from GPU work.  The H5 is read on every __getitem__
    # with no caching, so on a contended shared filesystem the epoch wall-clock
    # can be dominated by I/O; the paper needs to state what fraction of each
    # matched budget was real compute.
    t_data, t_compute = 0.0, 0.0
    pbar = tqdm(loader, desc=f"S3GM3D {epoch:04d} [{'Train' if training else 'Eval'}]",
                leave=False)
    _t_prev = time.perf_counter()
    for batch in pbar:
        t_data += time.perf_counter() - _t_prev
        _t_c0 = time.perf_counter()
        fields = batch["fields"].to(bundle.device, non_blocking=True)
        video = volume_to_video(fields, Nz, Ny, Nx, H_pad, W_pad)
        slabs = draw_slabs(video, t_win, n_slabs)
        del video
        loss = s3gm3d_loss(net, sde, slabs)

        if training and opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Upstream train.py applies no clipping.  We keep a very loose
            # NaN/blow-up guard only: it is inactive on healthy steps.
            gn = nn.utils.clip_grad_norm_(params_fn(), max_norm=cmp["grad_clip"])
            if torch.isfinite(loss) and torch.isfinite(gn):
                opt.step()
                if ema is not None:
                    ema.update(params_fn())
            else:
                opt.zero_grad(set_to_none=True)

        cur = float(loss.detach().cpu())
        total += cur
        count += 1
        if training:
            opt_steps[0] += 1
        pbar.set_postfix_str(f"loss={cur:.6e}")
        t_compute += time.perf_counter() - _t_c0
        _t_prev = time.perf_counter()

    if training:
        bundle.components["total_opt_steps"] = opt_steps[0]
        bundle.components["last_data_s"] = t_data
        bundle.components["last_compute_s"] = t_compute
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Periodic qualitative monitor
# ---------------------------------------------------------------------------
def visualize_s3gm3d(bundle: BaselineBundle, dataset, save_dir: Path, epoch: int,
                     snapshot_index: int = 0, n_steps: Optional[int] = None,
                     K: Optional[int] = None) -> dict:
    cmp = bundle.components
    device = bundle.device
    Nz, Ny, Nx = cmp["Nz"], cmp["Ny"], cmp["Nx"]
    H_pad, W_pad = cmp["H_pad"], cmp["W_pad"]
    scfg = cmp["sampling_cfg"]
    n_steps = int(scfg["monitor_N"] if n_steps is None else n_steps)
    K = int(scfg["monitor_K"] if K is None else K)
    cond_fields = list(cmp["vis_cond_fields"])
    n_obs = list(cmp["vis_n_obs"])

    item = dataset[int(snapshot_index)]
    coords = item["coords"].unsqueeze(0).to(device)
    truth = item["fields"].unsqueeze(0).to(device)
    coords_raw = item["coords_raw"].cpu().numpy()

    torch.manual_seed(12345 + int(snapshot_index))
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords, fields_full=truth, cond_fields=cond_fields,
        n_obs_min=n_obs, n_obs_max=n_obs)
    yv, mv = obs_to_video(ov, om, ofid, oi, dataset.num_fields,
                          dataset.num_points, Nz, Ny, Nx, H_pad, W_pad)

    obs_frac = float(int(om.sum())) / float(dataset.num_points * len(cond_fields))
    # Monitor guidance uses the stabilised JHU-tuned arm, NOT the run's frozen
    # sampling_cfg. A run launched with upstream alpha_case=0.5 / beta=0.4 would
    # otherwise monitor at alpha_dps ~= 5.0 at 1% observations, which diverges
    # at JHU scale and invalidates every periodic figure (fleet audit
    # 2026-08-29). Reported numbers select their arm explicitly in
    # eval_s3gm3d.py; this override affects training-time figures only.
    _mon_alpha_case = float(scfg.get("monitor_alpha_case", 0.05))
    _mon_beta = float(scfg.get("monitor_beta", 0.004))
    alpha = _mon_alpha_case / math.sqrt(max(obs_frac, 1e-12))
    print(f"[s3gm-monitor] alpha_case={_mon_alpha_case} -> alpha_dps={alpha:.4f} "
          f"beta={_mon_beta} (stabilised monitor arm; run sampling_cfg has "
          f"alpha_case={scfg['alpha_case']} beta={scfg['beta']})", flush=True)

    sde_s = VESDE(config=None, sigma_min=cmp["sigma_min"],
                  sigma_max=cmp["sigma_max"], N=n_steps)
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with torch.enable_grad():
        vol = s3gm_reconstruct_ensemble(
            bundle.model, sde_s, yv, mv, Nz, K=K, chunk=1, t_win=cmp["t_win"],
            overlap=cmp["overlap"], alpha=alpha, beta=_mon_beta,
            snr=float(scfg["snr"]), n_corrector_steps=int(scfg["n_corrector_steps"]),
            device=device, progress=True)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0.0

    ens = video_to_pointcloud(vol, Nz, Ny, Nx)                     # [K, N, C]
    mean_pc = ens.mean(dim=0, keepdim=True)
    std_pc = ens.std(dim=0, unbiased=False)[None] if K > 1 else torch.zeros_like(mean_pc)

    mu = dataset.mean.to(device)
    sd = dataset.std.to(device)
    pred = (mean_pc * sd.view(1, 1, -1) + mu.view(1, 1, -1))[0].cpu().numpy()
    true = (truth * sd.view(1, 1, -1) + mu.view(1, 1, -1))[0].cpu().numpy()
    spread = (std_pc * sd.view(1, 1, -1))[0].cpu().numpy()

    is_3d, slice_mask, coords_xy_plot, triang = midplane_slice(coords_raw)
    coords_xy = coords_raw[:, :2]
    valid = om[0].bool()
    oi_np = oi[0, valid].cpu().numpy()
    ofid_np = ofid[0, valid].cpu().numpy()

    os.makedirs(str(save_dir), exist_ok=True)
    metrics = {}
    names = [str(n) for n in dataset.field_names][:true.shape[1]]
    if len(names) < true.shape[1]:
        names = names + [f"field_{i}" for i in range(len(names), true.shape[1])]
    for c, name in enumerate(names):
        sc = None
        fsel = ofid_np == c
        if np.any(fsel):
            idx = oi_np[fsel]
            on = slice_mask[idx] if is_3d else np.ones(len(idx), bool)
            sc = coords_xy[idx[on]] if np.any(on) else None
        metrics[str(name)] = _save_single_field_plot(
            true_f=true[:, c][slice_mask], pred_f=pred[:, c][slice_mask],
            metric_true_f=true[:, c], metric_pred_f=pred[:, c],
            coords_xy=coords_xy_plot, sensor_coords=sc, field_name=str(name),
            epoch=epoch, save_dir=str(save_dir),
            file_prefix=f"s3gm3d_N{n_steps}", triang=triang)

    # Ensemble-spread panel: a collapsed posterior (spread ~ 0 everywhere) is
    # the failure this figure is here to expose.
    if K > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
        axs = np.atleast_1d(axs)
        for c, name in enumerate(names):
            im = axs[c].tricontourf(triang, spread[:, c][slice_mask], levels=40, cmap="viridis")
            fig.colorbar(im, ax=axs[c], fraction=0.046, pad=0.04)
            axs[c].set_title(f"{name} ensemble std (K={K})")
            axs[c].set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(str(save_dir), f"s3gm3d_N{n_steps}_epoch_{epoch:04d}_ensemble_std.png"), dpi=200)
        plt.close(fig)
        for c, name in enumerate(names):
            metrics[f"{name}_spread"] = float(np.mean(spread[:, c]))

    payload = {
        "epoch": int(epoch), "snapshot_index": int(snapshot_index),
        "method": "S3GM3D_VESDE", "n_denoising_steps": int(n_steps),
        "K": int(K), "alpha_dps": alpha, "beta": _mon_beta,
        "guidance_arm": "monitor_jhu_tuned", "observed_fraction": obs_frac,
        "cond_fields": [int(v) for v in cond_fields],
        "n_obs": [int(v) for v in n_obs],
        "infer_wall_s": wall, "infer_peak_gpu_mb": peak,
        "metrics": metrics,
    }
    with open(os.path.join(str(save_dir), f"s3gm3d_N{n_steps}_metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[instr] infer_one_field_s={wall:.2f} infer_peak_gpu_mb={peak:.1f} "
          f"n_denoising_steps={n_steps} K={K}", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class S3GM3DAdapter(BaseBaselineAdapter):
    name = "s3gm"

    def build_for_training(self, cfg, device, run_dir, train_set, val_set) -> BaselineBundle:
        stage = bconfig.resolve_stage_config(cfg)
        arch, diff, tr = stage["architecture"], stage["diffusion"], stage["training"]
        scfg = stage.get("sampling", {})
        d = cfg["shared"]["data"]
        Nx, Ny, Nz = int(d["num_x"]), int(d["num_y"]), int(d["num_z"])
        if Nz * Ny * Nx != int(train_set.num_points):
            raise ValueError(
                f"S3GM3D expects a full {Nz}x{Ny}x{Nx} grid but the dataset has "
                f"{train_set.num_points} points.")
        factor = 2 ** (len(arch["ch_mult"]) - 1)
        H_pad, W_pad = compute_pad_size(Ny, Nx, factor)

        net = UNetVideoModel(
            in_channels=train_set.num_fields,
            model_channels=int(arch["nf"]),
            out_channels=train_set.num_fields,
            num_res_blocks=int(arch["num_res_blocks"]),
            attention_resolutions=tuple(arch["attn_resolutions"]),
            image_size=max(H_pad, W_pad),
            dropout=float(arch["dropout"]),
            channel_mult=tuple(arch["ch_mult"]),
            conv_resample=True,
            dims=2,
            num_heads=int(arch["num_heads"]),
            use_rpe_net=True,
            use_checkpoint=bool(arch.get("use_checkpoint", False)),
        ).to(device)

        def all_params_fn():
            return list(net.parameters())

        # Upstream train.py: Adam(lr, betas=(0.9,0.999), eps=1e-8, weight_decay=0),
        # no LR schedule.  weight_decay stays configurable but defaults to 0.
        opt = torch.optim.Adam(all_params_fn(), lr=float(tr["learning_rate"]),
                               betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=float(tr.get("weight_decay") or 0.0))
        net.frb_score_param = str(diff.get("score_param", "raw"))
        if net.frb_score_param not in ("raw", "eps"):
            raise ValueError(f"s3gm diffusion.score_param must be 'raw' or 'eps', got {net.frb_score_param!r}")
        print(f"[s3gm3d] score_param={net.frb_score_param}", flush=True)
        ema = ExponentialMovingAverage(all_params_fn(), decay=float(diff["ema_rate"]))
        sde = VESDE(config=None, sigma_min=float(diff["sigma_min"]),
                    sigma_max=float(diff["sigma_max"]), N=int(diff["num_scales"]))

        sampling = {
            "alpha_case": float(scfg.get("alpha_case", 0.5)),
            "beta": float(scfg.get("beta", 0.4)),
            "snr": float(scfg.get("snr", 0.128)),
            "n_corrector_steps": int(scfg.get("n_corrector_steps", 0)),
            "monitor_N": int(scfg.get("monitor_N", 100)),
            "monitor_K": int(scfg.get("monitor_K", 2)),
            "sampling_N": int(scfg.get("sampling_N", 200)),
        }
        vis_cond = cfg["shared"]["conditioning"].get("vis_cond_fields") \
            or cfg["shared"]["conditioning"]["cond_fields"]
        vis_nobs = cfg["shared"]["conditioning"].get("vis_n_obs_list") \
            or cfg["shared"]["conditioning"]["n_obs_max_list"]
        if len(vis_nobs) != len(vis_cond):
            vis_nobs = [vis_nobs[0]] * len(vis_cond)

        n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
        bch = int(arch["nf"]) * int(arch["ch_mult"][-1])
        bres = max(H_pad, W_pad) // factor
        t_win = int(arch.get("t_win", 10))
        print(f"[capacity] trainable_params={n_par:,} "
              f"bottleneck={bch}ch x {bres}x{bres} x T{t_win} = "
              f"{bch * bres * bres * t_win:,} scalars", flush=True)

        return BaselineBundle(
            baseline_model="s3gm", training_stage=1, model=net, optimizer=opt,
            scheduler=None, ema=ema, device=device, run_dir=run_dir, config=cfg,
            dataset_train=train_set, dataset_val=val_set,
            components={
                "sde": sde, "Nz": Nz, "Ny": Ny, "Nx": Nx,
                "H_pad": H_pad, "W_pad": W_pad,
                "t_win": t_win,
                "overlap": int(arch.get("overlap", 1)),
                "slabs_per_snapshot": int(tr.get("slabs_per_snapshot", 16)),
                "grad_clip": float(tr.get("grad_clip", 1.0e9)),
                "all_params_fn": all_params_fn,
                "sampling_cfg": sampling,
                "sigma_min": float(diff["sigma_min"]),
                "sigma_max": float(diff["sigma_max"]),
                "vis_cond_fields": vis_cond,
                "vis_n_obs": vis_nobs,
                "trainable_params": n_par,
            },
        )

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        bundle.model.load_state_dict(badapters._checkpoint_model_state(checkpoint, "S3GM3D"))
        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.ema is not None and checkpoint.get("ema") is not None:
            bundle.ema.load_state_dict(checkpoint["ema"])
            bundle.ema.shadow_params = [p.to(bundle.device) for p in bundle.ema.shadow_params]

    def run_epoch(self, bundle, loader, training, epoch):
        if training and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(bundle.device)
        t0 = time.perf_counter()
        loss = run_epoch_s3gm3d(bundle, loader, training, epoch)
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(bundle.device) / 1024 ** 2 if torch.cuda.is_available() else 0.0
        if training:
            nb = max(len(loader), 1)
            c = bundle.components
            tc = float(c.get("last_compute_s", dt))
            td = float(c.get("last_data_s", 0.0))
            duty = tc / max(tc + td, 1e-9)
            print(f"[instr] epoch={epoch} train_epoch_s={dt:.3f} "
                  f"train_step_s={dt / nb:.4f} train_compute_step_s={tc / nb:.4f} "
                  f"train_peak_gpu_mb={peak:.1f} steps={nb} "
                  f"opt_steps_total={c.get('total_opt_steps', 0)} "
                  f"data_s={td:.3f} compute_s={tc:.3f} epoch_duty={duty:.3f}",
                  flush=True)
            c["last_train_step_s"] = dt / nb
            c["last_train_peak_mb"] = peak
        return loss, dt, peak

    def build_checkpoint(self, bundle, epoch, train_loss, val_loss) -> dict:
        c = bundle.components
        return {
            "baseline_model": "s3gm", "training_stage": 1,
            "model": bundle.model.state_dict(),
            "ema": bundle.ema.state_dict() if bundle.ema is not None else None,
            "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
            "scheduler": None,
            "epoch": int(epoch), "train_loss": float(train_loss),
            "val_loss": float(val_loss), "method": "S3GM3D_VESDE",
            "Num_x": c["Nx"], "Num_y": c["Ny"], "Num_z": c["Nz"],
            "trainable_params": c["trainable_params"],
            "train_step_s": c.get("last_train_step_s"),
            "train_peak_gpu_mb": c.get("last_train_peak_mb"),
            "total_opt_steps": c.get("total_opt_steps"),
            "last_data_s": c.get("last_data_s"),
            "last_compute_s": c.get("last_compute_s"),
        }

    @contextlib.contextmanager
    def evaluation_weights(self, bundle) -> Iterator[None]:
        if bundle.ema is None:
            yield
            return
        p = bundle.components["all_params_fn"]()
        bundle.ema.store(p)
        bundle.ema.copy_to(p)
        try:
            yield
        finally:
            bundle.ema.restore(p)

    def visualize(self, bundle, dataset, save_dir, epoch, snapshot_index,
                  n_steps=None, save_obs_consistency_plots=False):
        return visualize_s3gm3d(bundle, dataset, Path(save_dir), epoch,
                                snapshot_index=snapshot_index, n_steps=n_steps)
