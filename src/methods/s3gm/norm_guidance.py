"""Normalized, step-sized guidance for the S3GM 3-D sampler (standalone module).

WHY THIS EXISTS
---------------
Upstream S3GM's guided sampler (sampler/utils.py::generate_parallel_2d, ported
verbatim in src/baselines/s3gm/s3gm3d.py::s3gm_reconstruct) applies the guidance update as

    x <- temp_u - grad( alpha * sum(r^2) + beta * loss_consis )

i.e. the RAW gradient of an UNNORMALIZED sum of squares, with no step size.
The gradient magnitude therefore scales with the number of observed entries
(~39k at the JHU protocol) and with the residual scale, and upstream's
notebook coefficients (alpha=5, beta=0.4 at 1% density) explode predictions to
1e7-1e8 at JHU scale (fleet audit 2026-08-29; isolation_final.json).

THE FIX (guidance_mode="norm")
------------------------------
Each conditioning term is applied as a unit-normalized direction with an
explicit step size, per posterior member k:

    dx_k = zeta_obs * s(t) * g_obs_k / ||g_obs_k||
         + zeta_consis * s(t) * g_consis_k / ||g_consis_k||

where g_obs = grad_x sum(r^2) over the member's full slab stack and g_consis
is the slab-overlap-consistency gradient. The update norm is bounded by
(zeta_obs + zeta_consis) * s(t) by construction, independent of sensor count,
residual scale, and noise level -- the scale-explosion mode is structurally
removed.

CHOICE OF s(t), AGAINST THE LITERATURE
--------------------------------------
* DPS (Chung et al., ICLR 2023, Alg. 1) uses a fixed zeta with residual-norm
  normalization zeta_i = zeta / ||y - A(x0_hat)||, so the applied step is
  ~ zeta * grad ||r|| : approximately unit-scaled and CONSTANT across noise
  levels. Gradient-norm normalization zeta * g/||g|| (as used in FreeDoM,
  Yu et al. 2023, and common DPS re-implementations) is the tighter variant of
  the same idea: it bounds the update norm exactly instead of approximately.
* PiGDM (Song et al., ICLR 2023) instead folds guidance into the score with
  the adaptive weight r_t^2 = sigma_t^2 / (sigma_t^2 + 1) (unit data variance),
  so guidance decays together with the score coefficient as sigma -> 0.
  For this VE-SDE (sigma in [0.1, 20], per-field z-scored data so sigma_data=1)
  r_t^2 ~= 1 over most of the ladder and only decays below sigma ~ 1, i.e. its
  sole practical effect here is late-stage attenuation.

Primary choice: s(t) = 1 (DPS convention, constant zeta). Rationale: the
per-element magnitude of a unit-norm update spread over a member's ~9.2M
elements is zeta/3000 RMS, far below the sigma_min = 0.1 noise floor at the
end of sampling, so constant-zeta cannot dominate the terminal steps the way
the un-normalized update did. The PiGDM-motivated variant
sigma_scale="pigdm" (s(t) = sigma^2/(1+sigma^2)) is exposed and tuned as an
ablation arm, and the per-sigma traces (obs_rmse, max|x|, gradient norms)
adjudicate empirically.

guidance_mode="raw" reproduces the upstream update dx = zeta_obs*g_obs +
zeta_consis*g_consis (mathematically identical to upstream's single combined
gradient, same RNG consumption, same 1e8 clamp) so the un-normalized
alpha=1.0/beta=0.004 stability-edge arm can run through the same traced code
path.

This module monkey-patches s3gm3d.s3gm_reconstruct_ensemble via install();
it edits NO repo files.
"""
from __future__ import annotations
from paths import SRC_DIR

import json
import math
import sys
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


import methods.s3gm.s3gm3d as S


def s3gm_reconstruct_norm(
    net,
    sde,
    obs_video: torch.Tensor,       # [1, Z, C, H_pad, W_pad]
    obs_mask_video: torch.Tensor,  # [1, Z, C, H_pad, W_pad]
    Nz: int,
    K: int = 1,
    t_win: int = 10,
    overlap: int = 1,
    zeta_obs: float = 300.0,
    zeta_consis: float = 0.0,
    sigma_scale: str = "none",     # "none" (DPS-convention) | "pigdm"
    guidance_mode: str = "norm",   # "norm" | "raw" (upstream-equivalent)
    snr: float = 0.128,
    n_corrector_steps: int = 0,
    eps: float = 1e-12,
    device: Optional[torch.device] = None,
    progress: bool = True,
    collect_trace: bool = True,
):
    """Guided reconstruction with normalized + step-sized guidance.

    Structure and RNG consumption are byte-for-byte s3gm3d.s3gm_reconstruct;
    only the guidance update differs (see module docstring).

    Returns (out [K, Z, C, H_pad, W_pad], trace: list of per-step dicts).
    """
    device = device or obs_video.device
    C, H, W = obs_video.shape[2], obs_video.shape[3], obs_video.shape[4]
    ol = int(overlap)
    b = int(np.ceil((Nz - ol) / (t_win - ol)))
    ns_real = b * (t_win - ol) + ol
    shape = (K, b, t_win, C, H, W)

    y = obs_video.to(device)
    m = obs_mask_video.to(device)
    nobs = float(m.sum()) * K

    def x_to_sample(xx):
        out = xx.new_zeros(K, ns_real, C, H, W)
        for i in range(b):
            i_inv = b - i - 1
            out[:, i_inv * (t_win - ol): i_inv * (t_win - ol) + t_win] = xx[:, i_inv]
        return out

    def _unit_per_member(g):
        gv = g.reshape(K, -1)
        n = gv.norm(dim=1)
        u = (gv / n.clamp_min(1e-20)[:, None]).reshape_as(g)
        return u, n

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    x = sde.prior_sampling(list(shape)).to(device).float()
    x_mean = torch.zeros_like(x)

    trace = []
    n_clamped, dx_max_all = 0, 0.0
    it = tqdm(range(sde.N), desc="S3GM norm sampling", leave=False) if progress else range(sde.N)
    for i in it:
        t = timesteps[i]
        vec_t = torch.ones(K * b, device=device).float() * t
        xb = x.reshape(K * b, t_win, C, H, W)

        if n_corrector_steps > 0:
            xb = S._langevin_corrector(net, sde, xb, vec_t, snr, n_corrector_steps)

        z = torch.randn_like(x)
        zb = z.reshape(K * b, t_win, C, H, W)

        with torch.enable_grad():
            inp = xb.detach().clone().requires_grad_(True)
            score = S._net_fn(net, sde, inp, vec_t)
            with torch.no_grad():
                f, G = sde.discretize(inp.detach(), vec_t)
                rev_f = f - G[:, None, None, None, None] ** 2 * score.detach()
                temp_mean = inp.detach() - rev_f
                temp_u = temp_mean + G[:, None, None, None, None] * zb

            _, std = sde.marginal_prob(inp, vec_t)
            sigma_t = float(std.reshape(-1)[0])
            x0_hat = (std[:, None, None, None, None] ** 2 * score + inp)
            x0_hat = x0_hat.reshape(K, b, t_win, C, H, W)
            vol = x_to_sample(x0_hat)[:, :Nz]

            resid = (y - vol) * m
            loss_dps = torch.sum((resid ** 2).reshape(K, -1), dim=-1).sum()

            if b > 1:
                a_ = x0_hat[:, :-1, (t_win - ol):t_win].detach()
                c_ = x0_hat[:, 1:, :ol]
                loss_consis = torch.sum(((a_ - c_) ** 2).reshape(K, b - 1, -1), dim=-1)
                loss_consis = torch.sum(loss_consis)
            else:
                loss_consis = None

            g_obs = torch.autograd.grad(loss_dps, inp,
                                        retain_graph=(loss_consis is not None))[0]
            if loss_consis is not None:
                g_consis = torch.autograd.grad(loss_consis, inp)[0]
            else:
                g_consis = torch.zeros_like(g_obs)

            with torch.no_grad():
                if sigma_scale == "pigdm":
                    s_t = sigma_t ** 2 / (1.0 + sigma_t ** 2)
                elif sigma_scale == "none":
                    s_t = 1.0
                else:
                    raise ValueError(f"unknown sigma_scale {sigma_scale!r}")

                gv_obs = g_obs.reshape(K, -1)
                gv_con = g_consis.reshape(K, -1)
                n_obs_ = gv_obs.norm(dim=1)
                n_con_ = gv_con.norm(dim=1)

                if guidance_mode == "norm":
                    u_obs = (gv_obs / n_obs_.clamp_min(1e-20)[:, None]).reshape_as(g_obs)
                    u_con = (gv_con / n_con_.clamp_min(1e-20)[:, None]).reshape_as(g_consis)
                    dx = (zeta_obs * s_t) * u_obs + (zeta_consis * s_t) * u_con
                elif guidance_mode == "raw":
                    # Upstream-equivalent: raw gradient of the combined loss with
                    # the coefficients folded in, plus upstream's 1e8 clamp.
                    dx = zeta_obs * g_obs + zeta_consis * g_consis
                    n_clamped += int((dx.abs() > 1e8).sum())
                    dx = torch.clamp(dx, min=-1e8, max=1e8)
                else:
                    raise ValueError(f"unknown guidance_mode {guidance_mode!r}")

                dx_max = float(dx.abs().max())
                dx_max_all = max(dx_max_all, dx_max)
                xb = (temp_u - dx).detach()

                if collect_trace:
                    trace.append({
                        "i": int(i),
                        "t": float(t),
                        "sigma": sigma_t,
                        "s_t": float(s_t),
                        "obs_rmse_x0hat": math.sqrt(float(loss_dps) / max(nobs, 1.0)),
                        "g_obs_norm": float(n_obs_.mean()),
                        "g_consis_norm": float(n_con_.mean()),
                        "dx_max": dx_max,
                        "dx_rms": float(dx.pow(2).mean().sqrt()),
                        "x_max": float(xb.abs().max()),
                    })

        x = xb.reshape(K, b, t_win, C, H, W)
        x_mean = temp_mean.detach().reshape(K, b, t_win, C, H, W)
        if progress:
            it.set_postfix_str(f"obs_rmse={trace[-1]['obs_rmse_x0hat']:.3e}" if trace else "")

    out = x_to_sample(x_mean)[:, :Nz]
    with torch.no_grad():
        obs_rmse = float(torch.sqrt((((out - y) * m) ** 2).sum() / max(nobs, 1.0)))
        amp = float(out.abs().max())
    print(f"[sampler-norm] N={sde.N} K={K} mode={guidance_mode} "
          f"zeta_obs={zeta_obs:g} zeta_consis={zeta_consis:g} sigma_scale={sigma_scale} "
          f"obs_rmse_z={obs_rmse:.4e} max_abs={amp:.4e} dx_max={dx_max_all:.3e} "
          f"clamped_elems={n_clamped} "
          f"{'DIVERGED' if (obs_rmse > 10.0 or not np.isfinite(obs_rmse)) else 'ok'}",
          flush=True)
    return out, trace


# ---------------------------------------------------------------------------
# Ensemble wrapper + monkey-patch installer
# ---------------------------------------------------------------------------
_ORIG_ENSEMBLE = S.s3gm_reconstruct_ensemble


def s3gm_reconstruct_ensemble_norm(net, sde, obs_video, obs_mask_video, Nz,
                                   K=1, chunk=1, trace_store=None,
                                   trace_limit=8, **kw):
    """K posterior draws in chunks, through the normalized-guidance sampler.

    Accepts and DISCARDS the upstream guidance kwargs (alpha, beta) so it is a
    drop-in replacement for s3gm3d.s3gm_reconstruct_ensemble inside
    eval_s3gm3d.py; guidance strength comes from zeta_obs/zeta_consis instead.
    """
    kw.pop("alpha", None)
    kw.pop("beta", None)
    outs = []
    done = 0
    while done < K:
        k = min(chunk, K - done)
        out, tr = s3gm_reconstruct_norm(net, sde, obs_video, obs_mask_video,
                                        Nz, K=k, **kw)
        if trace_store is not None and len(trace_store) < trace_limit:
            trace_store.append(tr)
        outs.append(out.detach())
        done += k
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return torch.cat(outs, dim=0)


def install(zeta_obs, zeta_consis, sigma_scale="none", guidance_mode="norm",
            trace_store=None, trace_limit=8):
    """Route s3gm3d.s3gm_reconstruct_ensemble through normalized guidance.

    eval_s3gm3d.py resolves the sampler as an attribute of the s3gm3d module
    at call time, so patching the module attribute redirects it without
    touching any repo file. The alpha/beta it passes are ignored.
    """
    def patched(net, sde, obs_video, obs_mask_video, Nz, K=1, chunk=1, **kw):
        kw.pop("alpha", None)
        kw.pop("beta", None)
        return s3gm_reconstruct_ensemble_norm(
            net, sde, obs_video, obs_mask_video, Nz, K=K, chunk=chunk,
            zeta_obs=zeta_obs, zeta_consis=zeta_consis,
            sigma_scale=sigma_scale, guidance_mode=guidance_mode,
            trace_store=trace_store, trace_limit=trace_limit, **kw)

    S.s3gm_reconstruct_ensemble = patched
    print(f"[s3gm_norm_guidance] installed: mode={guidance_mode} "
          f"zeta_obs={zeta_obs:g} zeta_consis={zeta_consis:g} "
          f"sigma_scale={sigma_scale} (alpha/beta from callers are ignored)",
          flush=True)


def uninstall():
    S.s3gm_reconstruct_ensemble = _ORIG_ENSEMBLE


# ---------------------------------------------------------------------------
# Diagnostics figure (task 4)
# ---------------------------------------------------------------------------
def save_diag_figure(traces, path, title=""):
    """traces: list of per-draw step-trace lists. 3-panel per-sigma figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for tr in traces:
        sig = [s["sigma"] for s in tr]
        axs[0].plot(sig, [s["obs_rmse_x0hat"] for s in tr], lw=1)
        axs[1].plot(sig, [s["x_max"] for s in tr], lw=1)
        axs[2].plot(sig, [s["g_obs_norm"] for s in tr], lw=1, label="||g_obs||" if tr is traces[0] else None)
        axs[2].plot(sig, [s["g_consis_norm"] for s in tr], lw=1, ls="--", label="||g_consis||" if tr is traces[0] else None)
        axs[2].plot(sig, [max(s["dx_rms"], 1e-20) for s in tr], lw=1, ls=":", label="rms(dx)" if tr is traces[0] else None)
    axs[0].axhline(math.sqrt(2.0), color="k", ls=":", lw=0.8)
    axs[0].set_ylabel("obs_rmse (z-units, x0_hat)")
    axs[1].set_ylabel("max|x|")
    axs[2].set_ylabel("gradient / step norms")
    axs[2].set_yscale("log")
    axs[2].legend(fontsize=8)
    for ax in axs:
        ax.set_xlabel("sigma(t)")
        ax.set_xscale("log")
        ax.invert_xaxis()
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


print("[s3gm_norm_guidance] module loaded", flush=True)
