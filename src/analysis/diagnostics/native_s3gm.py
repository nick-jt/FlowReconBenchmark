"""S3GM on its native dataset: does our port reproduce Li et al. (2024), Fig. 3c?

Published target (li2024s3gm, Fig. 3c; Source Data Fig. 3, sheet Fig_3_c, rows
"S$^3$GM"): nRMSE of Kolmogorov flow reconstructed at k=2, Re=1500 (outside the
training range Re=100-1050) from velocity (u, v) observed on every 8th (4th) point
of the 64x64 grid in every frame; three solutions with different initial
conditions. Population std (ddof=0) reproduces the sheet's "Std." column.

The released kol_v0 EMA weights (Zenodo 10.5281/zenodo.13925732) are sampled by
  upstream : upstream sampler.utils.s3gm_sample_2d with the sample_kol.ipynb
             settings (1000 steps, no corrector, alpha_case 0.5 -> alpha =
             alpha_case * factor, beta 0.4, overlap 2, 106 frames, seed 123),
             obs_type 'regular';
  port     : methods.s3gm.s3gm3d.s3gm_reconstruct (our generate_parallel_2d
             port, used for the 3D JHTDB rows) with our UNetVideoModel copy, the
             same observations and the same seed;
  port2d   : methods.s3gm.adapter.dps_sample (our 2D benchmark sampler:
             predictor-corrector DPS, 5 Langevin steps, alpha_obs 1, no window
             coupling) on 11 independent 10-frame windows, 1000 steps;
  port2d_native : port2d with the upstream guidance weight (alpha_obs = alpha) and
             no corrector, separating the settings from the missing window coupling;
  *_denoised : the port2d arms plus one final Tweedie step at sigma_min (see run_port2d);
  nets     : upstream vs ported UNetVideoModel on identical random inputs.
Windows follow the notebook's draw (seed_samples=1, n_samples=3); window 0 is the
notebook's own sample. Metrics are in physical units over all 106 frames.

Upstream code is imported from the (untrusted) download directory; checkpoints
load with weights_only=True. Run with `python -I` (the script puts src/ on the path):
    python -I src/analysis/diagnostics/native_s3gm.py --root <download dir> \
        --modes nets upstream port port2d --factors 8 4
    python -I src/analysis/diagnostics/native_s3gm.py --summarize
"""
import argparse
import json
import random
import sys
import time
import types
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parent
OUT = REPO / "outputs/native_repro/s3gm"
SUMMARY = REPO / "outputs/paper/diagnostics/native_s3gm.json"
ROOT = REPO / "data/native_repro/s3gm/download"

# Source Data Fig. 3, sheet Fig_3_c (Run1-3, Mean, Std.) for S3GM.
PUBLISHED = {
    8: dict(runs=[0.03000054128227984, 0.024164834934374772, 0.022708695991708511],
            mean=0.025624690736121041, std=0.0031507815069519922),
    4: dict(runs=[0.010100469848804011, 0.010090177662578191, 0.01040630450323001],
            mean=0.01019898400487074, std=0.0001466579331966749),
}
PROVENANCE = {
    "code": "https://github.com/lzy12301/S3GM @ 2343293bf627e7b4afc0d63b11ad37ece0919653",
    "weights_and_data": "Zenodo 10.5281/zenodo.13925732: 'pretrained checkpoints.zip' (kol_v0, EMA, "
                        "epoch 256; md5 422df32bc9cf63310e5e7324dc9778b5), kolmogorov_flow_test.npy "
                        "(md5 78063d14f51e8fd81434cfc33431c2d9)",
    "torch2_workaround": "upstream sampler/sde.py:159 indexes a CPU sigma ladder with CUDA timesteps, "
                         "which torch>=2 rejects; the ladder is moved to the GPU (same values)",
    "windows": "the test file holds one trajectory per (Re, k); the three solutions are taken as three "
               "106-frame windows of the (Re=1500, k=2) trajectory drawn as in sample_kol.ipynb",
}
PORT2D_DIFFERENCES = [
    "single-frame state, no overlapping windows and no overlap-consistency guidance (upstream: 13 "
    "jointly denoised 10-frame windows, overlap 2, beta 0.4)",
    "fixed guidance weight alpha_obs (1.0 in the 2D configs) instead of alpha_case/sqrt(observed fraction)",
    "5 Langevin corrector steps per predictor step (snr 0.128, step clamped at 1); upstream uses none",
    "continuous sigma(t) step with eps 1e-3 instead of the discrete SMLD ladder with eps 1e-12",
    "returns the last iterate, which keeps noise at sigma(eps) ~ sigma_min; upstream returns the last "
    "predictor mean, a Tweedie denoise at sigma_min",
]
RE_TEST = [50, 125, 575, 1100, 1500]   # sample_kol.ipynb
K_TEST = [2, 4, 6, 8]
S_RE, S_K = 4, 0                        # Re=1500, k=2 (Fig. 3)
N_STEPS, ALPHA_CASE, BETA, GAMMA, OVERLAP, SNR = 1000, 0.5, 0.4, 0.5, 2, 0.128
SEED_SAMPLES, SEED_SAMPLE, N_WINDOWS, T_PRIME_REQ = 1, 123, 3, 100
PORT2D_MODES = ("port2d", "port2d_native", "port2d_denoised", "port2d_native_denoised")


def seed_all(seed):
    """Upstream sampler.utils.setup_seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_opt(ckpt_dir):
    opt = json.loads((ckpt_dir / "config.json").read_text())
    opt["num_channels"] = opt["num_components"] + opt["num_conditions"]
    return namedtuple("OPT", opt.keys())(**opt)


def build_net(cls, opt):
    """models.get_model(opt), for either copy of UNetVideoModel."""
    return cls(in_channels=opt.num_channels, model_channels=opt.nf, out_channels=opt.num_channels,
               num_res_blocks=opt.num_res_blocks, attention_resolutions=opt.attn_resolutions,
               image_size=opt.image_size, dropout=opt.dropout, channel_mult=opt.ch_mult,
               conv_resample=True, dims=opt.dims, num_heads=opt.num_heads, use_rpe_net=True)


def load_ema_weights(net, ckpt_dir):
    """Notebook: load model state, then ema.copy_to(net.parameters())."""
    ck = torch.load(ckpt_dir / "checkpoint.pth", map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in ck["model"].items()}
    net.load_state_dict(state, strict=True)
    shadow = ck["ema"]["shadow_params"]
    params = list(net.parameters())
    if len(shadow) != len(params):
        raise RuntimeError(f"EMA has {len(shadow)} tensors, net has {len(params)} parameters")
    with torch.no_grad():
        for p, s in zip(params, shadow):
            p.copy_(s)
    return int(ck["epoch"])


def native_windows(data_raw, opt):
    """sample_kol.ipynb cells 4 and 6 for (Re=1500, k=2) with n_samples=N_WINDOWS.

    Returns [N_WINDOWS, T', 4, H, W] physical fields (u, v, forcing, 100/Re), the
    start indices, and T'. np.random.RandomState.choice(replace=False) takes a prefix
    of one permutation, so window 0 is the notebook's n_samples=1 draw.
    """
    T = opt.num_frames
    b = int(np.ceil((T_PRIME_REQ - OVERLAP) / (T - OVERLAP)))
    t_prime = b * (T - OVERLAP) + OVERLAP
    _, z = np.meshgrid(np.linspace(0, 2 * np.pi, opt.image_size, endpoint=False),
                       np.linspace(0, 2 * np.pi, opt.image_size, endpoint=False), indexing="ij")
    x_test = data_raw.reshape(5, -1, 400, 64, 64, 2)
    tr = x_test[S_RE, S_K]
    rnd = np.random.RandomState(SEED_SAMPLES + S_RE * len(x_test) + S_K)
    starts = rnd.choice(len(tr) - T - t_prime - 10, N_WINDOWS, replace=False) + 10
    f = np.sin(K_TEST[S_K] * z)[None, None].repeat(t_prime, axis=0)
    r = 1.0 / RE_TEST[S_RE] * 100.0 * np.ones_like(f)
    wins = [np.concatenate([tr[i:i + t_prime].transpose(0, 3, 1, 2), f, r], axis=1) for i in starts]
    return np.stack(wins), [int(s) for s in starts], t_prime


def nrmse(pred, ref):
    """pred, ref: [T, C, H, W] physical. uv_frame_mean is the primary metric;
    all_channels_frame_mean is the notebook's cal_rmse(samples[0], x_pred[0])."""
    def per_frame(a, b):
        return np.sqrt(((a - b) ** 2).reshape(len(a), -1).sum(1) / (b ** 2).reshape(len(b), -1).sum(1))
    uv = per_frame(pred[:, :2], ref[:, :2])
    return {"uv_frame_mean": float(uv.mean()),
            "all_channels_frame_mean": float(per_frame(pred, ref).mean()),
            "uv_global": float(np.linalg.norm(pred[:, :2] - ref[:, :2]) / np.linalg.norm(ref[:, :2])),
            "uv_per_frame": uv.tolist()}


def import_upstream(code_dir):
    """Upstream imports tensorflow only for tf.io.gfile in trainer/utils.restore_checkpoint,
    which this script replaces with a weights_only load. The stub is removed after the
    import because einops picks its backends from sys.modules."""
    import matplotlib
    matplotlib.use("Agg")
    stub = "tensorflow" not in sys.modules
    if stub:
        sys.modules["tensorflow"] = types.ModuleType("tensorflow")
    sys.path.insert(0, str(code_dir))
    try:
        import models
        import sampler.utils as su
        from sampler.sde import VESDE
    finally:
        if stub:
            del sys.modules["tensorflow"]
    return models, su, VESDE


def run_upstream(su, VESDE_up, net, opt, win_scaled, factor, t_prime, device, seed):
    """sample_kol.ipynb cell 8 with obs_type='regular', scale=factor."""
    T = opt.num_frames
    sde = VESDE_up(opt, sigma_min=opt.beta_min, sigma_max=opt.beta_max, N=N_STEPS)
    # Upstream targets torch 1.12; torch>=2 refuses to index the CPU sigma ladder with
    # CUDA timesteps (sde.py:159). The same ladder held on the device is numerically identical.
    sde.discrete_sigmas = sde.discrete_sigmas.to(device)
    sparsity = 1 - 1 / factor ** 2
    alpha = ALPHA_CASE / np.sqrt(1 - sparsity)
    reap = lambda x, n: torch.stack([x for _ in range(n)])
    latent_mask = torch.ones([T, 1, 1, 1]).float().to(device)
    obs_mask = torch.zeros([T, 1, 1, 1]).float().to(device)
    frame_indices = torch.arange(T).to(device)
    net_fn = lambda x, t: net(x, x0=x, timesteps=t, latent_mask=reap(latent_mask, len(x)),
                              obs_mask=reap(obs_mask, len(x)), frame_indices=reap(frame_indices, len(x)))[0]
    transform = lambda x: x[:, :t_prime, :2, ::factor, ::factor]
    seed_all(seed)
    with torch.no_grad():
        y = transform(win_scaled[None])
        x = su.s3gm_sample_2d(opt, net_fn, sde, y, transform, alpha=alpha, beta=BETA, gamma=GAMMA,
                              T_prime_y=t_prime, T_prime=t_prime, overlap=OVERLAP,
                              corrector=su.NoneCorrector, snr=SNR, device=device, dtype="float",
                              eps=1e-12, probability_flow=False, continuous=True)
    return np.asarray(x[0])


def obs_mask_video(shape, factor):
    m = torch.zeros(shape)
    m[..., :2, ::factor, ::factor] = 1.0
    return m


def run_port(net, opt, win_scaled, factor, t_prime, device, seed):
    from methods.s3gm import s3gm3d
    from methods.s3gm.sde import VESDE
    sde = VESDE(config=None, sigma_min=opt.beta_min, sigma_max=opt.beta_max, N=N_STEPS)
    m = obs_mask_video((1,) + win_scaled.shape, factor).to(device)
    y = torch.from_numpy(win_scaled[None]).float().to(device) * m
    seed_all(seed)
    out = s3gm3d.s3gm_reconstruct(net, sde, y, m, Nz=t_prime, K=1, t_win=opt.num_frames, overlap=OVERLAP,
                                  alpha=ALPHA_CASE * factor, beta=BETA, snr=SNR, n_corrector_steps=0,
                                  eps=1e-12, device=device, progress=False)
    return out[0].cpu().numpy()


class FramesAsChannels(torch.nn.Module):
    """Presents a T-frame video net to dps_sample, which works on single frames
    [B, 1, C', H, W]: the T frames ride in C' = T*C and are unfolded for the net."""

    def __init__(self, net, T, C):
        super().__init__()
        self.net, self.T, self.C = net, T, C

    def forward(self, x, x0, timesteps, frame_indices, obs_mask, latent_mask):
        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
        v = x.reshape(B, self.T, self.C, H, W)
        out, _ = self.net(v, x0=v, timesteps=timesteps,
                          frame_indices=torch.arange(self.T, device=x.device).expand(B, self.T),
                          obs_mask=torch.zeros(B, self.T, 1, 1, 1, device=x.device),
                          latent_mask=torch.ones(B, self.T, 1, 1, 1, device=x.device))
        return out.reshape(B, 1, self.T * self.C, H, W), None


def run_port2d(net, opt, win_scaled, factor, t_prime, device, seed, n_corrector=5, alpha_obs=1.0,
               final_denoise=False):
    """final_denoise adds one Tweedie step x + sigma_end^2 * score at the terminal level
    sigma(eps=1e-3) ~ sigma_min. dps_sample returns the last iterate, which still carries that
    noise; the upstream sampler removes it by returning the last predictor mean."""
    from methods.s3gm.adapter import _score_fn_nograd, dps_sample
    from methods.s3gm.sde import VESDE
    T, C = opt.num_frames, win_scaled.shape[1]
    starts = list(range(0, t_prime - T + 1, T))
    if starts[-1] + T < t_prime:
        starts.append(t_prime - T)
    vids = torch.from_numpy(np.stack([win_scaled[s:s + T] for s in starts])).float()
    m = obs_mask_video(vids.shape, factor)
    shape = (len(starts), T * C) + vids.shape[-2:]
    sde = VESDE(config=None, sigma_min=opt.beta_min, sigma_max=opt.beta_max, N=N_STEPS)
    folded = FramesAsChannels(net, T, C)
    seed_all(seed)
    with torch.enable_grad():
        grid = dps_sample(folded, sde, (shape[0], 1) + shape[1:],
                          (vids * m).reshape(shape).to(device), m.reshape(shape).to(device), device,
                          N_steps=N_STEPS, snr=SNR, n_corrector_steps=n_corrector, alpha_obs=alpha_obs)
    if final_denoise:
        x5 = grid.detach().unsqueeze(1)
        t_end = torch.full((len(starts),), 1e-3, device=device)
        sig = sde.marginal_prob(x5, t_end)[1].view(-1, 1, 1, 1, 1)
        grid = (x5 + sig ** 2 * _score_fn_nograd(folded, sde, x5, t_end)).squeeze(1)
    wins = grid.reshape(len(starts), T, C, *vids.shape[-2:]).detach().cpu().numpy()
    out = np.zeros_like(win_scaled)
    for s, w in zip(starts, wins):   # later windows overwrite the overlap
        out[s:s + T] = w
    return out


def check_nets(net_up, net_port, opt, device):
    """Identical weights and inputs through both UNetVideoModel copies."""
    g = torch.Generator().manual_seed(0)
    T, C, H = opt.num_frames, opt.num_channels, opt.image_size
    res = {}
    for sigma in (opt.beta_min, 1.0, opt.beta_max):
        x = (torch.randn(2, T, C, H, H, generator=g) * (1.0 + sigma)).to(device)
        kw = dict(x0=x, timesteps=torch.full((2,), float(sigma), device=device),
                  frame_indices=torch.arange(T, device=device).expand(2, T),
                  obs_mask=torch.zeros(2, T, 1, 1, 1, device=device),
                  latent_mask=torch.ones(2, T, 1, 1, 1, device=device))
        with torch.no_grad():
            a, b = net_up(x, **kw)[0], net_port(x, **kw)[0]
        res[f"sigma_{sigma:g}"] = {"max_abs_diff": float((a - b).abs().max()),
                                   "max_abs_out": float(a.abs().max())}
    return res


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.root / "ckpt" / "kol_v0"
    opt = load_opt(ckpt_dir)
    mean, std = np.load(ckpt_dir / "scalar.npy")
    data_raw = np.load(args.root / "zenodo/kolmogorov_flow_test.npy")
    wins, starts, t_prime = native_windows(data_raw, opt)
    scaled = wins.copy()
    scaled[:, :, :opt.num_components] = (scaled[:, :, :opt.num_components] - mean) / std
    (OUT / "runs").mkdir(parents=True, exist_ok=True)

    models, su, VESDE_up = import_upstream(args.root / "code")
    from methods.s3gm.unet_video import UNetVideoModel as PortNet
    net_up = build_net(models.UNetVideoModel, opt).to(device).eval()
    net_port = build_net(PortNet, opt).to(device).eval()
    epoch = load_ema_weights(net_up, ckpt_dir)
    load_ema_weights(net_port, ckpt_dir)
    meta = {"ckpt_epoch": epoch, "window_starts": starts, "t_prime": t_prime,
            "scalar_mean": float(mean), "scalar_std": float(std),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "numpy": np.__version__, "data_shape": list(data_raw.shape)}

    if "nets" in args.modes:
        (OUT / "runs/nets.json").write_text(json.dumps({**meta, "nets": check_nets(net_up, net_port, opt, device)}, indent=2))
    for factor in args.factors:
        for w in args.windows:
            for mode in [m for m in args.modes if m != "nets"]:
                t0 = time.perf_counter()
                if mode == "upstream":
                    pred = run_upstream(su, VESDE_up, net_up, opt, scaled[w], factor, t_prime, device, args.seed)
                elif mode == "port":
                    pred = run_port(net_port, opt, scaled[w], factor, t_prime, device, args.seed)
                elif mode in PORT2D_MODES:
                    native = "native" in mode
                    pred = run_port2d(net_port, opt, scaled[w], factor, t_prime, device, args.seed,
                                      n_corrector=0 if native else 5,
                                      alpha_obs=ALPHA_CASE * factor if native else 1.0,
                                      final_denoise=mode.endswith("_denoised"))
                else:
                    raise ValueError(mode)
                sec = time.perf_counter() - t0
                pred = pred.astype(np.float32)
                tag = f"{mode}_f{factor}_w{w}" + ("" if args.seed == SEED_SAMPLE else f"_s{args.seed}")
                np.save(OUT / f"runs/{tag}.npy", pred)
                phys = pred.copy()
                phys[:, :opt.num_components] = phys[:, :opt.num_components] * std + mean
                rec = {**meta, "mode": mode, "factor": factor, "window": w, "seed": args.seed, "seconds": sec,
                       "nrmse": nrmse(phys, wins[w]), "max_abs_scaled": float(np.abs(pred).max())}
                (OUT / f"runs/{tag}.json").write_text(json.dumps(rec, indent=2))
                print(f"[{tag}] uv nRMSE {rec['nrmse']['uv_frame_mean']:.4f}  "
                      f"all-ch {rec['nrmse']['all_channels_frame_mean']:.4f}  {sec:.0f}s", flush=True)


def summarize():
    recs = [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*_f*_w*.json"))]
    for r in recs:
        r.setdefault("seed", SEED_SAMPLE)

    def stats(rs):
        row = {}
        for key in ("uv_frame_mean", "all_channels_frame_mean", "uv_global"):
            v = np.array([r["nrmse"][key] for r in rs])
            row[key] = {"runs": v.tolist(), "mean": float(v.mean()), "std_ddof0": float(v.std())}
        row["windows"] = [r["window"] for r in rs]
        row["seeds"] = [r["seed"] for r in rs]
        row["seconds_per_run"] = float(np.mean([r["seconds"] for r in rs]))
        return row

    rows = {}
    for factor in sorted({r["factor"] for r in recs}, reverse=True):
        for mode in ("upstream", "port") + PORT2D_MODES:
            rs = sorted((r for r in recs if r["mode"] == mode and r["factor"] == factor),
                        key=lambda r: (r["seed"], r["window"]))
            main = [r for r in rs if r["seed"] == SEED_SAMPLE]
            if main:
                rows[f"{mode}_f{factor}"] = stats(main)
            if len(rs) > len(main):
                rows[f"{mode}_f{factor}_all_seeds"] = stats(rs)
    agree = {}
    for p in sorted((OUT / "runs").glob("port_f*_w*.npy")):
        q = p.with_name(p.name.replace("port_", "upstream_"))
        if q.exists():
            a, b = np.load(q), np.load(p)
            agree[p.stem.replace("port_", "")] = {
                "max_abs_diff_scaled": float(np.abs(a - b).max()),
                "rel_l2_diff_uv": float(np.linalg.norm(a[:, :2] - b[:, :2]) / np.linalg.norm(a[:, :2]))}
    nets = OUT / "runs/nets.json"
    out = {"published": {"source": "Li et al. 2024 (li2024s3gm), Fig. 3c; Source Data Fig. 3 "
                                   "(42256_2024_938_MOESM6_ESM.xlsx), sheet Fig_3_c, rows S$^3$GM",
                         "setup": "Kolmogorov flow 64x64, k=2, Re=1500 (unseen; training Re 100-1050), "
                                  "u and v observed on every factor-th grid point in all frames, "
                                  "metric nRMSE, three solutions",
                         "by_factor": {str(k): v for k, v in PUBLISHED.items()}},
           "reproduced": rows, "port_vs_upstream": agree,
           "nets": json.loads(nets.read_text())["nets"] if nets.exists() else None,
           "meta": {k: recs[0][k] for k in ("ckpt_epoch", "window_starts", "t_prime", "device", "torch")} if recs else None,
           "provenance": PROVENANCE, "port2d_differences": PORT2D_DIFFERENCES}
    SUMMARY.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in rows.items():
        print(f"{k:14s} uv {v['uv_frame_mean']['mean']:.4f} +- {v['uv_frame_mean']['std_ddof0']:.4f}   "
              f"all-ch {v['all_channels_frame_mean']['mean']:.4f}")
    print(json.dumps(agree, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, str(SRC))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--modes", nargs="+", default=["nets", "upstream", "port", "port2d"])
    ap.add_argument("--factors", nargs="+", type=int, default=[8, 4])
    ap.add_argument("--windows", nargs="+", type=int, default=list(range(N_WINDOWS)))
    ap.add_argument("--seed", type=int, default=SEED_SAMPLE)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a)
