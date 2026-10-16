"""Cost-vs-resolution scaling for the remaining baseline families.

  senseiver : point-based deterministic (Perceiver encode + query decode).
              Chunk-decodable like ours, so the honest expectation is a flat
              memory curve; the differentiator is generative/calibration, not
              memory. Inference re-encodes sensors per chunk (their forward
              is monolithic) - noted in the row.
  gen4turb  : ElucidatedDiffusion voxel UNet (their repo, unmodified).
              Inference rows report ONE denoiser forward (peak memory equals
              the full 32-step sample's; time = 32x the row). Training rows
              are one loss+backward step at batch 1 (their production uses
              batch 6 - even more expensive).

Run from src/:  python benchmark_scaling_baselines.py --mode senseiver|gen4turb
"""
from paths import THIRD_PARTY_DIR

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

GEN4TURB_DM = (f"{THIRD_PARTY_DIR}/"
               "Gen4Turbulence/3_flow_reconstruction/dm")

def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2

def lattice_coords(res, n, dev, g):
    idx = torch.randint(0, res, (1, n, 3), device=dev, generator=g).float()
    return idx / max(res - 1, 1)

def bench_senseiver(args):
    from methods.senseiver.model import Senseiver
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    g = torch.Generator(device=dev).manual_seed(0)
    model = Senseiver(n_fields=4, coord_dim=3, num_latents=128,
                      latent_dim=256).to(dev)
    M = 19531
    ofid = torch.cat([torch.zeros(M, dtype=torch.long),
                      2 * torch.ones(M, dtype=torch.long)]).to(dev)[None]
    rows = []
    # ---- inference: full-field chunked decode
    model.eval()
    for res in args.resolutions:
        n_pts = res ** 3
        oc = torch.rand(1, 2 * M, 3, device=dev, generator=g)
        ov = torch.randn(1, 2 * M, 1, device=dev, generator=g)
        om = torch.ones(1, 2 * M, device=dev)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
                for c0 in range(0, n_pts, args.chunk):
                    n_c = min(args.chunk, n_pts - c0)
                    qc = lattice_coords(res, n_c, dev, g)
                    _ = model(qc, oc, ov, om, ofid).float().cpu()
            torch.cuda.synchronize()
            rows.append(dict(model="senseiver_infer", res=res, n_points=n_pts,
                             seconds=round(time.perf_counter() - t0, 3),
                             peak_mb=round(peak_mb(), 1), ok=True,
                             note="re-encodes sensors per chunk"))
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(model="senseiver_infer", res=res, n_points=n_pts,
                             ok=False, error="OOM"))
            torch.cuda.empty_cache()
        print(rows[-1], flush=True)
    # ---- training: their protocol (batch 6 as trained on FireBench; JHU used 20)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, Q = args.train_batch, 19531
    ofid_b = ofid.expand(B, -1)
    for res in args.resolutions:
        times = []
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(2 + args.iters):
                qc = torch.cat([lattice_coords(res, Q, dev, g) for _ in range(B)], 0)
                x1 = torch.randn(B, Q, 4, device=dev, generator=g)
                oc = torch.rand(B, 2 * M, 3, device=dev, generator=g)
                ov = torch.randn(B, 2 * M, 1, device=dev, generator=g)
                om = torch.ones(B, 2 * M, device=dev)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.autocast("cuda", torch.bfloat16, enabled=True):
                    loss = torch.nn.functional.mse_loss(
                        model(qc, oc, ov, om, ofid_b), x1)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                torch.cuda.synchronize()
                if i >= 2:
                    times.append(time.perf_counter() - t0)
            rows.append(dict(model="senseiver_train", res=res, n_points=res ** 3,
                             sec_per_step=round(float(np.mean(times)), 4),
                             peak_mb=round(peak_mb(), 1), ok=True, batch=B))
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(model="senseiver_train", res=res, n_points=res ** 3,
                             ok=False, error="OOM", batch=B))
            torch.cuda.empty_cache()
        print(rows[-1], flush=True)
    return rows

def bench_gen4turb(args):
    sys.path.insert(0, GEN4TURB_DM)
    from utils.architecture import Unet
    from utils.diffusion import ElucidatedDiffusion
    dev = torch.device("cuda:0")
    rows = []
    for res in args.resolutions:
        if res % 8:
            rows.append(dict(model="gen4turb", res=res, ok=False,
                             error="dims must be divisible by 8"))
            print(rows[-1], flush=True); continue
        torch.cuda.empty_cache()
        try:
            net = Unet(dim=16, dim_mults=(1, 2, 4, 8), channels=4,
                       self_condition=False, flash_attn=True).to(dev)
            model = ElucidatedDiffusion(net, channels=4, image_size_h=res,
                                        image_size_w=res, image_size_d=res,
                                        sigma_data=0.5).to(dev)
            x = torch.randn(1, 4, res, res, res, device=dev)
            cond = torch.randn(1, 8, res, res, res, device=dev)
            # inference: one denoiser forward (peak == full sample's peak)
            torch.cuda.reset_peak_memory_stats()
            sigma = torch.full((1,), 1.0, device=dev)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
                _ = model.preconditioned_network_forward(x, sigma, cond)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
                _ = model.preconditioned_network_forward(x, sigma, cond)
            torch.cuda.synchronize()
            rows.append(dict(model="gen4turb_infer_step", res=res, n_points=res ** 3,
                             seconds=round(time.perf_counter() - t0, 4),
                             peak_mb=round(peak_mb(), 1), ok=True,
                             note="one of 32 sample steps"))
            print(rows[-1], flush=True)
            # training: one loss+backward at batch 1 (production: batch 6)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            torch.cuda.reset_peak_memory_stats()
            times = []
            for i in range(1 + args.iters):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.autocast("cuda", torch.bfloat16, enabled=True):
                    loss = model(x, cond)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                torch.cuda.synchronize()
                if i >= 1:
                    times.append(time.perf_counter() - t0)
            rows.append(dict(model="gen4turb_train", res=res, n_points=res ** 3,
                             sec_per_step=round(float(np.mean(times)), 4),
                             peak_mb=round(peak_mb(), 1), ok=True, batch=1))
            del net, model, x, cond, opt
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(model="gen4turb_train", res=res, n_points=res ** 3,
                             ok=False, error="OOM", batch=1))
            torch.cuda.empty_cache()
        except Exception as e:
            rows.append(dict(model="gen4turb", res=res, ok=False,
                             error=f"{type(e).__name__}: {e}"))
        print(rows[-1], flush=True)
        torch.cuda.empty_cache()
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["senseiver", "gen4turb"])
    p.add_argument("--resolutions", type=int, nargs="+",
                   default=[64, 125, 192, 256, 384, 512, 768, 1024])
    p.add_argument("--chunk", type=int, default=131072)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--train-batch", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    rows = bench_senseiver(args) if args.mode == "senseiver" else bench_gen4turb(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=1)
        print("wrote", args.out)

if __name__ == "__main__":
    main()
