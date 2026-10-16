"""Runtime and memory benchmark: point-cloud reconstruction vs grid baselines.

Three measurements an ICLR reader needs for a 3D architecture claim:

  params   parameter count of every model under comparison
  train    per-step wall time and peak GPU memory at the training protocol
  scale    peak memory and wall time for reconstructing a full field as the
           output resolution grows

The third is the one that separates the two families. Our model evaluates the
velocity pointwise given the sensors, so a reconstruction can be chunked over
query points and peak memory is set by the chunk, not the field. A
convolutional latent model has to hold the whole padded volume plus every
encoder activation, so its memory grows with R^3 and it is additionally locked
to the resolution it was trained at.

Run:  python benchmark_cost.py --mode params|train|scale|grid --out results.json
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import torch


def cuda_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def reset_peak():
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def grid_coords(res, device):
    """Unit-cube coordinates for an res^3 grid, C-ordered."""
    a = torch.linspace(0, 1, res, device=device)
    g = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), dim=-1)
    return g.reshape(1, -1, 3)


def bench_scale(args):
    """Full-field reconstruction cost vs output resolution, our model."""
    from methods.dmfgen.ensemble_eval import load_run
    dev = torch.device("cuda:0")
    model, dataset, cfg = load_run(args.run_dir, args.ckpt, str(dev))
    model.eval()

    n_obs = int(args.n_obs)
    # Fixed sensor budget; only the query count grows.
    obs_coords = torch.rand(1, n_obs * 2, 3, device=dev)
    obs_values = torch.randn(1, n_obs * 2, 1, device=dev)
    obs_mask = torch.ones(1, n_obs * 2, device=dev)
    obs_field_ids = torch.cat([torch.zeros(n_obs, dtype=torch.long),
                               2 * torch.ones(n_obs, dtype=torch.long)]).to(dev)[None]

    def reconstruct(res):
        # Chunked over query points. This is exact for this backbone: the
        # velocity at a point depends only on (t, state there, sensors),
        # so integrating chunk-by-chunk equals integrating jointly. Peak
        # memory is therefore set by the chunk, not by the field size.
        # Lattice coordinates are generated per chunk so the full query
        # set is never resident on the GPU.
        n_pts = res ** 3
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            parts = []
            for c0 in range(0, n_pts, args.chunk):
                idx = torch.arange(c0, min(c0 + args.chunk, n_pts), device=dev)
                k = idx % res; j = (idx // res) % res; i = idx // (res * res)
                cc = (torch.stack([i, j, k], -1).float() / max(res - 1, 1))[None]
                parts.append(model.sample(
                    coords=cc, obs_coords=obs_coords, obs_values=obs_values,
                    obs_mask=obs_mask, obs_field_ids=obs_field_ids,
                    n_steps=args.n_steps,
                    obs_consistency_mode="none",
                ).float().cpu())
            return torch.cat(parts, dim=1)

    # Untimed calls absorb the one-time KeOps JIT compile and CUDA warm-up.
    for _ in range(args.warmup):
        reconstruct(args.resolutions[0])
    rows = []
    for res in args.resolutions:
        n_pts = res ** 3
        reset_peak()
        try:
            times = []
            for _ in range(args.iters):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out = reconstruct(res)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            rows.append(dict(model="ours", res=res, n_points=n_pts, n_steps=args.n_steps,
                             seconds=round(float(np.median(times)), 3),
                             seconds_all=[round(t, 3) for t in times],
                             peak_mb=round(cuda_peak_mb(), 1), ok=True))
            del out
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(model="ours", res=res, n_points=n_pts,
                             seconds=None, peak_mb=None, ok=False, error="OOM"))
            torch.cuda.empty_cache()
        print(rows[-1], flush=True)
        torch.cuda.empty_cache()
    return rows


def bench_grid(args):
    """Same sweep for the latent-FM 3D conv autoencoder."""
    from methods.latent_fm.model import ConvAE3D
    dev = torch.device("cuda:0")
    rows = []
    for res in args.resolutions:
        torch.cuda.empty_cache(); reset_peak()
        try:
            ae = ConvAE3D(n_fields=4, base_ch=48, latent_ch=48, n_levels=3,
                          Num_z=res, Num_y=res, Num_x=res).to(dev).eval()
            x = torch.randn(1, 4, res, res, res, device=dev)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
                z = ae.encode(x) if hasattr(ae, "encode") else ae(x)
                _ = ae.decode(z) if hasattr(ae, "decode") else z
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
                z = ae.encode(x) if hasattr(ae, "encode") else ae(x)
                _ = ae.decode(z) if hasattr(ae, "decode") else z
            torch.cuda.synchronize()
            rows.append(dict(model="convae3d", res=res, n_points=res ** 3,
                             seconds=round(time.perf_counter() - t0, 3),
                             peak_mb=round(cuda_peak_mb(), 1), ok=True,
                             params=count_params(ae)))
            del ae, x, z
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(model="convae3d", res=res, n_points=res ** 3,
                             seconds=None, peak_mb=None, ok=False, error="OOM"))
        except Exception as e:                       # resolution-locked layers
            rows.append(dict(model="convae3d", res=res, n_points=res ** 3,
                             ok=False, error=f"{type(e).__name__}: {e}"))
        print(rows[-1], flush=True)
        torch.cuda.empty_cache()
    return rows


def bench_train(args):
    """Per-step training time and peak memory at the training protocol."""
    from methods.dmfgen.ensemble_eval import load_run
    dev = torch.device("cuda:0")
    model, dataset, cfg = load_run(args.run_dir, args.ckpt, str(dev))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, Q, M = args.batch, args.n_query, int(args.n_obs)
    rows = []
    for _ in range(args.warmup + args.iters):
        pass
    times = []
    reset_peak()
    for i in range(args.warmup + args.iters):
        coords = torch.rand(B, Q, 3, device=dev)
        x1 = torch.randn(B, Q, 4, device=dev)
        oc = torch.rand(B, M * 2, 3, device=dev)
        ov = torch.randn(B, M * 2, 1, device=dev)
        om = torch.ones(B, M * 2, device=dev)
        ofi = torch.cat([torch.zeros(M, dtype=torch.long),
                         2 * torch.ones(M, dtype=torch.long)]).to(dev)[None].expand(B, -1)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            loss, _ = model.training_loss(x1=x1, coords=coords, obs_coords=oc,
                                          obs_values=ov, obs_mask=om,
                                          obs_field_ids=ofi, compute_metrics=False)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize()
        if i >= args.warmup:
            times.append(time.perf_counter() - t0)
    rows.append(dict(model="ours", batch=B, n_query=Q, n_obs=M,
                     sec_per_step=round(float(np.mean(times)), 4),
                     peak_mb=round(cuda_peak_mb(), 1),
                     params=count_params(model)))
    print(rows[-1], flush=True)
    return rows



def bench_grid_train(args):
    """ConvAE3D TRAINING step vs resolution: forward + MSE + backward + opt.

    This is the latent-FM stage-1 cost, and a strict LOWER BOUND on the cost
    of producing a latent-FM model at that resolution (their production run
    uses batch 8; we use batch 1, which is maximally generous to them).
    """
    from methods.latent_fm.model import ConvAE3D
    dev = torch.device("cuda:0")
    rows = []
    for res in args.resolutions:
        torch.cuda.empty_cache(); reset_peak()
        try:
            ae = ConvAE3D(n_fields=4, base_ch=48, latent_ch=48, n_levels=3,
                          Num_z=res, Num_y=res, Num_x=res).to(dev).train()
            opt = torch.optim.AdamW(ae.parameters(), lr=1e-4)
            times = []
            for i in range(1 + args.iters):
                x = torch.randn(1, 4, res, res, res, device=dev)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.autocast("cuda", torch.bfloat16, enabled=True):
                    z = ae.encode(x) if hasattr(ae, "encode") else ae(x)
                    y = ae.decode(z) if hasattr(ae, "decode") else z
                    loss = torch.nn.functional.mse_loss(y, x)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                torch.cuda.synchronize()
                if i >= 1:
                    times.append(time.perf_counter() - t0)
                del x, z, y, loss
            rows.append(dict(model="convae3d_train", res=res, n_points=res ** 3,
                             sec_per_step=round(float(np.mean(times)), 4),
                             peak_mb=round(cuda_peak_mb(), 1), ok=True))
            del ae, opt
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(model="convae3d_train", res=res, n_points=res ** 3,
                             sec_per_step=None, peak_mb=None, ok=False, error="OOM"))
        except Exception as e:
            rows.append(dict(model="convae3d_train", res=res, n_points=res ** 3,
                             ok=False, error=f"{type(e).__name__}: {e}"))
        print(rows[-1], flush=True)
        torch.cuda.empty_cache()
    return rows


def bench_train_scale(args):
    """OUR training step vs dataset resolution.

    Queries are a fixed-size random subsample of the discretization, so the
    step cost is resolution-independent by construction; this measures it
    anyway (coords snapped to each lattice) so the flat line is data, not an
    assertion.
    """
    from methods.dmfgen.ensemble_eval import load_run
    dev = torch.device("cuda:0")
    model, dataset, cfg = load_run(args.run_dir, args.ckpt, str(dev))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, Q, M = args.batch, args.n_query, int(args.n_obs)
    rows = []
    for res in args.resolutions:
        times = []
        torch.cuda.empty_cache(); reset_peak()
        for i in range(args.warmup + args.iters):
            coords = torch.randint(0, res, (B, Q, 3), device=dev).float() / max(res - 1, 1)
            x1 = torch.randn(B, Q, 4, device=dev)
            oc = torch.randint(0, res, (B, M * 2, 3), device=dev).float() / max(res - 1, 1)
            ov = torch.randn(B, M * 2, 1, device=dev)
            om = torch.ones(B, M * 2, device=dev)
            ofi = torch.cat([torch.zeros(M, dtype=torch.long),
                             2 * torch.ones(M, dtype=torch.long)]).to(dev)[None].expand(B, -1)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", torch.bfloat16, enabled=True):
                loss, _ = model.training_loss(x1=x1, coords=coords, obs_coords=oc,
                                              obs_values=ov, obs_mask=om,
                                              obs_field_ids=ofi, compute_metrics=False)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize()
            if i >= args.warmup:
                times.append(time.perf_counter() - t0)
        rows.append(dict(model="ours_train", res=res, n_points=res ** 3,
                         sec_per_step=round(float(np.mean(times)), 4),
                         peak_mb=round(cuda_peak_mb(), 1), ok=True))
        print(rows[-1], flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["scale", "grid", "train", "params", "grid_train", "train_scale"])
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--ckpt", type=str, default="best.pt")
    p.add_argument("--resolutions", type=int, nargs="+",
                   default=[64, 96, 125, 160, 192, 224, 256])
    p.add_argument("--n-obs", type=int, default=19531)
    p.add_argument("--n-steps", type=int, default=16)
    p.add_argument("--chunk", type=int, default=131072)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--n-query", type=int, default=39062)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    fn = {"scale": bench_scale, "grid": bench_grid, "train": bench_train,
          "grid_train": bench_grid_train, "train_scale": bench_train_scale}[args.mode]
    rows = fn(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
