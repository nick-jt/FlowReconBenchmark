"""Cost instrumentation for the 3-D FNO architecture ablation.

Reports, greppable and as JSON:
  train_step_seconds        one optimizer step, full 125^3 grid, no validation
  train_peak_gpu_gb         peak CUDA allocation during that step
  inference_seconds_field   wall clock for one full 125^3 field at the eval NFE
  inference_peak_gpu_gb     peak CUDA allocation during that inference
"""

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

from methods.dmfgen.helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from methods.dmfgen.evaluate import _build_model, _normalize_eval_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-obs", type=int, nargs="+", default=[19531, 19531])
    ap.add_argument("--cond-fields", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    cfg = _normalize_eval_config(json.load(open(run_dir / "args.json")))
    dev = torch.device("cuda:0")

    ds = TurbulentCombustionH5Dataset(
        cfg["data"], split="val", train_ratio=cfg.get("train_ratio", 0.75),
        field_names=cfg.get("field_names"), seed=cfg.get("seed", 42),
        time_stride=cfg.get("time_stride", 1),
        stats_path=str(run_dir / "dataset_stats.pt"))

    model = _build_model(cfg, ds)
    ck = torch.load(run_dir / a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(dev)

    item = ds[0]
    coords1 = item["coords"].unsqueeze(0).to(dev)
    fields1 = item["fields"].unsqueeze(0).to(dev)
    res = {"run_dir": str(run_dir), "ckpt": a.ckpt, "nfe": a.nfe,
           "batch_size": a.batch_size, "n_points": int(coords1.shape[1]),
           "gpu": torch.cuda.get_device_name(0),
           "data_path": cfg.get("data"),
           "measurement_note": "timed loops are data-resident: no dataset/HDF5 access inside them",
           "params_numel": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
           "params_real_dof": int(sum(p.numel() * (2 if p.is_complex() else 1)
                                      for p in model.parameters() if p.requires_grad))}

    # ---- training step (full grid, no validation) ----
    B = a.batch_size
    coords = coords1.expand(B, -1, -1).contiguous()
    fields = fields1.expand(B, -1, -1).contiguous()
    torch.manual_seed(0)
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords, fields_full=fields, cond_fields=a.cond_fields,
        n_obs_min=a.n_obs, n_obs_max=a.n_obs)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Reproduce the production step exactly, including the binned spectral loss:
    # the trainer appends a contiguous grid block to the query set, so the
    # measured step time is the one that belongs in the paper's cost table.
    spec_w = float(cfg.get("spectral_weight", 0.0) or 0.0)
    blk_shape = None
    coords_s, fields_s = coords, fields
    if spec_w > 0.0:
        from methods.dmfgen.spectral_loss import block_indices
        gs = (coords.shape[1] ** (1.0 / 3.0))
        side = int(round(gs))
        b_idx, blk_shape = block_indices(
            (side, side, side), int(cfg.get("spectral_block", 32)), coords.device)
        coords_s = torch.cat([coords, coords[:, b_idx]], dim=1)
        fields_s = torch.cat([fields, fields[:, b_idx]], dim=1)
    res["spectral_weight"] = spec_w
    res["spectral_block_shape"] = list(blk_shape) if blk_shape else None

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss, _ = model.training_loss(
            x1=fields_s, coords=coords_s, obs_coords=oc, obs_values=ov,
            obs_mask=om, obs_field_ids=ofid, obs_indices=oi,
            compute_metrics=False,
            spectral_block_shape=blk_shape,
            spectral_weight=spec_w if blk_shape is not None else 0.0,
            spectral_bins=int(cfg.get("spectral_bins", 12)),
            spectral_window=bool(cfg.get("spectral_window", False)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    train_step(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(a.repeats):
        train_step()
    torch.cuda.synchronize()
    res["train_step_seconds"] = (time.perf_counter() - t0) / a.repeats
    res["train_peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    del opt
    torch.cuda.empty_cache()

    # ---- inference: one full 125^3 field ----
    model.eval()
    torch.manual_seed(0)
    oc1, ov1, om1, oi1, ofid1 = build_sparse_condition(
        coords_full=coords1, fields_full=fields1, cond_fields=a.cond_fields,
        n_obs_min=a.n_obs, n_obs_max=a.n_obs)
    with torch.no_grad():
        def infer():
            return model.sample(
                coords=coords1, obs_coords=oc1, obs_values=ov1, obs_mask=om1,
                obs_field_ids=ofid1, n_steps=a.nfe, clamp_indices=oi1,
                ode_solver=cfg.get("ode_solver", "euler"))
        infer(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(a.repeats):
            infer()
        torch.cuda.synchronize()
        res["inference_seconds_field"] = (time.perf_counter() - t0) / a.repeats
        res["inference_peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30

    out = a.out or str(run_dir / "cost_instrumentation.json")
    json.dump(res, open(out, "w"), indent=1)
    print("[cost] " + " ".join(f"{k}={v}" for k, v in res.items()), flush=True)
    print(f"[cost] train_step_seconds={res['train_step_seconds']:.4f} "
          f"train_peak_gpu_gb={res['train_peak_gpu_gb']:.2f} "
          f"inference_seconds_field={res['inference_seconds_field']:.4f} "
          f"inference_peak_gpu_gb={res['inference_peak_gpu_gb']:.2f}", flush=True)
    print(f"[cost] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
