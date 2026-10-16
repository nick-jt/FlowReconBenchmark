"""Gen4Turb on the shared JHTDB protocol, scored on the central 120^3 crop.

The upstream 3D U-Net needs every side divisible by 8, so it reconstructs the
central 120^3 crop of the held-out cube (offset 2 per axis, the crop written by
data_prep.export_for_baselines --crop 120). The sensors are the canonical draw
(anchors_jhu.draw_sensors: seed*777 + held-out index, all 50 held-out snapshots)
restricted to the crop; U_y and p are unobserved. Members leave the sampler in the
min-max units of their training run (load_norm), are mapped back to physical units and scored in the shared
z-score units (train-split stats of --run-dir) on the crop points. The TUNE/TEST
split by held-out parity is applied afterwards by analysis.paired_ci_jhu.

    FRB_GEN4TURB_ROOT=/path/to/Gen4Turbulence python -m methods.gen4turb.eval_jhu \
        --ckpt models_uxuz_anneal/model_4190.pt --run-dir <JHU run> --out <json>
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from paths import site_path
from methods.classical.anchors_jhu import (COND_FIELDS, DATA, FIELD_NAMES, draw_sensors,
                                                 enable_percentile_fastpath, load_split)
from methods.dmfgen.ensemble_eval import check_canonical_fingerprint, ensemble_metrics
from methods.dmfgen.helpers import TurbulentCombustionH5Dataset

SIDE, CROP, OFF = 125, 120, 2


def load_norm(dm: Path, ckpt: str):
    """Min-max shift and span [C], sigma_data and their source for the run that wrote `ckpt`.

    train_jhu.py keeps MIN_u.npy, MAX_u.npy and Par.pkl next to its checkpoints. Earlier runs
    (models/, models_uxuz/, models_uxuz_anneal/) used the shared upstream data/ files and dm/Par.pkl.
    """
    run = (dm / ckpt).parent
    src, par = (run, run / "Par.pkl") if (run / "Par.pkl").exists() else (dm.parent / "data", dm / "Par.pkl")
    lo = np.load(src / "MIN_u.npy").astype(np.float32)
    span = np.load(src / "MAX_u.npy").astype(np.float32) - lo
    return lo, span, float(pickle.load(open(par, "rb"))["sigma_data"]), str(src)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint relative to the upstream dm/ directory")
    p.add_argument("--run-dir", required=True, help="JHU run whose dataset_stats.pt sets the z-scoring")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n-obs", type=int, default=19531, help="sensors per observed channel on the full cube")
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dm = site_path("gen4turb_root") / "3_flow_reconstruction" / "dm"
    sys.path.insert(0, str(dm))
    from utils.architecture import Unet
    from utils.diffusion import ElucidatedDiffusion

    dev = "cuda:0"
    enable_percentile_fastpath()
    lo, span, sigma_data, norm_src = load_norm(dm, args.ckpt)
    print(f"[gen4turb] min-max from {norm_src}: MIN {lo} span {span}; sigma_data {sigma_data:.5f}", flush=True)
    net =Unet(dim=16, dim_mults=(1, 2, 4, 8), channels=4, self_condition=True, flash_attn=True)
    model = ElucidatedDiffusion(net, channels=4, image_size_h=CROP, image_size_w=CROP,
                                image_size_d=CROP, sigma_data=sigma_data).to(dev)
    model.load_state_dict(torch.load(dm / args.ckpt, map_location="cpu"))
    model.eval()

    ds =TurbulentCombustionH5Dataset(DATA, split="val", train_ratio=0.75, field_names=FIELD_NAMES,
                                      seed=42, time_stride=1, stats_path=str(Path(args.run_dir) / "dataset_stats.pt"))
    mean, std = ds.mean.numpy(), ds.std.numpy()
    n_snap = min(args.n_snapshots, len(ds))
    Y = load_split(ds, ds.indices[:n_snap])                       # [n, N, C], shared z-score
    crop = (slice(OFF, OFF + CROP),) * 3

    # the crop must be the exported one Gen4Turb was trained on (frame 150 + i = held-out i)
    u_np = np.load(dm.parent / "data" / "u.npy", mmap_mode="r")    # [C, 200, 120, 120, 120]
    ref = np.moveaxis(np.asarray(u_np[:, int(ds.indices[0])]), 0, -1)
    got = Y[0].reshape(SIDE, SIDE, SIDE, -1)[crop] * std + mean
    if not np.allclose(got, ref, atol=1e-4 * np.abs(ref).max()):
        raise SystemExit("[gen4turb] crop or frame mapping does not match the exported u.npy")

    snaps = []
    for i in range(n_snap):
        sens = draw_sensors(ds.coords, torch.from_numpy(Y[i]), snap=i, n_obs=args.n_obs,
                            seed=args.seed, device=dev)
        n_tot = sum(int(v[0].size) for v in sens.values())
        check_canonical_fingerprint(i, n_tot, sum(int(v[0].sum()) for v in sens.values()),
                                    args.seed, COND_FIELDS, [args.n_obs] * len(COND_FIELDS))
        truth = Y[i].reshape(SIDE, SIDE, SIDE, -1)[crop]          # [CROP]*3 + [C]
        x_norm = np.moveaxis((truth * std + mean - lo) / span, -1, 0)
        mask = np.zeros((4, CROP, CROP, CROP), dtype=np.float32)
        kept = 0
        for fld, (idx, _) in sens.items():
            a = np.stack(np.unravel_index(idx, (SIDE,) * 3)) - OFF
            a = a[:, np.all((a >= 0) & (a < CROP), axis=0)]
            mask[fld, a[0], a[1], a[2]] = 1.0
            kept += a.shape[1]
        cond = torch.from_numpy(np.concatenate([x_norm * mask, mask])[None]).float().to(dev)
        ens = []
        t0 = time.time()
        with torch.no_grad():
            for k in range(args.K):
                torch.manual_seed(9000 + 97 * i + k)
                ens.append(model.sample(cond)[0].float().cpu().numpy())
        dt = (time.time() - t0) / args.K
        ens = np.moveaxis(np.stack(ens), 1, -1) * span + lo          # [K, CROP^3 grid, C] physical
        ens_z = ((ens - mean) / std).reshape(args.K, -1, 4)
        m = ensemble_metrics(ens_z, truth.reshape(-1, 4), list(FIELD_NAMES))
        m.update(snapshot=i, sensors_kept=kept, sensors_drawn=n_tot, sample_seconds=round(dt, 3))
        snaps.append(m)
        a = m["aggregate"]
        print(f"[gen4turb] snap {i}: relL2 {a['rel_l2_mean']:.4f} crps {a['crps']:.4f} "
              f"spr/err {a['spread_error_ratio']:.3f} kept {kept}/{n_tot} {dt:.2f} s/member", flush=True)

    out = {
        "snapshots": snaps,
        "protocol": {"data": DATA, "split": "val (all 50 held-out snapshots; parity split downstream)",
                     "sensor_draw": "anchors_jhu.draw_sensors, seed*777 + held-out index, cuda",
                     "scored_points": f"central {CROP}^3 crop, offset {OFF}",
                     "standardization": f"train-split stats of {args.run_dir}",
                     "min_max": {"source": norm_src, "min": lo.tolist(), "span": span.tolist(),
                                 "sigma_data": sigma_data},
                     "sampler": "upstream ElucidatedDiffusion.sample (32-step Heun, S_churn 80, clamp)"},
        "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
