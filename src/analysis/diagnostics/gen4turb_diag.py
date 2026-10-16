"""Gen4Turb Ux/Uz asymmetry diagnostic.

Per case it samples K members on the 120^3 crop and reports, per channel in the
shared z-score units: rel L2 of the ensemble mean, RMSE, mean bias, fluctuation
RMSE, amplitude ratio std(pred)/std(truth), correlation, RMSE at the sensor
voxels, and the share of the error energy at |k| <= 4.

Cases (held-out frames and training frames):
  std    canonical sensors on Ux, Uz
  swap   reflection about the x=z plane: spatial axes 0<->2 and channels Ux<->Uz,
         a valid isotropic input; the output is reflected back before scoring
  dense  Ux and Uz observed at every voxel
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from methods.classical.anchors_jhu import DATA, FIELD_NAMES, draw_sensors, load_split
from methods.gen4turb.eval_jhu import load_norm
from methods.dmfgen.helpers import TurbulentCombustionH5Dataset

SIDE, CROP, OFF = 125, 120, 2
ROOT = Path(os.environ["FRB_GEN4TURB_ROOT"]) / "3_flow_reconstruction"
RUN = "outputs/runs/JHU/baseline_fno/fno3d_xcube_DemoN90_20261003_145236"
OUT = Path("outputs/paper/diagnostics/gen4turb_diag")
SWAP = [2, 1, 0, 3]


def reflect(a):            # [C, X, Y, Z] -> channels Ux<->Uz, axes X<->Z
    return np.ascontiguousarray(a[SWAP].transpose(0, 3, 2, 1))


def stats(ens_z, truth_z, mask):
    """ens_z [K, X, Y, Z, C], truth_z [X, Y, Z, C], mask [C, X, Y, Z]."""
    m = ens_z.mean(0)
    k = np.fft.fftfreq(CROP) * CROP
    kk = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    out = {}
    for c, name in enumerate(FIELD_NAMES):
        e = m[..., c] - truth_z[..., c]
        y = truth_z[..., c]
        E = np.abs(np.fft.fftn(e)) ** 2
        sens = mask[c] > 0
        out[name] = dict(
            rel_l2=float(np.linalg.norm(e) / np.linalg.norm(y)),
            rmse=float(np.sqrt((e ** 2).mean())),
            bias=float(e.mean()),
            fluct_rmse=float(e.std()),
            amp_mean=float(m[..., c].std() / y.std()),
            amp_member=float(np.mean([ens_z[j, ..., c].std() for j in range(len(ens_z))]) / y.std()),
            corr=float(np.corrcoef(m[..., c].ravel(), y.ravel())[0, 1]),
            rmse_at_sensors=float(np.sqrt((e[sens] ** 2).mean())) if sens.any() else None,
            lowk_err_share=float(E[kk <= 4].sum() / E.sum()),
        )
    return out


def main():
    dm = ROOT / "dm"
    sys.path.insert(0, str(dm))
    from utils.architecture import Unet
    from utils.diffusion import ElucidatedDiffusion

    dev = "cuda:0"
    u_np =np.load(ROOT / "data" / "u.npy", mmap_mode="r")
    ds = TurbulentCombustionH5Dataset(DATA, split="val", train_ratio=0.75, field_names=FIELD_NAMES,
                                      seed=42, time_stride=1, stats_path=f"{RUN}/dataset_stats.pt")
    mean, std = ds.mean.numpy(), ds.std.numpy()
    OUT.mkdir(parents=True, exist_ok=True)

    ckpts = sys.argv[1].split(",")
    frames = [int(f) for f in sys.argv[2].split(",")]        # absolute H5 frame ids
    cases = sys.argv[3].split(",")
    K = int(sys.argv[4])
    results = []
    for ck in ckpts:
        lo, span, sigma_data, norm_src = load_norm(dm, ck)
        print(f"[diag] {ck}: min-max from {norm_src}, sigma_data {sigma_data:.5f}", flush=True)
        net =Unet(dim=16, dim_mults=(1, 2, 4, 8), channels=4, self_condition=True, flash_attn=True)
        model = ElucidatedDiffusion(net, channels=4, image_size_h=CROP, image_size_w=CROP,
                                    image_size_d=CROP, sigma_data=sigma_data).to(dev)
        model.load_state_dict(torch.load(dm / ck, map_location="cpu"))
        model.eval()
        for fr in frames:
            Y = load_split(ds, [fr])[0]
            truth_z = Y.reshape(SIDE, SIDE, SIDE, -1)[(slice(OFF, OFF + CROP),) * 3]
            ref = np.asarray(u_np[:, fr])                     # [C, X, Y, Z] physical
            assert np.allclose(np.moveaxis(ref, 0, -1), truth_z * std + mean, atol=1e-4)
            snap = fr - 150 if fr >= 150 else fr               # canonical draw index for held-out frames
            sens = draw_sensors(ds.coords, torch.from_numpy(Y), snap=snap, n_obs=19531, seed=0, device=dev)
            mask = np.zeros((4, CROP, CROP, CROP), dtype=np.float32)
            for fld, (idx, _) in sens.items():
                a = np.stack(np.unravel_index(idx, (SIDE,) * 3)) - OFF
                a = a[:, np.all((a >= 0) & (a < CROP), axis=0)]
                mask[fld, a[0], a[1], a[2]] = 1.0
            b = lambda v: v[:, None, None, None]
            for case in cases:
                msk = mask.copy()
                if case == "dense":
                    msk[[0, 2]] = 1.0
                # reflect in physical units, then normalize with each slot's own min/max
                phys, min_ = (reflect(ref), reflect(msk)) if case == "swap" else (ref, msk)
                xin = (phys - b(lo)) / b(span)
                cond = torch.from_numpy(np.concatenate([xin * min_, min_])[None]).float().to(dev)
                ens = []
                with torch.no_grad():
                    for k in range(K):
                        torch.manual_seed(9000 + 97 * snap + k)
                        s = model.sample(cond)[0].float().cpu().numpy() * b(span) + b(lo)
                        ens.append(reflect(s) if case == "swap" else s)
                ens_z = (np.moveaxis(np.stack(ens), 1, -1) - mean) / std
                st = stats(ens_z, truth_z, msk)
                rec = dict(ckpt=ck, frame=fr, case=case, K=K, per_field=st)
                results.append(rec)
                print(json.dumps(rec), flush=True)
                if fr == frames[0]:
                    np.savez_compressed(OUT / f"{Path(ck).parent.name}_{Path(ck).stem}_f{fr}_{case}.npz",
                                        mean=ens_z.mean(0).astype(np.float32), member0=ens_z[0].astype(np.float32),
                                        truth=truth_z.astype(np.float32))
    tag = sys.argv[5] if len(sys.argv) > 5 else "diag"
    json.dump(results, open(OUT / f"{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
