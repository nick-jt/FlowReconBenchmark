"""Gallery field dump for the Gen4Turb row (JHU cross-cube).

Gen4Turb runs in its own environment: the upstream loader's 120^3 centre crop of
the 125^3 cubes (export_for_baselines.py), min-max normalisation, and a shared
1 % mask drawn on that grid. Its gallery panel is therefore its own crop of the
same held-out frame with its own strict draw (U_x and U_z observed at 1 %, U_y and
p unobserved); the identical 125^3 sensor set of the other columns cannot be
injected on a 120^3 grid. The dump records the crop and the draw so the figure
labels the panel accordingly.

    python -m figures.dump_gen4turb_field --ckpt models_uxuz_anneal/model_4190.pt \
        --frame 153 --out outputs/pof/field_dumps/jhu_gen4turb.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from methods.gen4turb.eval_jhu import load_norm
from paths import site_path

DM = site_path("gen4turb_root") / "3_flow_reconstruction" / "dm"
DATA = DM.parent / "data"
G = 120
FIELDS = ["Ux", "Uy", "Uz", "p"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models_uxuz_anneal/model_4190.pt",
                    help="relative to the dm/ directory (the anneal arm's reported checkpoint)")
    ap.add_argument("--frame", type=int, default=153, help="absolute frame (held-out cube: 150 + val index)")
    ap.add_argument("--coverage", type=float, default=0.01)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vi = a.frame - 150
    if vi < 0:
        raise SystemExit("frame must be in the held-out cube (>= 150)")

    sys.path.insert(0, str(DM))
    from utils.architecture import Unet
    from utils.diffusion import ElucidatedDiffusion

    dev = torch.device("cuda:0")
    lo, span, sigma_data, _ = load_norm(DM, a.ckpt)
    net = Unet(dim=16, dim_mults=(1, 2, 4, 8), channels=4, self_condition=True, flash_attn=True)
    model = ElucidatedDiffusion(net, channels=4, image_size_h=G, image_size_w=G, image_size_d=G,
                                sigma_data=sigma_data).to(dev)
    model.load_state_dict(torch.load(str(DM / a.ckpt), map_location="cpu"))
    model.eval()

    u = np.load(DATA / "u.npy", mmap_mode="r")                      # [4, 200, G, G, G]
    mn = lo.reshape(1, 4, 1, 1, 1)
    mx = (lo + span).reshape(1, 4, 1, 1, 1)
    tr = np.asarray(u[:, :150]).reshape(4, -1)
    z_mean, z_std = tr.mean(axis=1).astype(np.float32), tr.std(axis=1).astype(np.float32)

    x_phys = torch.from_numpy(np.asarray(u[:, a.frame])[None].copy())
    x_norm = ((x_phys - torch.from_numpy(mn)) / torch.from_numpy(mx - mn)).float().to(dev)
    n_vox, n_obs = G ** 3, int(round(a.coverage * G ** 3))
    g = torch.Generator().manual_seed(1000 + vi)
    mask = torch.zeros(1, 4, n_vox)
    idx, fid = [], []
    for c in (0, 2):                                           # strict variant: own 1 % per observed field
        sel = torch.randperm(n_vox, generator=g)[:n_obs]
        mask[0, c, sel] = 1
        idx.append(sel.numpy()); fid.append(np.full(sel.numel(), c))
    mask = mask.reshape(1, 4, G, G, G).to(dev)
    cond = torch.cat([x_norm * mask, mask], dim=1)
    ens = []
    with torch.no_grad():
        for k in range(a.K):
            torch.manual_seed(9000 + 97 * vi + k)               # gen4turb_eval.py seeding
            ens.append(model.sample(cond.float())[0].float().cpu())
    ens = torch.stack(ens)
    ens_p = ens * torch.from_numpy(mx - mn)[0].float() + torch.from_numpy(mn)[0].float()
    ens_p = ens_p.reshape(a.K, 4, -1).permute(0, 2, 1).numpy().astype(np.float32)   # [K, N, 4] physical
    truth = x_phys[0].reshape(4, -1).T.numpy().astype(np.float32)
    zz = np.stack(np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 3)
    s_idx, s_fid = np.concatenate(idx), np.concatenate(fid)
    rel = {f: float(np.linalg.norm((ens_p[0, :, j] - truth[:, j]) / z_std[j]) /
                    (np.linalg.norm((truth[:, j] - z_mean[j]) / z_std[j]) + 1e-12))
           for j, f in enumerate(FIELDS)}
    meta = {"method": "gen4turb", "checkpoint": str(DM / a.ckpt), "absolute_frame": a.frame,
            "snapshot_index": vi, "grid": f"{G}^3 centre crop of the 125^3 cube (upstream loader)",
            "sensor_draw": f"strict variant, own {a.coverage:.0%} per observed field on the {G}^3 grid, "
                           "torch.Generator seed 1000+val_idx (gen4turb_eval.py)",
            "cond_fields": [0, 2], "n_sensors": int(s_idx.size), "sensor_idx_sum": int(s_idx.sum()),
            "K": a.K, "units": "physical (min-max de-normalised); norm_mean/std are z-score train stats",
            "rel_l2_member0": rel}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, truth=truth, pred_sample=ens_p[0], pred_mean=ens_p.mean(0),
                        pred_std=ens_p.std(0), coords=zz.astype(np.float32) / (G - 1),
                        coords_raw=zz.astype(np.float32), sensor_indices=s_idx.astype(np.int64),
                        sensor_field_ids=s_fid.astype(np.int64), names=np.array(FIELDS),
                        norm_mean=z_mean, norm_std=z_std, meta=np.array(json.dumps(meta)))
    print(json.dumps(rel, indent=1)); print(f"[dump] wrote {a.out}")


if __name__ == "__main__":
    main()
