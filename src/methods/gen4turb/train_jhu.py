"""Train Gen4Turb on the JHTDB crop with a shared velocity range and octahedral augmentation.

This is the upstream Gen4Turbulence script 3_flow_reconstruction/dm/train_model_uxuz.py
(upstream 8af0a7b plus the U_x/U_z observation mask), as used for the reported anneal arm:
upstream Unet and ElucidatedDiffusion (dim 16, mults 1-2-4-8), batch 6, Adam at 1e-4 with
the cosine decay stepped every batch (4200 epochs, 105k steps), the U_x/U_z mask with
per-channel coverage U(0.001, 0.01), the 120^3 crop of training frames 0-149, and a
checkpoint every 10 epochs. It differs from that arm in four places:

  1. U_x, U_y and U_z share one symmetric min-max range, +-max|u| over the three velocity
     channels of the training frames (the range of the augmented training set, so a sign
     flip stays inside it); p keeps its own. The upstream per-channel range gave the
     channels different amplitudes and offsets in the network's units.
  2. Octahedral augmentation (augment_symmetry.octahedral_augment, the 48 signed axis
     permutations the other learned JHU methods train with), one draw per sample. Without
     it the trained network reconstructs whatever field sits in the U_z slot about twice
     as badly as the U_x slot.
  3. Seed 2027 (all earlier arms used 23).
  4. No validation pass: upstream picked best_model.pt on frames 150-167, which belong to
     the held-out cube. Checkpoints are chosen on the TUNE half instead.

The run directory <dm>/<run> receives the checkpoints plus MIN_u.npy, MAX_u.npy and
Par.pkl, which eval_jhu.load_norm reads back.

    FRB_GEN4TURB_ROOT=/path/to/Gen4Turbulence python -m methods.gen4turb.train_jhu \
        --run models_uxuz_octa_s2027
"""
import argparse
import datetime
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from methods.dmfgen.augment_symmetry import octahedral_augment
from paths import site_path

N_TRAIN = 150     # frames 0-149 = training cubes 0-2
BATCH = 6


def shared_velocity_range(u):
    """Per-channel (lo, hi) of u [T, C, X, Y, Z], with one symmetric range for channels 0-2."""
    lo = u.min(axis=(0, 2, 3, 4)).astype(np.float32)
    hi = u.max(axis=(0, 2, 3, 4)).astype(np.float32)
    m = max(-lo[:3].min(), hi[:3].max())
    lo[:3], hi[:3] = -m, m
    return lo, hi


def augment(x, rng):
    """One octahedral draw per sample of x [bs, C, X, Y, Z] in min-max units.

    With the symmetric velocity range u = 0 sits at 0.5, so the sign flips act on x - 0.5;
    p (channel 3) is only moved spatially. Returns the batch and how many samples changed.
    """
    bs, nf, *grid = x.shape
    out = torch.empty_like(x)
    moved = 0
    for b in range(bs):
        f = x[b].permute(1, 2, 3, 0).reshape(-1, nf) - 0.5
        g = octahedral_augment(f, grid, vel_idx=(0, 1, 2), rng=rng)
        moved += int(not torch.equal(g, f))
        out[b] = (g + 0.5).reshape(*grid, nf).permute(3, 0, 1, 2)
    return out, moved


def sr_prep(x: torch.Tensor, y: torch.Tensor, mask_size=0.125):
    """Upstream train_model_uxuz.py sr_prep, unchanged."""
    bs, nf, nx, ny, nz = x.shape
    device, dtype = x.device, x.dtype

    # Ux/Uz-only observation model matching the cross-cube sparse-sensing
    # protocol: independent random point masks on channels 0 and 2 with
    # per-channel coverage U(0.001, 0.01); channels 1 and 3 unobserved.
    total_voxels = nx * ny * nz
    mask = torch.zeros((bs, nf, total_voxels), device=device, dtype=dtype)
    for b in range(bs):
        for c in (0, 2):
            cov = random.uniform(0.001, 0.01)
            idx = torch.randperm(total_voxels, device=device)[:int(total_voxels * cov)]
            mask[b, c, idx] = 1
    mask = mask.reshape(bs, nf, nx, ny, nz)
    x_in = torch.cat([x * mask, mask], dim=1)
    return x_in, y


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="output directory, relative to the upstream dm/ directory")
    p.add_argument("--epochs", type=int, default=4200)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=2027)
    a = p.parse_args()

    dm = site_path("gen4turb_root") / "3_flow_reconstruction" / "dm"
    sys.path.insert(0, str(dm))
    from utils.architecture import Unet
    from utils.diffusion import ElucidatedDiffusion

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda")
    out = dm / a.run
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    u = np.load(dm.parent / "data" / "u.npy", mmap_mode="r")[:, :N_TRAIN]       # [C, T, X, Y, Z]
    u = np.ascontiguousarray(u.transpose(1, 0, 2, 3, 4))                         # [T, C, X, Y, Z]
    lo, hi = shared_velocity_range(u)
    u -= lo.reshape(1, -1, 1, 1, 1)
    u /= (hi - lo).reshape(1, -1, 1, 1, 1)
    sigma_data = float(np.std(u))
    print(f"[data] {u.shape} in {time.time() - t0:.1f}s; MIN {lo} MAX {hi}; sigma_data {sigma_data:.5f}; "
          f"normalized range [{u.min():.4f}, {u.max():.4f}]", flush=True)
    np.save(out / "MIN_u.npy", lo)
    np.save(out / "MAX_u.npy", hi)
    nx, ny, nz = u.shape[-3:]
    par = {"inp_shift": lo, "inp_scale": hi - lo, "out_shift": lo, "out_scale": hi - lo,
           "nx": nx, "ny": ny, "nz": nz, "nf": 4, "num_epochs": a.epochs, "sigma_data": sigma_data,
           "channels": 4, "self_condition": True, "seed": a.seed, "augment": "octahedral",
           "velocity_range": "shared symmetric"}
    pickle.dump(par, open(out / "Par.pkl", "wb"))
    data = torch.from_numpy(u)

    net = Unet(dim=16, dim_mults=(1, 2, 4, 8), channels=4, self_condition=True,
               flash_attn=True).to(device).to(torch.float32)
    model = ElucidatedDiffusion(net, channels=4, image_size_h=nx, image_size_w=ny, image_size_d=nz,
                                sigma_data=sigma_data)
    print(f"[model] {sum(q.numel() for q in model.parameters()):,} parameters", flush=True)
    loader = DataLoader(torch.arange(N_TRAIN), batch_size=BATCH, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=a.epochs * len(loader))

    t_start = time.time()
    for epoch in range(a.epochs):
        model.train()
        t_ep = time.time()
        train_loss, moved = 0.0, 0
        for idx in loader:
            h_fidel, m = augment(data[idx].to(device), rng)
            moved += m
            l_fidel, _ = sr_prep(h_fidel, h_fidel)
            optimizer.zero_grad()
            loss = model(h_fidel, l_fidel)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(loader)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] - Epoch {epoch + 1}/{a.epochs}, Train Loss: {train_loss:.4e}, "
              f"LR: {scheduler.get_last_lr()[0]:.4e}, augmented {moved}/{N_TRAIN}, "
              f"train time: {time.time() - t_ep:.2f}", flush=True)
        if epoch != 0 and epoch % a.save_every == 0:
            torch.save(model.state_dict(), out / f"model_{epoch}.pt")
    print(f"Training finished in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
