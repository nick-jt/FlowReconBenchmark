"""Spectral ceiling of the JHTDB latent-FM autoencoder.

Encodes and decodes true fields (no sensors, no flow matching) and reports the
band energy relative to the DNS and the band correlation with the DNS that
survive, pooled over the observed channels U_x and U_z, with the Hann-windowed
estimator and bands of fig_spectra_stats.py, and the relative L2 error of each
channel (U_x, U_y, U_z, p). Frames: 153 (the frame of the
spectra figure) and the 25 TEST frames of the held-out cube (even indices,
absolute frames 150, 152, ..., 198).

Run on a GPU node:
  python -m spectra.ae_ceiling_jhu --ae <latent_fm Stage1 run>/best.pt
Writes outputs/paper/figures/ae_ceiling_jhu.json.
"""
import argparse
import json

import h5py
import numpy as np
import torch

from paths import DATA_DIR, PAPER_DIR
from methods.common.config import safe_torch_load
from methods.common.helpers_baseline import grid3d_to_pointcloud, pointcloud_to_grid3d
from methods.latent_fm.model import ConvAE3D
from methods.dmfgen.spectral_utils import band_corr, band_ratio, shell_spectrum

N = 125
BANDS = {"inertial": (8, 31), "dissip": (32, 50)}
OBS = (0, 2)                        # U_x, U_z observed
FIGURE_FRAME = 153
TEST_FRAMES = list(range(150, 200, 2))


def load_ae(path, device):
    """Stage-1 autoencoder and the standardization it was trained with."""
    ck = safe_torch_load(path, map_location="cpu")
    sd = ck["model"]
    # num_res_blocks from the state-dict keys, as in the latent-FM adapter
    nrb = next(i - 1 for i in range(1, 20) if f"encoder.{i}.weight" in sd)
    ae = ConvAE3D(4, base_ch=int(ck["ae_base_ch"]), latent_ch=int(ck["ae_latent_ch"]),
                  n_levels=int(ck["ae_n_levels"]), Num_z=N, Num_y=N, Num_x=N, num_res_blocks=nrb)
    ae.load_state_dict(sd)
    return ae.to(device).eval(), ck["mean"].numpy().ravel(), ck["std"].numpy().ravel()


def round_trip(ae, mean, std, truth, device):
    """(N^3, 4) physical field -> its autoencoder reconstruction, physical units."""
    x = torch.from_numpy((truth - mean) / std)[None].to(device)
    with torch.no_grad():
        g = ae.decode(ae.encode(pointcloud_to_grid3d(x, N, N, N)))
    return grid3d_to_pointcloud(g, N, N, N)[0].cpu().numpy() * std + mean


def band_metrics(rec, truth):
    cubes = lambda a: [a[:, c].reshape(N, N, N) for c in OBS]
    s_rec = sum(shell_spectrum(c) for c in cubes(rec))
    s_true = sum(shell_spectrum(c) for c in cubes(truth))
    corr = band_corr(cubes(rec), cubes(truth), BANDS)
    out = {b: {"energy": band_ratio(s_rec, s_true, *r), "corr": corr[b]} for b, r in BANDS.items()}
    out["rel_l2"] = (np.linalg.norm(rec - truth, axis=0) / np.linalg.norm(truth, axis=0)).tolist()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ae", required=True, help="latent-FM Stage1 checkpoint (best.pt)")
    args = p.parse_args()

    device = "cuda"
    ae, mean, std = load_ae(args.ae, device)
    fields = h5py.File(DATA_DIR / "JHU_4cubes_stride100.h5", "r")["fields"]   # (1, frame, N^3, 1, 1, 4)
    res = {}
    for fr in [FIGURE_FRAME] + TEST_FRAMES:
        truth = fields[0, fr, :, 0, 0, :]
        res[fr] = band_metrics(round_trip(ae, mean, std, truth, device), truth)
        print(fr, json.dumps(res[fr]), flush=True)

    test = {b: {m: [min(v), float(np.mean(v)), max(v)]
                for m in ("energy", "corr") for v in [[res[f][b][m] for f in TEST_FRAMES]]}
            for b in BANDS}
    summary = {"ae_checkpoint": args.ae, "bands": BANDS, "observed_channels": ["Ux", "Uz"],
               "figure_frame": res[FIGURE_FRAME], "test_frames": TEST_FRAMES,
               "test_min_mean_max": test, "per_frame": res}
    out = PAPER_DIR / "figures" / "ae_ceiling_jhu.json"
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(test, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
