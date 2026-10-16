"""Spectral ceiling of the Kolmogorov latent-FM autoencoder (stage 1, frozen-protocol
run DemoN60): encode -> decode the TRUE field (no sampling, no sensors) and measure the
small-scale energy it keeps, with the estimator of fig_spectra_kolmogorov.py
(unwindowed, E(k) = sum_shell |w_hat|^2 / 2k^2). Ten evenly spaced TEST-half frames,
read in physical units from the dataset and standardized with the checkpoint's stats.

    python -m spectra.ae_ceiling_kolm
writes outputs/paper/figures/ae_ceiling_kolm.json."""
import json
import h5py
import numpy as np
import torch
from paths import REPO_ROOT
from methods.latent_fm.model import ConvAE
from methods.common.helpers_baseline import pointcloud_to_grid

torch.set_num_threads(16)
RUN = REPO_ROOT / "outputs/runs/kolmogorov2d/baseline_latent_fm/Baseline_latent_fm_Stage1_DemoN60_20261003_144213"
ck = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
ae = ConvAE(n_fields=1, base_ch=ck["ae_base_ch"], latent_ch=ck["ae_latent_ch"], n_levels=ck["ae_n_levels"],
            Num_y=ck["Num_y"], Num_x=ck["Num_x"], num_res_blocks=ck["ae_num_res_blocks"])
ae.load_state_dict(ck["model"]); ae.eval()
mean, std = ck["mean"].float().reshape(-1), ck["std"].float().reshape(-1)

S = 256
kk = np.fft.fftfreq(S) * S
KX, KY = np.meshgrid(kk, kk, indexing="ij")
km = np.sqrt(KX ** 2 + KY ** 2); sh = np.rint(km).astype(int)
wE = np.where(km > 0, 1.0 / (2 * np.maximum(km, 1e-12) ** 2), 0.0)
BANDS = {"k1-8": (1, 8), "k8-31": (8, 31), "k32-128": (32, 128)}


def bands(s, t):
    Fs, Ft = np.fft.fft2(s), np.fft.fft2(t)
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (sh >= lo) & (sh <= hi)
        Es, Et = np.sum(wE[m] * np.abs(Fs[m]) ** 2), np.sum(wE[m] * np.abs(Ft[m]) ** 2)
        out[b] = (float(Es / Et), float(np.sum(wE[m] * np.real(Fs[m] * np.conj(Ft[m]))) / np.sqrt(Es * Et)))
    return out


f = h5py.File(REPO_ROOT / "data/kolmogorov2d/Kolmogorov2D_shu_stride4.h5", "r")
held = np.arange(2560, 3200)[0::2][::32][:10]            # includes 2816, the gallery frame of Fig. 4
per = {}
for fr in held:
    pts = torch.tensor(f["fields"][0, fr, :, 0, 0, :1], dtype=torch.float32)   # physical units
    x = pointcloud_to_grid(((pts - mean) / std)[None], S, S)
    with torch.no_grad():
        xh = ae._crop(ae.decode(ae.encode(x)))
    per[int(fr)] = {"rel_l2": float(np.linalg.norm(xh.numpy() - x.numpy()) / np.linalg.norm(x.numpy())),
                    **{b: v for b, v in bands(xh[0, 0].numpy(), x[0, 0].numpy()).items()}}
rows = np.array([[v["rel_l2"], *v["k32-128"]] for v in per.values()])
res = {"frames": held.tolist(), "per_frame": per,
       "range": {"rel_l2": [rows[:, 0].min(), rows[:, 0].max()], "small_energy": [rows[:, 1].min(), rows[:, 1].max()],
                 "small_corr": [rows[:, 2].min(), rows[:, 2].max()]}}
print(json.dumps(res["range"], indent=1)); print("frame 2816:", per[2816])
json.dump(res, open(REPO_ROOT / "outputs/paper/figures/ae_ceiling_kolm.json", "w"), indent=1)
