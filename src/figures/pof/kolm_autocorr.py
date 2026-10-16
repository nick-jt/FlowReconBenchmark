"""Temporal autocorrelation of the raw Kolmogorov data.

If adjacent raw frames are already decorrelated, then a 'same trajectory,
unseen frame' split is NOT leaky for this dataset — which would explain the
near-null protocol-B result and sharpen what leakage actually requires.
"""
from paths import DATA_DIR, THIRD_PARTY_DIR
import numpy as np

import os
# Host-portable: the origin path is tried last. $KOLM_SHU_NPY wins, then the
# Delta dataset location, so the script runs wherever the data was transferred.
_CANDS = [os.environ.get("KOLM_SHU_NPY"),
          f"{DATA_DIR}/kolmogorov_shu.npy",
          f"{THIRD_PARTY_DIR}/"
          "sparse-reconstruction/data/kolmogorov_shu.npy"]
_PATH = next((c for c in _CANDS if c and os.path.exists(c)), None)
if _PATH is None:
    raise SystemExit(f"kolmogorov_shu.npy not found; tried {[c for c in _CANDS if c]}")
A = np.load(_PATH, mmap_mode="r")
rng = np.random.default_rng(0)
trajs = rng.choice(A.shape[0], 6, replace=False)
lags = [1, 2, 4, 8, 16, 32]
out = {l: [] for l in lags}
for t in trajs:
    blk = np.asarray(A[t, :160], dtype=np.float64)          # 160 frames
    blk -= blk.mean(axis=(1, 2), keepdims=True)
    nrm = np.sqrt((blk ** 2).sum(axis=(1, 2)))
    for l in lags:
        a, b = blk[:-l], blk[l:]
        na, nb = nrm[:-l], nrm[l:]
        out[l].append(float(np.mean((a * b).sum(axis=(1, 2)) / (na * nb))))

print("Kolmogorov raw-frame temporal correlation (6 trajectories, 160 frames each)")
for l in lags:
    print(f"  lag {l:3d} raw frames : r = {np.mean(out[l]):+.4f}")
print()
print("For reference, JHU DNS: r(lag=1) = 1.00, r(lag=100) = 0.67")
print("Our H5 uses stride 4; protocol-B probe frames sit 2 raw frames from a train frame.")
