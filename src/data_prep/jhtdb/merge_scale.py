"""Merge the 20-cube scale campaign into one H5, with the ORIGINAL held-out
cube3 appended as the validation block.

Layout: frames [0, 1200) = cube04..cube23 (60 each, stride 80)
        frames [1200, 1250) = the existing cube3 (50 frames, stride 100)

With a block split at train_ratio = 1200/1250 = 0.96 and gap 0, training sees
only the new cubes and validation is bit-identical to the cube3 block every
published number was measured on. That is the point of the layout: the
validation loss of a scale run is then directly comparable to N29's, and the
comparison isolates training-set size from every other variable.

cube3 is spatially disjoint from cube04..cube23 by construction -- the
downloader rejected any origin overlapping the original four cubes.
"""
import glob
from pathlib import Path

import h5py
import numpy as np

ROOT = str(Path(__file__).resolve().parent)
NEW = sorted(glob.glob(f"{ROOT}/outputfiles_scale/JHU_cube*_stride80.h5"))
VAL = f"{ROOT}/outputfiles_diverse/JHU_cube3_stride100.h5"
OUT = f"{ROOT}/outputfiles_scale/JHU_20cubes_plus_cube3.h5"

assert len(NEW) == 20, f"expected 20 scale cubes, found {len(NEW)}"

with h5py.File(NEW[0], "r") as f:
    coords = f["coordinates"][:]
    n_p, n_f = f["fields"].shape[2], f["fields"].shape[5]
with h5py.File(VAL, "r") as f:
    n_val = f["fields"].shape[1]
    assert f["fields"].shape[2] == n_p, "validation cube has a different point count"

n_train = sum(h5py.File(p, "r")["fields"].shape[1] for p in NEW)
total = n_train + n_val
print(f"train frames {n_train} + val frames {n_val} = {total}; "
      f"train_ratio = {n_train / total:.6f}", flush=True)

with h5py.File(OUT, "w") as out:
    out.create_dataset("coordinates", data=coords)
    dset = out.create_dataset("fields", shape=(1, total, n_p, 1, 1, n_f),
                              dtype=np.float32, chunks=(1, 1, n_p, 1, 1, n_f))
    out.create_dataset("time", data=np.arange(total, dtype=np.float32))
    out.attrs["source"] = "JHU isotropic1024coarse; 20 disjoint 125^3 cubes (stride 80) + original cube3 (stride 100)"
    out.attrs["layout"] = f"frames [0,{n_train})=cube04..cube23 ; [{n_train},{total})=cube3 (validation)"
    out.attrs["train_ratio"] = n_train / total
    t = 0
    for path in NEW:
        with h5py.File(path, "r") as f:
            k = f["fields"].shape[1]
            for i in range(k):
                dset[0, t + i] = f["fields"][0, i]
            t += k
        print(f"merged {path.split('/')[-1]} -> frames up to {t}", flush=True)
    with h5py.File(VAL, "r") as f:
        for i in range(n_val):
            dset[0, t + i] = f["fields"][0, i]
        t += n_val
    print(f"appended validation cube3 -> {t} total frames", flush=True)
print("wrote", OUT, flush=True)
