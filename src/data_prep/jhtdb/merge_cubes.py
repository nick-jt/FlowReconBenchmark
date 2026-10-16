"""Concatenate the four diverse JHU cubes along the time axis into one H5.

Frames 0-49 = cube0, 50-99 = cube1, 100-149 = cube2, 150-199 = cube3.
All cubes are 125^3 crops with identical relative grids, so cube0's
coordinates serve for all (the dataset normalizes coords to [0,1] anyway).
With train_ratio=0.75 and a block split (gap 0), cubes 0-2 train and the
spatially disjoint cube3 validates: a true cross-domain protocol.
"""
from pathlib import Path

import h5py
import numpy as np

ROOT = str(Path(__file__).resolve().parent / "outputfiles_diverse")
CUBES = [f"{ROOT}/JHU_cube{i}_stride100.h5" for i in range(4)]
OUT = f"{ROOT}/JHU_4cubes_stride100.h5"

with h5py.File(CUBES[0], "r") as f0:
    coords = f0["coordinates"][:]
    n_t, n_p, n_f = f0["fields"].shape[1], f0["fields"].shape[2], f0["fields"].shape[5]

with h5py.File(OUT, "w") as out:
    out.create_dataset("coordinates", data=coords)
    dset = out.create_dataset("fields", shape=(1, 4 * n_t, n_p, 1, 1, n_f),
                              dtype=np.float32)
    out.create_dataset("time", data=np.arange(4 * n_t, dtype=np.float32))
    out.attrs["source"] = "JHU isotropic1024coarse, 4 disjoint 125^3 cubes, stride 100"
    out.attrs["layout"] = "frames [0,50)=cube0 [50,100)=cube1 [100,150)=cube2 [150,200)=cube3"
    for ci, path in enumerate(CUBES):
        with h5py.File(path, "r") as f:
            for t in range(n_t):
                dset[0, ci * n_t + t] = f["fields"][0, t]
        print(f"cube{ci} merged", flush=True)
print("wrote", OUT, flush=True)
