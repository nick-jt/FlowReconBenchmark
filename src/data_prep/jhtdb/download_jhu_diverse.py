"""Download temporally-strided, multi-cube JHTDB isotropic1024coarse data.

Fixes the structural problems of the original dataset (617 consecutive
frames of one cube): 4 spatially disjoint 125^3 cubes x 50 frames at
stride 100 (dt = 0.2 flow units between kept frames; r(dt=100 frames)
was measured at ~0.67, and cubes are spatially independent).

Reuses the query machinery of test_jht.py. Sequential queries only
(JHTDB rate courtesy). Output: outputfiles_diverse/cube{n}/step_*.h5
in the same per-frame format as the original pipeline, then consolidate
with consolidate_diverse() into one H5 per cube in the
TurbulentCombustionH5Dataset layout.
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("JHTDB_OUTDIR", "./outputfiles_diverse")
from test_jht import (  # noqa: E402
    query_variable_chunked, make_periodic_subcube_points, CUBE, GRID_FULL, L,
)
from givernylocal.turbulence_dataset import turb_dataset  # noqa: E402

OUTROOT = Path(__file__).parent / "outputfiles_diverse"
TOKEN = os.environ["JHTDB_TOKEN"]

DT_STORED = 0.002          # isotropic1024coarse frame spacing (flow units)
TIME_STRIDE = 100          # kept-frame stride
N_FRAMES = 50
TIME_INDICES = np.arange(0, TIME_STRIDE * N_FRAMES, TIME_STRIDE)

# Original cutout was at (228, 51, 563); these four are well separated
# from it and from each other (periodic domain, 1024^3).
CUBE_ORIGINS = [(0, 0, 0), (512, 256, 128), (256, 700, 900), (800, 400, 300)]


def main() -> None:
    ds = turb_dataset(dataset_title="isotropic1024coarse",
                      output_path=str(OUTROOT), auth_token=TOKEN)
    for ci, (i0, j0, k0) in enumerate(CUBE_ORIGINS):
        cube_dir = OUTROOT / f"cube{ci}"
        cube_dir.mkdir(parents=True, exist_ok=True)
        coords = make_periodic_subcube_points(i0, j0, k0, CUBE, CUBE, CUBE,
                                              GRID_FULL)
        np.save(cube_dir / "coords.npy", coords)
        for ti in TIME_INDICES:
            out = cube_dir / f"step_{ti:05d}.h5"
            if out.exists():
                continue
            t = float(ti) * DT_STORED
            vel = query_variable_chunked(ds, "velocity", t, i0, j0, k0,
                                         cube=CUBE, chunk_i=63)
            pre = query_variable_chunked(ds, "pressure", t, i0, j0, k0,
                                         cube=CUBE, chunk_i=63)
            with h5py.File(out, "w") as f:
                f.create_dataset("velocity", data=vel)
                f.create_dataset("pressure", data=pre)
                f.create_dataset("time", data=np.array([t], dtype=np.float32))
                f.attrs.update({"time_index": int(ti), "cube_origin": (i0, j0, k0)})
            print(f"[cube{ci}] wrote {out.name}", flush=True)


def consolidate_diverse() -> None:
    """Merge per-frame files into one H5 per cube (dataset layout)."""
    for cube_dir in sorted(OUTROOT.glob("cube*")):
        steps = sorted(cube_dir.glob("step_*.h5"))
        if not steps:
            continue
        coords = np.load(cube_dir / "coords.npy")
        n_p = coords.shape[0]
        out = OUTROOT / f"JHU_{cube_dir.name}_stride{TIME_STRIDE}.h5"
        with h5py.File(out, "w") as f:
            f.create_dataset("coordinates", data=coords[:, None, None, :])
            dset = f.create_dataset("fields",
                                    shape=(1, len(steps), n_p, 1, 1, 4),
                                    dtype=np.float32)
            times = []
            for si, sp in enumerate(steps):
                with h5py.File(sp, "r") as g:
                    dset[0, si, :, 0, 0, :3] = g["velocity"][:]
                    dset[0, si, :, 0, 0, 3] = g["pressure"][:, 0]
                    times.append(float(g["time"][0]))
            f.create_dataset("time", data=np.array(times, dtype=np.float32))
            f.create_dataset("field_names",
                             data=np.array([b"U", b"V", b"W", b"p"]))
            f.attrs.update({"cube_size": CUBE, "time_stride": TIME_STRIDE,
                            "dataset_title": "isotropic1024coarse"})
        print("consolidated ->", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "consolidate":
        consolidate_diverse()
    else:
        main()
