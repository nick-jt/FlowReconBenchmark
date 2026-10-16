"""Scale campaign: 20 additional disjoint 125^3 cubes, 60 frames each.

Brings the corpus to ~1,400 decorrelated snapshots (existing 4 cubes x 50 +
new 20 x 60), the scale at which the memorization-to-generalization
transition is expected (Kadkhodaie et al.). Politeness toward JHTDB, per the
same rules as the first campaign: strictly sequential sub-2M-point queries,
skip-existing (restarts are free), retry loop with long sleeps around
transient failures. Cube origins sit on the 128-stride lattice and are
checked for overlap against the four existing cubes; the held-out validation
cube (cube3) stays untouched so every earlier cross-cube number remains
comparable.

Usage:  python download_jhu_scale.py            # download (resumable)
        python download_jhu_scale.py consolidate
"""
import os, sys, time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("JHTDB_OUTDIR", "./outputfiles_scale")
from test_jht import (  # noqa: E402
    query_variable_chunked, make_periodic_subcube_points, CUBE, GRID_FULL,
)
from givernylocal.turbulence_dataset import turb_dataset  # noqa: E402

OUTROOT = Path(__file__).parent / "outputfiles_scale"
TOKEN = os.environ["JHTDB_TOKEN"]
DT_STORED = 0.002
TIME_STRIDE = 80
N_FRAMES = 60
TIME_INDICES = np.arange(0, TIME_STRIDE * N_FRAMES, TIME_STRIDE)

EXISTING = [(0, 0, 0), (512, 256, 128), (256, 700, 900), (800, 400, 300)]

def overlaps(a, b, size=CUBE, dom=GRID_FULL):
    for x, y in zip(a, b):
        d = abs(x - y)
        d = min(d, dom - d)                     # periodic distance
        if d >= size:
            return False
    return True

# 128-stride lattice positions, keeping only those disjoint from all
# existing cubes; deterministic order for reproducibility.
lattice = [(i * 128, j * 128, k * 128)
           for i in range(8) for j in range(8) for k in range(8)]
rng = np.random.default_rng(2027)
rng.shuffle(lattice)
NEW_ORIGINS = []
for p in lattice:
    if any(overlaps(p, q) for q in EXISTING + NEW_ORIGINS):
        continue
    NEW_ORIGINS.append(p)
    if len(NEW_ORIGINS) == 20:
        break
assert len(NEW_ORIGINS) == 20, len(NEW_ORIGINS)


def main():
    ds = turb_dataset(dataset_title="isotropic1024coarse",
                      output_path=str(OUTROOT), auth_token=TOKEN)
    for ci, (i0, j0, k0) in enumerate(NEW_ORIGINS):
        cube_dir = OUTROOT / f"cube{ci + 4:02d}"
        cube_dir.mkdir(parents=True, exist_ok=True)
        coords = make_periodic_subcube_points(i0, j0, k0, CUBE, CUBE, CUBE,
                                              GRID_FULL)
        np.save(cube_dir / "coords.npy", coords)
        for ti in TIME_INDICES:
            out = cube_dir / f"step_{ti:05d}.h5"
            if out.exists():
                continue
            t = float(ti) * DT_STORED
            for attempt in range(40):
                try:
                    vel = query_variable_chunked(ds, "velocity", t, i0, j0,
                                                 k0, cube=CUBE, chunk_i=63)
                    pre = query_variable_chunked(ds, "pressure", t, i0, j0,
                                                 k0, cube=CUBE, chunk_i=63)
                    break
                except Exception as e:
                    print(f"[cube{ci+4:02d} t={ti}] retry {attempt+1}: {e}",
                          flush=True)
                    time.sleep(300)
            else:
                raise RuntimeError(f"cube{ci+4} t={ti}: exhausted retries")
            with h5py.File(out, "w") as f:
                f.create_dataset("velocity", data=vel)
                f.create_dataset("pressure", data=pre)
                f.create_dataset("time", data=np.array([t], dtype=np.float32))
                f.attrs.update({"time_index": int(ti),
                                "cube_origin": (i0, j0, k0)})
            print(f"[cube{ci+4:02d}] wrote {out.name}", flush=True)
    print("DOWNLOAD_COMPLETE", flush=True)


def consolidate():
    for ci in range(4, 24):
        cube_dir = OUTROOT / f"cube{ci:02d}"
        files = sorted(cube_dir.glob("step_*.h5"))
        if len(files) != N_FRAMES:
            print(f"cube{ci:02d}: {len(files)}/{N_FRAMES} frames, skipping")
            continue
        out = OUTROOT / f"JHU_cube{ci:02d}_stride{TIME_STRIDE}.h5"
        if out.exists():
            continue
        coords = np.load(cube_dir / "coords.npy")
        with h5py.File(out, "w") as f:
            f.create_dataset("coordinates", data=coords[:, None, None, :])
            dset = f.create_dataset(
                "fields", shape=(1, N_FRAMES, coords.shape[0], 1, 1, 4),
                dtype=np.float32)
            times = np.zeros(N_FRAMES, dtype=np.float32)
            for fi, fp in enumerate(files):
                with h5py.File(fp, "r") as g:
                    dset[0, fi, :, 0, 0, :3] = g["velocity"][:]
                    dset[0, fi, :, 0, 0, 3] = g["pressure"][:].squeeze()
                    times[fi] = g["time"][0]
            f.create_dataset("time", data=times)
            f.attrs.update({"cube_size": CUBE, "time_stride": TIME_STRIDE})
        print("consolidated ->", out, flush=True)
    print("CONSOLIDATE_COMPLETE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "consolidate":
        consolidate()
    else:
        main()
