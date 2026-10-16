"""Convert the OpenFOAM 2D cylinder runs to the canonical benchmark H5 layout.

Reads each Re case under --runs (openfoam/cylinder2d/submit_all.sh output),
takes the phase-2 sampling window t in (--t-min, --t-max], and writes:

1) Native mesh export  Cylinder2D_mesh.h5
   coordinates (N,1,1,3)  cell centres (z column constant 0.5)
   fields      (1,T,N,1,1,3)  [Ux, Uy, p]
   time        (T,) global frame index
   surface_indices (dataset): cells adjacent to the cylinder wall, for the
   surface-tap sensor protocol.
2) Grid export         Cylinder2D_grid.h5  (for grid-locked baselines)
   Uniform ROI x [-3,17], y [-5,5] at 400x200 (dx=0.05), linear interpolation
   from cell centres (Delaunay built once); points inside the cylinder are 0
   with body_mask recording validity. Same frame ordering as the mesh export.

Frame ordering = cross-Re holdout: train Re blocks first (default 60,100,150,
200) then held-out Re (80,250), so JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 with
train_ratio = n_train/(n_train+n_test) reproduces the protocol. Train-only
normalization stats land in cylinder2d_manifest.json.
"""
from paths import DATA_DIR

import argparse
import glob
import json
import os
import re

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"h5py required: {exc}")

DEFAULT_RUNS = f"{DATA_DIR}/cylinder2d/runs"
DEFAULT_OUT = f"{DATA_DIR}/cylinder2d"

_NONUNIFORM_RE = re.compile(
    r"internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(", re.S
)
_UNIFORM_RE = re.compile(r"internalField\s+uniform\s+(\(([^)]*)\)|[-0-9eE.+]+)\s*;")


def read_of_internal_field(path: str, n_cells: int | None = None) -> np.ndarray:
    """Parse an OpenFOAM ASCII volScalarField/volVectorField internalField."""
    with open(path) as fh:
        txt = fh.read()
    m = _NONUNIFORM_RE.search(txt)
    if m:
        kind, count = m.group(1), int(m.group(2))
        start = m.end()
        end = txt.index(")\n;", start)
        body = txt[start:end]
        if kind == "vector":
            vals = np.fromstring(body.replace("(", " ").replace(")", " "), sep=" ")
            arr = vals.reshape(count, 3)
        else:
            arr = np.fromstring(body, sep=" ")
            assert arr.size == count, (path, arr.size, count)
            arr = arr.reshape(count)
        return arr.astype(np.float32)
    m = _UNIFORM_RE.search(txt)
    if m:
        if n_cells is None:
            raise ValueError(f"uniform field in {path} needs n_cells")
        if m.group(2) is not None:
            vec = np.fromstring(m.group(2), sep=" ").astype(np.float32)
            return np.tile(vec, (n_cells, 1))
        return np.full(n_cells, float(m.group(1)), dtype=np.float32)
    raise ValueError(f"could not parse internalField in {path}")


def time_dirs(case: str, t_min: float, t_max: float) -> list[str]:
    out = []
    for d in os.listdir(case):
        try:
            t = float(d)
        except ValueError:
            continue
        if t_min < t <= t_max and os.path.isdir(os.path.join(case, d)):
            out.append(d)
    return sorted(out, key=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--train-res", default="60,100,150,200")
    ap.add_argument("--test-res", default="80,250")
    ap.add_argument("--t-min", type=float, default=120.0)
    ap.add_argument("--t-max", type=float, default=270.0)
    ap.add_argument("--grid-nx", type=int, default=400)
    ap.add_argument("--grid-ny", type=int, default=200)
    ap.add_argument("--roi", default="-3,17,-5,5")
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    train_res = [int(r) for r in args.train_res.split(",")]
    test_res = [int(r) for r in args.test_res.split(",")]
    re_sequence = train_res + test_res

    # Cell centres from the first case (identical mesh across Re).
    first_case = os.path.join(args.runs, f"Re{re_sequence[0]}")
    c_path = os.path.join(first_case, "0", "C")
    if not os.path.exists(c_path):
        raise SystemExit(f"cell centres missing: {c_path} (postProcess -func writeCellCentres)")
    centres = read_of_internal_field(c_path)  # (N, 3)
    n_pts = centres.shape[0]
    coords = np.empty((n_pts, 3), dtype=np.float32)
    coords[:, 0] = centres[:, 0]
    coords[:, 1] = centres[:, 1]
    coords[:, 2] = 0.5

    r = np.hypot(coords[:, 0], coords[:, 1])
    surface_indices = np.where(r < 0.53)[0]  # within 0.03 D of the wall (r0=0.5): two O-grid layers and part of a third

    per_case = {}
    for Re in re_sequence:
        case = os.path.join(args.runs, f"Re{Re}")
        dirs = time_dirs(case, args.t_min, args.t_max)
        if not dirs:
            raise SystemExit(f"no sampling-window time dirs in {case}")
        per_case[Re] = dirs
    counts = {Re: len(d) for Re, d in per_case.items()}
    print(f"[convert] frames per Re: {counts}")
    total_t = sum(counts.values())
    n_train_frames = sum(counts[Re] for Re in train_res)

    os.makedirs(args.out_dir, exist_ok=True)
    mesh_h5 = os.path.join(args.out_dir, "Cylinder2D_mesh.h5")

    frame_re = np.empty(total_t, dtype=np.int32)
    with h5py.File(mesh_h5, "w") as f:
        f.create_dataset("coordinates", data=coords.reshape(n_pts, 1, 1, 3))
        fields = f.create_dataset(
            "fields", shape=(1, total_t, n_pts, 1, 1, 3), dtype="float32",
            chunks=(1, 1, n_pts, 1, 1, 3),
        )
        f.create_dataset("time", data=np.arange(total_t, dtype=np.float32))
        f.create_dataset("surface_indices", data=surface_indices)
        t0 = 0
        for Re in re_sequence:
            case = os.path.join(args.runs, f"Re{Re}")
            for d in per_case[Re]:
                u = read_of_internal_field(os.path.join(case, d, "U"), n_pts)
                p = read_of_internal_field(os.path.join(case, d, "p"), n_pts)
                fields[0, t0, :, 0, 0, 0] = u[:, 0]
                fields[0, t0, :, 0, 0, 1] = u[:, 1]
                fields[0, t0, :, 0, 0, 2] = p
                frame_re[t0] = Re
                t0 += 1
        f.create_dataset("frame_re", data=frame_re)
        f.attrs["field_names"] = ["Ux", "Uy", "p"]
        f.attrs["domain"] = "cylinder D=1 at origin; x[-8,25] y[-8,8]; nondim U=D=1"
        f.attrs["train_res"] = train_res
        f.attrs["test_res"] = test_res
        f.attrs["layout"] = (
            f"frames [0,{n_train_frames}) = train Re {train_res}; "
            f"[{n_train_frames},{total_t}) = held-out Re {test_res}; "
            f"block split gap 0, train_ratio {n_train_frames / total_t:.4f}"
        )

    # Train-only stats (streamed).
    with h5py.File(mesh_h5, "r") as f:
        s = np.zeros(3)
        s2 = np.zeros(3)
        cnt = 0
        for t in range(n_train_frames):
            fr = f["fields"][0, t, :, 0, 0, :].astype(np.float64)
            s += fr.sum(0)
            s2 += (fr**2).sum(0)
            cnt += fr.shape[0]
    mean = s / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean**2, 1e-12))

    manifest = {
        "mesh_h5": mesh_h5,
        "n_points": int(n_pts),
        "n_surface_points": int(len(surface_indices)),
        "frames_per_re": {str(k): int(v) for k, v in counts.items()},
        "train_res": train_res,
        "test_res": test_res,
        "n_train_frames": int(n_train_frames),
        "n_total_frames": int(total_t),
        "train_ratio": n_train_frames / total_t,
        "field_names": ["Ux", "Uy", "p"],
        "train_stats": {
            name: {"mean": float(mean[i]), "std": float(std[i])}
            for i, name in enumerate(["Ux", "Uy", "p"])
        },
    }

    if not args.skip_grid:
        from scipy.spatial import Delaunay
        from scipy.interpolate import LinearNDInterpolator

        x0, x1, y0, y1 = (float(v) for v in args.roi.split(","))
        gx = np.linspace(x0, x1, args.grid_nx, dtype=np.float32)
        gy = np.linspace(y0, y1, args.grid_ny, dtype=np.float32)
        gyy, gxx = np.meshgrid(gy, gx, indexing="ij")  # row-major (y, x)
        gpts = np.stack([gxx.ravel(), gyy.ravel()], axis=1)
        body = np.hypot(gpts[:, 0], gpts[:, 1]) <= 0.5
        tri = Delaunay(coords[:, :2].astype(np.float64))
        grid_h5 = os.path.join(args.out_dir, "Cylinder2D_grid.h5")
        gcoords = np.empty((gpts.shape[0], 3), dtype=np.float32)
        gcoords[:, :2] = gpts
        gcoords[:, 2] = 0.5
        with h5py.File(mesh_h5, "r") as fin, h5py.File(grid_h5, "w") as f:
            f.create_dataset("coordinates", data=gcoords.reshape(-1, 1, 1, 3))
            gfields = f.create_dataset(
                "fields", shape=(1, total_t, gpts.shape[0], 1, 1, 3), dtype="float32",
                chunks=(1, 1, gpts.shape[0], 1, 1, 3),
            )
            f.create_dataset("time", data=np.arange(total_t, dtype=np.float32))
            f.create_dataset("body_mask", data=(~body))  # True = fluid point
            f.create_dataset("frame_re", data=frame_re)
            for t in range(total_t):
                fr = fin["fields"][0, t, :, 0, 0, :]
                interp = LinearNDInterpolator(tri, fr, fill_value=0.0)
                vals = interp(gpts).astype(np.float32)
                vals[body] = 0.0
                gfields[0, t, :, 0, 0, :] = vals
                if t % 200 == 0:
                    print(f"[grid] frame {t}/{total_t}")
            f.attrs["field_names"] = ["Ux", "Uy", "p"]
            f.attrs["roi"] = [x0, x1, y0, y1]
            f.attrs["grid_shape"] = [args.grid_ny, args.grid_nx]
            f.attrs["layout"] = (
                f"same frame order as {os.path.basename(mesh_h5)}; body interior zeroed"
            )
        manifest["grid_h5"] = grid_h5
        manifest["grid_shape"] = [args.grid_ny, args.grid_nx]
        manifest["roi"] = [x0, x1, y0, y1]

    manifest_path = os.path.join(args.out_dir, "cylinder2d_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {mesh_h5} ({total_t} frames x {n_pts} pts) and {manifest_path}")

    if args.verify:
        with h5py.File(mesh_h5, "r") as f:
            c = f["coordinates"][:, 0, 0, :]
            assert np.allclose(c[:, 2], 0.5)
            assert (np.hypot(c[:, 0], c[:, 1]) > 0.499).all(), "points inside cylinder?"
            fr = f["fields"][0, 0, :, 0, 0, :]
            assert np.isfinite(fr).all()
            si = f["surface_indices"][:]
            assert 100 <= len(si) <= 400, f"surface ring size {len(si)} unexpected"
            u_inf = fr[np.argmax(np.abs(c[:, 0] + 7.9)), 0]
            print(f"verify: PASS ({len(si)} surface cells; inlet-adjacent Ux ~ {u_inf:.3f})")


if __name__ == "__main__":
    main()
