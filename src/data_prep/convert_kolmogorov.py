"""Convert the Shu et al. 2D Kolmogorov npy to the canonical benchmark H5 layout.

Source: baselines/sparse-reconstruction/data/kolmogorov_shu.npy, float32
(40 trajectories, 320 frames, 256, 256) vorticity on [0, 2pi)^2.

Output H5 matches the Merged_CH4COTU1P.h5 conventions consumed by
helpers.TurbulentCombustionH5Dataset and the baseline adapters:
  coordinates (N, 1, 1, 3)  -- 2D data stored 3-wide with constant z = 0.5
  fields      (1, T, N, 1, 1, F)
  time        (T,)
Point ordering is row-major (iy * Num_x + ix), the order
helpers_baseline.pointcloud_to_grid assumes.

Split design (trajectory holdout, the 2D analogue of the JHU cross-cube
protocol): trajectories are shuffled once with --split-seed, the first
--train-traj become the train block and the rest the test block; frames are
written train-block-first so JHU_SPLIT_MODE=block with gap 0 and
train_ratio = train_frames / total reproduces the protocol. Frames are
strided by --stride within each trajectory to reduce temporal correlation.
Normalization stats are computed on TRAIN frames only.
"""
from paths import DATA_DIR, THIRD_PARTY_DIR

import argparse
import json
import os

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"h5py required: {exc}")

DEFAULT_NPY = (
    f"{THIRD_PARTY_DIR}/"
    "sparse-reconstruction/data/kolmogorov_shu.npy"
)
DEFAULT_OUT = f"{DATA_DIR}/kolmogorov2d"


def build_coordinates(nx: int, ny: int) -> np.ndarray:
    xs = 2.0 * np.pi * np.arange(nx, dtype=np.float32) / nx
    ys = 2.0 * np.pi * np.arange(ny, dtype=np.float32) / ny
    coords = np.empty((ny * nx, 3), dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")  # row-major: y slowest
    coords[:, 0] = xx.ravel()
    coords[:, 1] = yy.ravel()
    coords[:, 2] = 0.5
    return coords


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default=DEFAULT_NPY)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--train-traj", type=int, default=32)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="re-open the output and check round-trip")
    args = ap.parse_args()

    data = np.load(args.npy, mmap_mode="r")
    n_traj, n_frames, ny, nx = data.shape
    if args.train_traj >= n_traj:
        raise SystemExit(f"--train-traj {args.train_traj} must be < n_traj {n_traj}")

    rng = np.random.default_rng(args.split_seed)
    order = rng.permutation(n_traj)
    train_trajs = np.sort(order[: args.train_traj])
    test_trajs = np.sort(order[args.train_traj :])

    frame_ids = np.arange(0, n_frames, args.stride)
    fpt = len(frame_ids)  # frames per trajectory after stride
    traj_sequence = np.concatenate([train_trajs, test_trajs])
    total_t = fpt * n_traj
    n_train_frames = fpt * len(train_trajs)
    n_pts = ny * nx

    os.makedirs(args.out_dir, exist_ok=True)
    out_h5 = os.path.join(args.out_dir, f"Kolmogorov2D_shu_stride{args.stride}.h5")

    coords = build_coordinates(nx, ny)

    # Train-only stats in a first streaming pass (Welford unnecessary: two-pass
    # on mmap'd data is cheap and exact).
    s = 0.0
    s2 = 0.0
    cnt = 0
    for tj in train_trajs:
        block = np.asarray(data[tj, frame_ids], dtype=np.float64)
        s += block.sum()
        s2 += (block**2).sum()
        cnt += block.size
    mean = s / cnt
    std = float(np.sqrt(max(s2 / cnt - mean**2, 1e-12)))
    mean = float(mean)

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("coordinates", data=coords.reshape(n_pts, 1, 1, 3))
        fields = f.create_dataset(
            "fields",
            shape=(1, total_t, n_pts, 1, 1, 1),
            dtype="float32",
            chunks=(1, 1, n_pts, 1, 1, 1),
        )
        time_axis = np.arange(total_t, dtype=np.float32)
        f.create_dataset("time", data=time_axis)
        t0 = 0
        for tj in traj_sequence:
            block = np.asarray(data[tj, frame_ids], dtype=np.float32)  # (fpt, ny, nx)
            fields[0, t0 : t0 + fpt, :, 0, 0, 0] = block.reshape(fpt, n_pts)
            t0 += fpt
        f.attrs["source"] = os.path.abspath(args.npy)
        f.attrs["field_names"] = ["vorticity"]
        f.attrs["grid_shape"] = [ny, nx]
        f.attrs["domain"] = "[0,2pi)^2 periodic"
        f.attrs["stride"] = args.stride
        f.attrs["frames_per_traj"] = fpt
        f.attrs["split_seed"] = args.split_seed
        f.attrs["train_trajs"] = train_trajs
        f.attrs["test_trajs"] = test_trajs
        f.attrs["layout"] = (
            f"frames [0,{n_train_frames}) = {len(train_trajs)} train trajectories, "
            f"frames [{n_train_frames},{total_t}) = {len(test_trajs)} held-out trajectories; "
            f"block split gap 0 with train_ratio {n_train_frames / total_t:.4f} reproduces the protocol"
        )

    manifest = {
        "h5": out_h5,
        "n_traj": int(n_traj),
        "frames_per_traj": int(fpt),
        "stride": int(args.stride),
        "split_seed": int(args.split_seed),
        "train_trajs": train_trajs.tolist(),
        "test_trajs": test_trajs.tolist(),
        "n_train_frames": int(n_train_frames),
        "n_total_frames": int(total_t),
        "train_ratio": n_train_frames / total_t,
        "grid": [int(ny), int(nx)],
        "field_names": ["vorticity"],
        "train_stats": {"vorticity": {"mean": mean, "std": std}},
    }
    manifest_path = os.path.join(args.out_dir, "kolmogorov2d_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {out_h5} ({total_t} frames x {n_pts} pts) and {manifest_path}")
    print(f"train stats (train frames only): mean {mean:.6f} std {std:.6f}")

    if args.verify:
        with h5py.File(out_h5, "r") as f:
            c = f["coordinates"][:, 0, 0, :]
            assert c.shape == (n_pts, 3), c.shape
            assert np.allclose(c[:, 2], 0.5)
            # frame 0 of the output is frame_ids[0] of the first train trajectory
            got = f["fields"][0, 0, :, 0, 0, 0].reshape(ny, nx)
            want = np.asarray(data[traj_sequence[0], frame_ids[0]], dtype=np.float32)
            assert np.array_equal(got, want), "round-trip mismatch on frame 0"
            # last frame maps to the last test trajectory's last strided frame
            got = f["fields"][0, total_t - 1, :, 0, 0, 0].reshape(ny, nx)
            want = np.asarray(data[traj_sequence[-1], frame_ids[-1]], dtype=np.float32)
            assert np.array_equal(got, want), "round-trip mismatch on last frame"
            # row-major ordering check: coords index iy*nx+ix carries x fastest
            assert np.allclose(c[:nx, 1], c[0, 1]), "first row should share y"
            assert not np.allclose(c[nx, 1], c[0, 1]), "second row should advance y"
        print("verify: PASS (shapes, z-column, round-trip frames, row-major order)")


if __name__ == "__main__":
    main()
