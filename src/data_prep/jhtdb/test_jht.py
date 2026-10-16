import os
import math
import random
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData


OUTDIR = Path(os.environ.get("JHTDB_OUTDIR", "./outputfiles")).resolve()
TOKEN = os.environ["JHTDB_TOKEN"]
DATASET_TITLE = os.environ.get("JHTDB_DATASET", "isotropic1024coarse")

GRID_FULL = 1024
CUBE = 125
SEED = 42

BASE_N_TIMES = int(os.environ.get("JHTDB_BASE_N_TIMES", "300"))
BASE_TIME_START = float(os.environ.get("JHTDB_BASE_TIME_START", "0.0"))
BASE_TIME_END = float(os.environ.get("JHTDB_BASE_TIME_END", "1.0"))
TIME_STEP = float(
    os.environ.get(
        "JHTDB_TIME_STEP",
        str((BASE_TIME_END - BASE_TIME_START) / (BASE_N_TIMES - 1)),
    )
)

# The first downloaded batch was global indices 0..299. By default this appends
# the next 1000 files as step_00300.h5..step_01299.h5.
TIME_INDEX_START = int(os.environ.get("JHTDB_TIME_INDEX_START", "300"))
N_TIMES = int(os.environ.get("JHTDB_N_TIMES", "1000"))

TIME_INDICES = np.arange(TIME_INDEX_START, TIME_INDEX_START + N_TIMES, dtype=np.int64)
TIMES = BASE_TIME_START + TIME_INDICES.astype(np.float64) * TIME_STEP

TEMPORAL_METHOD = "none"
SPATIAL_METHOD = "lag6"
SPATIAL_OPERATOR = "field"

L = 2.0 * math.pi


def make_periodic_subcube_points(
    i0: int, j0: int, k0: int, ni: int, nj: int, nk: int, nfull: int
) -> np.ndarray:
    dx = L / nfull

    ii = (np.arange(ni) + i0) % nfull
    jj = (np.arange(nj) + j0) % nfull
    kk = (np.arange(nk) + k0) % nfull

    x = ii.astype(np.float32) * dx
    y = jj.astype(np.float32) * dx
    z = kk.astype(np.float32) * dx

    pts = np.empty((ni * nj * nk, 3), dtype=np.float32)
    pts[:, 0] = np.repeat(x, nj * nk)
    pts[:, 1] = np.tile(np.repeat(y, nk), ni)
    pts[:, 2] = np.tile(z, ni * nj)
    return pts


def query_variable_chunked(
    ds,
    var_name: str,
    time_value: float,
    i0: int,
    j0: int,
    k0: int,
    cube: int = 128,
    chunk_i: int = 64,
) -> np.ndarray:
    parts = []
    n_chunks = math.ceil(cube / chunk_i)
    print(
        f"  [{var_name}] querying t={time_value:.6f} in {n_chunks} chunk(s) "
        f"(cube={cube}, chunk_i={chunk_i})"
    )

    for i_start in range(0, cube, chunk_i):
        ni = min(chunk_i, cube - i_start)
        chunk_idx = i_start // chunk_i + 1
        print(
            f"    [{var_name}] chunk {chunk_idx}/{n_chunks}: i_start={i_start}, ni={ni}"
        )

        pts = make_periodic_subcube_points(
            i0=i0 + i_start,
            j0=j0,
            k0=k0,
            ni=ni,
            nj=cube,
            nk=cube,
            nfull=GRID_FULL,
        ).astype(np.float64)

        arr = getData(
            ds,
            var_name,
            float(time_value),
            TEMPORAL_METHOD,
            SPATIAL_METHOD,
            SPATIAL_OPERATOR,
            pts,
        )

        arr = np.asarray(arr, dtype=np.float32)

        if var_name == "velocity":
            arr = arr.reshape(ni * cube * cube, 3)
        elif var_name == "pressure":
            if arr.ndim == 1:
                arr = arr[:, None]
            arr = arr.reshape(ni * cube * cube, 1)
        else:
            raise ValueError(f"Unsupported variable: {var_name}")

        print(f"    [{var_name}] chunk {chunk_idx}/{n_chunks} result shape={arr.shape}")
        parts.append(arr)

    out = np.concatenate(parts, axis=0)
    print(f"  [{var_name}] concatenated shape={out.shape}")
    return out


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    steps_dir = OUTDIR / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    print("Running file:", __file__)
    print("Working directory:", Path.cwd())
    print("Output directory:", OUTDIR)
    print(
        "Config:",
        {
            "dataset": DATASET_TITLE,
            "grid_full": GRID_FULL,
            "cube": CUBE,
            "base_n_times": BASE_N_TIMES,
            "base_time_start": BASE_TIME_START,
            "base_time_end": BASE_TIME_END,
            "time_step": TIME_STEP,
            "time_index_start": TIME_INDEX_START,
            "n_times": N_TIMES,
            "seed": SEED,
            "temporal_method": TEMPORAL_METHOD,
            "spatial_method": SPATIAL_METHOD,
            "spatial_operator": SPATIAL_OPERATOR,
        },
    )
    print(
        "Time range:",
        f"indices={int(TIME_INDICES[0])}..{int(TIME_INDICES[-1])}, "
        f"start={float(TIMES[0]):.6f}, end={float(TIMES[-1]):.6f}, "
        f"count={len(TIMES)}",
    )

    ds = turb_dataset(
        dataset_title=DATASET_TITLE,
        output_path=str(OUTDIR),
        auth_token=TOKEN,
    )
    print("Connected to turbulence dataset API.")

    coords_path = OUTDIR / "coordinates.h5"
    if coords_path.exists():
        with h5py.File(coords_path, "r") as f:
            start_index_ijk = f.attrs.get("start_index_ijk")
            if start_index_ijk is None:
                raise KeyError(
                    f"{coords_path} is missing attr 'start_index_ijk'; "
                    "cannot safely append timesteps to the same spatial block"
                )
            i0, j0, k0 = (int(v) for v in start_index_ijk)
            existing_cube = int(f.attrs.get("cube_size", CUBE))
            existing_full_grid = int(f.attrs.get("full_grid", GRID_FULL))
            if existing_cube != CUBE or existing_full_grid != GRID_FULL:
                raise ValueError(
                    f"{coords_path} describes cube={existing_cube}, "
                    f"full_grid={existing_full_grid}, but current config has "
                    f"cube={CUBE}, full_grid={GRID_FULL}"
                )
        print(f"Coordinates file already exists: {coords_path}")
        print(f"Using existing block start index: ({i0}, {j0}, {k0})")
    else:
        # Fix one spatial block, vary only time.
        i0 = random.randrange(GRID_FULL)
        j0 = random.randrange(GRID_FULL)
        k0 = random.randrange(GRID_FULL)
        print(f"Chosen block start index: ({i0}, {j0}, {k0})")
        print("Building and writing coordinates once...")
        coords_xyz = make_periodic_subcube_points(
            i0, j0, k0, CUBE, CUBE, CUBE, GRID_FULL
        )

        with h5py.File(coords_path, "w") as f:
            f.create_dataset("coordinates", data=coords_xyz)
            f.attrs["dataset_title"] = DATASET_TITLE
            f.attrs["start_index_ijk"] = np.array([i0, j0, k0], dtype=np.int32)
            f.attrs["cube_size"] = CUBE
            f.attrs["full_grid"] = GRID_FULL
            f.attrs["n_points"] = coords_xyz.shape[0]

        print(f"Wrote coordinates file: {coords_path}")

    # Download timestep data one at a time
    skipped_existing = 0
    written_steps = 0
    failed_steps = 0
    failure_counts = {
        "query_error": 0,
        "invalid_shape": 0,
        "write_error": 0,
    }
    failed_examples = []
    expected_points = CUBE * CUBE * CUBE
    for batch_idx, (global_t_idx, t) in enumerate(
        tqdm(
            zip(TIME_INDICES.tolist(), TIMES.tolist()),
            total=N_TIMES,
            desc="Downloading timesteps",
        )
    ):
        step_path = steps_dir / f"step_{global_t_idx:05d}.h5"
        if step_path.exists():
            skipped_existing += 1
            if skipped_existing <= 3 or skipped_existing % 25 == 0:
                print(
                    f"Skipping existing step file ({skipped_existing} skipped so far): "
                    f"{step_path.name}"
                )
            continue

        print(
            f"\nBatch index {batch_idx}/{N_TIMES - 1}, "
            f"global time index {global_t_idx}, t={t:.6f}"
        )

        try:
            vel = query_variable_chunked(
                ds, "velocity", float(t), i0, j0, k0, cube=CUBE, chunk_i=64
            )
            pres = query_variable_chunked(
                ds, "pressure", float(t), i0, j0, k0, cube=CUBE, chunk_i=64
            )
        except Exception as exc:
            failed_steps += 1
            failure_counts["query_error"] += 1
            failed_examples.append((global_t_idx, "query_error", repr(exc)))
            print(
                f"FAILED step {step_path.name}: query_error while fetching "
                f"t={t:.6f} -> {exc!r}"
            )
            continue

        if vel.shape != (expected_points, 3) or pres.shape != (expected_points, 1):
            failed_steps += 1
            failure_counts["invalid_shape"] += 1
            failed_examples.append(
                (
                    global_t_idx,
                    "invalid_shape",
                    f"velocity={vel.shape}, pressure={pres.shape}, expected_points={expected_points}",
                )
            )
            print(
                f"FAILED step {step_path.name}: invalid_shape "
                f"(velocity={vel.shape}, pressure={pres.shape}, "
                f"expected velocity=({expected_points}, 3), "
                f"expected pressure=({expected_points}, 1))"
            )
            continue

        try:
            with h5py.File(step_path, "w") as f:
                f.create_dataset("velocity", data=vel)
                f.create_dataset("pressure", data=pres)
                f.create_dataset("time", data=np.array([t], dtype=np.float32))
                f.attrs["time_index"] = global_t_idx
                f.attrs["batch_index"] = batch_idx
                f.attrs["time_index_start"] = TIME_INDEX_START
                f.attrs["time_step"] = TIME_STEP
                f.attrs["n_points"] = vel.shape[0]
        except Exception as exc:
            failed_steps += 1
            failure_counts["write_error"] += 1
            failed_examples.append((global_t_idx, "write_error", repr(exc)))
            print(
                f"FAILED step {step_path.name}: write_error while writing "
                f"t={t:.6f} -> {exc!r}"
            )
            continue

        written_steps += 1

        print(
            f"Wrote {step_path.name} | velocity={vel.shape} pressure={pres.shape} "
            f"time={t:.6f}"
        )

    print(
        "Done.",
        {
            "written_steps": written_steps,
            "skipped_existing": skipped_existing,
            "failed_steps": failed_steps,
            "failure_counts": failure_counts,
        },
    )
    if written_steps == 0:
        if skipped_existing == N_TIMES:
            print(
                "No new step files were written because all target step files already existed."
            )
        elif failed_steps > 0:
            print(
                "No new step files were written because each attempted timestep failed. "
                "See FAILED messages above for exact reasons."
            )
        else:
            print(
                "No new step files were written. This is unexpected given current settings."
            )

    if failed_examples:
        print("Example failures (up to first 5):")
        for t_idx, reason, details in failed_examples[:5]:
            print(f"  step_{t_idx:05d}.h5 | {reason} | {details}")


if __name__ == "__main__":
    main()
