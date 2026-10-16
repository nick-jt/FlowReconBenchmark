'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training
'''

# ═════════ Imports ═════════
import os
import time
import csv
import math
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

import h5py
from datetime import datetime
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Union
from scipy.ndimage import binary_dilation
from figures.regular_grid import regular_grid_from_points, imshow_points
from protocol import held_out_subset

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

__all__ = [
    "FIELD_NAMES",
    "normalize_coords",
    "TurbulentCombustionH5Dataset",
    "collate_snapshots",
    "validate_regular_grid_compatibility",
    "pointcloud_to_grid",
    "pointcloud_to_grid3d",
    "splat_to_grid",
    "gather_from_grid",
    "splat_obs_to_grid",
    "build_sparse_condition",
    "nearest_fill_grid",
    "midplane_slice",
    "compute_pad_size",
    "pointcloud_to_grid_padded",
    "grid_to_pointcloud",
    "grid3d_to_pointcloud",
    "build_obs_grid_mask",
    "build_obs_grid_mask3d",
    "scatter_sensors_to_nodes",
]

# ═════════ §1. Utility helpers ═════════

def _to_int_list(x: Union[int, Sequence[int], None]) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]

def _broadcast_per_field(
    values: Union[int, Sequence[int]],
    cond_fields: Sequence[int],
    name: str,
) -> list[int]:
    values = _to_int_list(values)
    if len(values) == 1:
        values = values * len(cond_fields)
    if len(values) != len(cond_fields):
        raise ValueError(
            f"{name} must have length 1 or match len(cond_fields). "
            f"Got {len(values)} vs {len(cond_fields)}."
        )
    return values

def _normalized_l2(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))

def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    cmin = coords.min(dim=0).values
    cmax = coords.max(dim=0).values
    scale = (cmax - cmin).clamp_min(1e-8)
    return (coords - cmin) / scale



def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

# ═════════ §2. Datasets ═════════

class TurbulentCombustionH5Dataset(Dataset):
    """Treat each time snapshot as one point-cloud sample."""

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        field_names: Tuple[str, ...] = ("CH4", "CO", "T", "U_1", "p"),
        stats_path: Optional[str] = None,
        stats_chunk: int = 32,
        time_stride: int = 1,
        sensor_pool: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.h5_path     = str(h5_path)
        self.split       = split
        self.field_names = field_names
        self.stats_chunk = stats_chunk
        self.time_stride = time_stride
        self._h5         = None
        # sensor_pool "surface": restrict sensors to the H5's `surface_indices`
        # pool; exposed per sample as `valid_sensor_mask` [N] bool, which the
        # adapters' run_epoch_* already forward to build_sparse_condition.
        self.sensor_pool = sensor_pool
        self.valid_sensor_mask = None

        with h5py.File(self.h5_path, "r") as f:
            self.num_times  = int(f["fields"].shape[1])
            raw_coords      = torch.from_numpy(f["coordinates"][:, 0, 0, :].astype(np.float32))

            self.coords_raw = raw_coords.clone()
            self.coords     = normalize_coords(raw_coords)
            self.num_points = int(raw_coords.shape[0])
            self.num_fields = int(f["fields"].shape[-1])
            self.times      = torch.from_numpy(f["time"][:].astype(np.float32))
            if sensor_pool is not None:
                if sensor_pool != "surface":
                    raise ValueError(f"unknown sensor_pool {sensor_pool!r} (expected None or 'surface')")
                if "surface_indices" in f:
                    pool_np = f["surface_indices"][:]
                else:
                    # Sidecar written by add_grid_surface_indices.py when the H5
                    # could not be opened r+ (HDF5 lock held by running jobs).
                    side = Path(self.h5_path).with_suffix(".surface_indices.npy")
                    if not side.exists():
                        raise KeyError(f"{self.h5_path} has no `surface_indices` dataset and no "
                                       f"sidecar {side.name}; sensor_pool='surface' needs one "
                                       "(see add_grid_surface_indices.py)")
                    pool_np = np.load(side)
                pool = torch.from_numpy(np.asarray(pool_np).astype(np.int64))
                mask = torch.zeros(self.num_points, dtype=torch.bool)
                mask[pool] = True
                self.valid_sensor_mask = mask
                self.sensor_pool_size = int(mask.sum())
                print(f"[dataset] sensor_pool=surface: {self.sensor_pool_size} of "
                      f"{self.num_points} points eligible ({self.h5_path.split('/')[-1]})")

        all_indices = np.arange(0, self.num_times, self.time_stride, dtype=np.int64)
        split_mode = os.environ.get("JHU_SPLIT_MODE", "shuffle")
        if split_mode == "block":
            # Temporally-blocked split: contiguous val block at the end,
            # separated from train by a decorrelation gap. Consecutive DNS
            # frames correlate at r~1.0, so shuffled splits leak.
            gap = int(os.environ.get("JHU_SPLIT_GAP", "50"))
            # round() not int(): 1.0-0.8 = 0.19999... makes int() undercount
            # n_val by 1 and silently leak the first val-block frame into
            # train (caught on the Kolmogorov trajectory-holdout split).
            n_val = max(1, int(round(len(all_indices) * (1.0 - train_ratio))))
            n_train = max(1, len(all_indices) - n_val - gap)
        else:
            rng = np.random.default_rng(seed)
            rng.shuffle(all_indices)
            n_train = int(len(all_indices) * train_ratio)
        if split == "train":
            self.indices = all_indices[:n_train]
        elif split in ("val", "tune", "test"):
            # In block mode the gap frames right after the train block are
            # discarded entirely; the held-out set is only the decorrelated
            # tail. "val" is the whole held-out set; "tune" / "test" are its
            # odd / even halves (protocol): selection on TUNE, scores on TEST.
            offset = gap if split_mode == "block" else 0
            self.indices = held_out_subset(np.sort(all_indices[n_train + offset:]), split)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.indices = np.sort(self.indices)
        print(f"[split] {split}: {len(self.indices)} frames "
              f"[{int(self.indices[0])}..{int(self.indices[-1])}] of {self.num_times}", flush=True)

        # Same exact octahedral augmentation the point-cloud model gets, so the
        # baselines are trained under identical data conditions.
        # Grid-based baselines take only the grid-safe symmetries; the
        # point-cloud-only ones (continuous SO(3)) would need interpolation
        # here and are deliberately withheld rather than approximated.
        self.augment_octahedral = False
        self._grid_shape = None
        requested = {s.strip() for s in os.environ.get("JHU_AUGMENT", "").split(",") if s.strip()}
        if "octahedral" in requested and split == "train":
            side = int(round(self.num_points ** (1.0 / 3.0)))
            if side ** 3 == self.num_points:
                self.augment_octahedral = True
                self._grid_shape = (side, side, side)
                print(f"[dataset] octahedral augmentation ON (grid {side}^3, train split)")
        # Single-axis reflection (FireBench lateral symmetry) — same operator
        # the point-cloud model trains with, so no method gets an advantage.
        self.augment_reflect_y = False
        if "reflect_y" in requested and split == "train":
            env_grid = os.environ.get("AUG_GRID_SHAPE", "")
            gs = tuple(int(s) for s in env_grid.split(",") if s.strip())
            if len(gs) == 3 and int(np.prod(gs)) == self.num_points:
                self.augment_reflect_y = True
                self._grid_shape = gs
                print(f"[dataset] reflect_y augmentation ON (grid {gs}, train split)")
            else:
                print(f"[dataset] reflect_y needs AUG_GRID_SHAPE matching {self.num_points} pts; skipped")

        self.stats_path = stats_path or str(Path(self.h5_path).with_suffix(".stats.pt"))
        self.mean, self.std = self._load_or_compute_stats(train_indices=np.sort(all_indices[:n_train]))

    def _require_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _load_or_compute_stats(self, train_indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        stats_path = Path(self.stats_path)
        if stats_path.exists():
            obj = _torch_load_compat(stats_path, map_location="cpu")
            return obj["mean"].float(), obj["std"].float()

        h5 = self._require_h5()
        total_sum = torch.zeros(self.num_fields, dtype=torch.float64)
        total_sq = torch.zeros(self.num_fields, dtype=torch.float64)
        total_count = 0

        for start in range(0, len(train_indices), self.stats_chunk):
            idx = train_indices[start : start + self.stats_chunk]
            arr = h5["fields"][0, idx, :, 0, 0, :]  # [Tchunk, N, C]
            x = torch.from_numpy(arr.astype(np.float32))
            total_sum += x.sum(dim=(0, 1), dtype=torch.float64)
            total_sq += (x.double() ** 2).sum(dim=(0, 1))
            total_count += x.shape[0] * x.shape[1]

        mean = (total_sum / total_count).float()
        var = (total_sq / total_count - mean.double() ** 2).clamp_min(1e-12).float()
        std = torch.sqrt(var)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "std": std}, stats_path)
        return mean, std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        t_idx = int(self.indices[i])
        h5 = self._require_h5()
        x = h5["fields"][0, t_idx, :, 0, 0, :].astype(np.float32)
        x = torch.from_numpy(x)
        x = (x - self.mean) / self.std
        if getattr(self, "augment_octahedral", False):
            from methods.dmfgen.augment_symmetry import octahedral_augment
            x = octahedral_augment(x, self._grid_shape, vel_idx=(0, 1, 2))
        if getattr(self, "augment_reflect_y", False):
            from methods.dmfgen.augment_symmetry import reflect_axis_augment
            x = reflect_axis_augment(x, self._grid_shape, axis=1, sign_flip_idx=[1])
        out = {
            "coords": self.coords.clone(),          # normalized coordinates for model
            "coords_raw": self.coords_raw.clone(),  # original physical coordinates for plotting
            "fields": x,                    
            "time_index": torch.tensor(t_idx, dtype=torch.long),
            "physical_time": self.times[t_idx].clone(),
        }
        if self.valid_sensor_mask is not None:
            out["valid_sensor_mask"] = self.valid_sensor_mask.clone()
        return out

# -------------------------------------------




# -------------------------------------------





# ═════════ §3. Collation & grid validation ═════════

_OPTIONAL_COLLATE_KEYS = (
    'physical_time',       # absent on SHIFT-WING (per-case steady RANS, no time)
    'coords_raw',
    'valid_sensor_mask',
    # SHIFT-WING surface observation pool + exact flow-condition tokens. These
    # were previously dropped on the floor: the whitelist below is the only
    # thing that reaches a training loop, so without them no baseline could
    # ever see the surface pool even with the right dataset wired in.
    'obs_pool_coords',
    'obs_pool_values',
    'obs_pool_field_ids',
    'param_coords',
    'param_values',
    'param_field_ids',
)


def collate_snapshots(batch):
    """Shared collate that handles optional keys like valid_sensor_mask, coords_raw.

    `physical_time` is optional rather than required: every H5 dataset carries
    it, but SHIFT-WING cases are steady-state solutions with no time axis.
    """
    out = {
        'coords':        torch.stack([b['coords']        for b in batch], dim=0),
        'fields':        torch.stack([b['fields']        for b in batch], dim=0),
        'time_index':    torch.stack([b['time_index']    for b in batch], dim=0),
    }
    for key in _OPTIONAL_COLLATE_KEYS:
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out

def validate_regular_grid_compatibility(
    dataset: Dataset,
    Num_x: Optional[int],
    Num_y: Optional[int],
    decimals: int = 6,
    atol: float = 1e-5,
) -> Dict[str, object]:
    """
    Validate that a point-cloud dataset can be interpreted as a Num_x by Num_y regular grid.

    Required behavior for grid baselines:
      - Num_x and Num_y must be provided in YAML / args
      - Num_x * Num_y must match the number of points
      - the coordinate set must contain exactly Num_x unique x-values
        and Num_y unique y-values (up to rounding)
      - every (x, y) tensor-product grid cell must exist exactly once

    Like the FFM-FNO path, this function does not require row-major HDF5 point
    order or uniform physical spacing. It reports those facts and returns a
    coordinate-derived row-major permutation so direct-reshape grid baselines
    can reorder fields/observations safely before reshaping.

    Raises ValueError if the dataset is not compatible with the requested grid.
    Returns a small diagnostics dict for launch-time logging.
    """
    def _grid_baseline_error(reason: str) -> ValueError:
        return ValueError(
            "Regular-grid validation failed for a grid baseline "
            f"(SiT/FNO/S3GM-style): {reason} The dataset may be a valid point "
            "cloud, but it cannot be safely interpreted as a complete structured "
            "grid for this baseline. Use a point-cloud model or add an explicit "
            "dataset-specific gridding/interpolation step."
        )

    if Num_x is None or Num_y is None:
        raise ValueError(
            "Direct-reshape grid baselines require Num_x and Num_y to be explicitly "
            "provided in YAML / args."
        )

    Num_x = int(Num_x)
    Num_y = int(Num_y)

    if Num_x <= 0 or Num_y <= 0:
        raise ValueError(f"Num_x and Num_y must be positive, got Num_x={Num_x}, Num_y={Num_y}.")

    expected_points = Num_x * Num_y
    if int(dataset.num_points) != expected_points:
        raise _grid_baseline_error(
            f"Grid mismatch: dataset has {dataset.num_points} points, but "
            f"Num_x * Num_y = {Num_x} * {Num_y} = {expected_points}."
        )

    coords = dataset.coords.cpu()
    x = torch.round(coords[:, 0] * (10 ** decimals)) / (10 ** decimals)
    y = torch.round(coords[:, 1] * (10 ** decimals)) / (10 ** decimals)

    unique_x = int(torch.unique(x).numel())
    unique_y = int(torch.unique(y).numel())

    if unique_x != Num_x or unique_y != Num_y:
        raise _grid_baseline_error(
            f"Dataset unique counts are ({unique_x}, {unique_y}) in (x, y), "
            f"but requested (Num_x, Num_y)=({Num_x}, {Num_y})."
        )

    unique_x_vals = torch.unique(x, sorted=True)
    unique_y_vals = torch.unique(y, sorted=True)

    def _spacing_stats(values: torch.Tensor) -> tuple[bool, torch.Tensor]:
        if values.numel() <= 1:
            diffs = torch.ones(1, dtype=values.dtype, device=values.device)
            return True, diffs
        diffs = values[1:] - values[:-1]
        if diffs.numel() <= 1:
            return True, diffs
        regular = bool(torch.allclose(diffs, diffs.median().expand_as(diffs), atol=atol, rtol=1e-4))
        return regular, diffs

    x_spacing_regular, x_diffs = _spacing_stats(unique_x_vals)
    y_spacing_regular, y_diffs = _spacing_stats(unique_y_vals)
    coord_dim = int(coords.shape[-1])
    coords_grid = coords.reshape(Num_y, Num_x, coord_dim)
    x_grid = torch.round(coords_grid[..., 0] * (10 ** decimals)) / (10 ** decimals)
    y_grid = torch.round(coords_grid[..., 1] * (10 ** decimals)) / (10 ** decimals)

    x_first_row = x_grid[0]
    y_first_col = y_grid[:, 0]
    x_expected_asc = unique_x_vals
    x_expected_desc = torch.flip(unique_x_vals, dims=[0])
    y_expected_asc = unique_y_vals
    y_expected_desc = torch.flip(unique_y_vals, dims=[0])

    x_constant_down_columns = bool(torch.allclose(
        x_grid,
        x_first_row.view(1, Num_x).expand(Num_y, Num_x),
        atol=atol,
        rtol=1e-4,
    ))
    y_constant_across_rows = bool(torch.allclose(
        y_grid,
        y_first_col.view(Num_y, 1).expand(Num_y, Num_x),
        atol=atol,
        rtol=1e-4,
    ))
    x_varies_along_axis1 = bool(
        torch.allclose(x_first_row, x_expected_asc, atol=atol, rtol=1e-4)
        or torch.allclose(x_first_row, x_expected_desc, atol=atol, rtol=1e-4)
    )
    y_varies_along_axis0 = bool(
        torch.allclose(y_first_col, y_expected_asc, atol=atol, rtol=1e-4)
        or torch.allclose(y_first_col, y_expected_desc, atol=atol, rtol=1e-4)
    )

    row_major = (
        x_constant_down_columns
        and y_constant_across_rows
        and x_varies_along_axis1
        and y_varies_along_axis0
    )

    _, x_rank = torch.unique(x, sorted=True, return_inverse=True)
    _, y_rank = torch.unique(y, sorted=True, return_inverse=True)
    point_to_grid = (y_rank.long() * Num_x + x_rank.long()).contiguous()
    unique_cells = int(torch.unique(point_to_grid).numel())
    if unique_cells != expected_points:
        raise _grid_baseline_error(
            "The coordinate set is not a complete tensor-product grid. "
            f"Detected unique counts are (x={unique_x}, y={unique_y}), but "
            f"unique (x, y) cells are {unique_cells} out of {expected_points}."
        )

    diagnostics = []
    if not x_spacing_regular:
        diagnostics.append("unique x coordinates are not approximately regularly spaced")
    if not y_spacing_regular:
        diagnostics.append("unique y coordinates are not approximately regularly spaced")
    if not row_major:
        diagnostics.append(
            "flattened point order is not row-major; grid baselines will apply a coordinate-derived permutation"
        )

    grid_order = torch.argsort(point_to_grid).contiguous()
    return {
        "Num_x": Num_x,
        "Num_y": Num_y,
        "num_points": expected_points,
        "unique_x": unique_x,
        "unique_y": unique_y,
        "complete_tensor_product": True,
        "row_major": row_major,
        "requires_permutation": not row_major,
        "spacing_regular": x_spacing_regular and y_spacing_regular,
        "x_spacing_min": float(x_diffs.min().item()),
        "x_spacing_median": float(x_diffs.median().item()),
        "x_spacing_max": float(x_diffs.max().item()),
        "y_spacing_min": float(y_diffs.min().item()),
        "y_spacing_median": float(y_diffs.median().item()),
        "y_spacing_max": float(y_diffs.max().item()),
        "diagnostics": diagnostics,
        "grid_order": grid_order,
        "point_to_grid": point_to_grid,
        "first_row_original_indices": [int(v) for v in grid_order[: min(8, grid_order.numel())].tolist()],
        "first_original_to_grid_indices": [int(v) for v in point_to_grid[: min(8, point_to_grid.numel())].tolist()],
    }

# ═════════ §4. Grid ↔ point-cloud interchange ═════════

# ----- Irregular <-> regular grid transfer helpers ---------------------------

def pointcloud_to_grid(x: torch.Tensor, Num_y: int, Num_x: int) -> torch.Tensor:
    """[B, N, C] -> [B, C, Num_y, Num_x]. Assumes N == Num_y * Num_x, row-major."""
    B = x.shape[0]
    return x.reshape(B, Num_y, Num_x, -1).permute(0, 3, 1, 2).contiguous()


def pointcloud_to_grid3d(x: torch.Tensor, Num_z: int, Num_y: int, Num_x: int) -> torch.Tensor:
    """[B, N, C] -> [B, C, Num_z, Num_y, Num_x]. Assumes N == Num_z * Num_y * Num_x."""
    B = x.shape[0]
    return x.reshape(B, Num_z, Num_y, Num_x, -1).permute(0, 4, 1, 2, 3).contiguous()


def splat_to_grid(coords_2d: torch.Tensor, values: torch.Tensor, Ny: int, Nx: int):
    """
    Bilinearly splat per-point values onto a regular (Ny, Nx) grid.

    Args:
        coords_2d: [B, K, 2] normalized coords in [0, 1]
        values:    [B, K, D]
        Ny, Nx:    target grid size
    Returns:
        grid:   [B, D, Ny, Nx] values averaged by accumulated weight
        weight: [B, 1, Ny, Nx] accumulated weights
    """
    B, K, D = values.shape
    device = values.device
    dtype = values.dtype

    px = coords_2d[..., 0] * (Nx - 1)
    py = coords_2d[..., 1] * (Ny - 1)

    ix0 = px.long().clamp(0, Nx - 2)
    iy0 = py.long().clamp(0, Ny - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    wx = (px - ix0.float()).unsqueeze(-1)
    wy = (py - iy0.float()).unsqueeze(-1)

    grid = torch.zeros(B, D, Ny * Nx, device=device, dtype=dtype)
    weight = torch.zeros(B, 1, Ny * Nx, device=device, dtype=dtype)

    for dy, dwy in [(iy0, 1 - wy), (iy1, wy)]:
        for dx, dwx in [(ix0, 1 - wx), (ix1, wx)]:
            w = dwy * dwx
            flat = (dy * Nx + dx).unsqueeze(1)        # [B, 1, K]

            wf = (values * w).permute(0, 2, 1)        # [B, D, K]
            grid.scatter_add_(2, flat.expand_as(wf), wf)

            ww = w.permute(0, 2, 1)                    # [B, 1, K]
            weight.scatter_add_(2, flat, ww)

    weight_c = weight.clamp(min=1e-6)
    grid = grid / weight_c
    return grid.reshape(B, D, Ny, Nx), weight.reshape(B, 1, Ny, Nx)

def gather_from_grid(grid: torch.Tensor, coords_2d: torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation from regular grid to arbitrary query points.

    Args:
        grid:      [B, C, Ny, Nx]
        coords_2d: [B, M, 2] normalized coords in [0, 1]
    Returns:
        values: [B, M, C]
    """
    sample_pts = coords_2d * 2.0 - 1.0
    sample_pts = sample_pts.unsqueeze(2)                 # [B, M, 1, 2]
    gathered = F.grid_sample(
        grid, sample_pts, mode="bilinear",
        padding_mode="border", align_corners=True,
    )                                                     # [B, C, M, 1]
    return gathered.squeeze(-1).permute(0, 2, 1)

def splat_obs_to_grid(obs_coords_2d: torch.Tensor,
                      obs_values: torch.Tensor,
                      obs_mask: torch.Tensor,
                      obs_field_ids: torch.Tensor,
                      n_fields: int,
                      Ny: int,
                      Nx: int):
    """
    Scatter sparse sensor observations onto per-field (Ny, Nx) grids.

    Args:
        obs_coords_2d: [B, M, 2] normalized coords in [0, 1]
        obs_values:    [B, M, 1]
        obs_mask:      [B, M]
        obs_field_ids: [B, M] integer field id for each sensor (-1 if invalid)
        n_fields:      number of target channels
        Ny, Nx:        grid size
    Returns:
        val_grid: [B, n_fields, Ny, Nx]
        msk_grid: [B, n_fields, Ny, Nx]
    """
    B, M, _ = obs_coords_2d.shape
    device = obs_coords_2d.device
    dtype = obs_coords_2d.dtype

    val_grid = torch.zeros(B, n_fields, Ny * Nx, device=device, dtype=dtype)
    count_grid = torch.zeros(B, n_fields, Ny * Nx, device=device, dtype=dtype)

    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        xy = obs_coords_2d[b, valid]
        ix = (xy[:, 0] * (Nx - 1)).long().clamp(0, Nx - 1)
        iy = (xy[:, 1] * (Ny - 1)).long().clamp(0, Ny - 1)
        flat = iy * Nx + ix
        # Accumulate; multiple sensors colliding in the same cell are averaged
        # after the loop. Assignment would silently drop all but the last.
        val_grid[b].index_put_((fld, flat), val, accumulate=True)
        count_grid[b].index_put_(
            (fld, flat), torch.ones_like(val), accumulate=True,
        )

    val_grid = val_grid / count_grid.clamp_min(1.0)
    msk_grid = (count_grid > 0).to(dtype)

    return val_grid.reshape(B, n_fields, Ny, Nx), msk_grid.reshape(B, n_fields, Ny, Nx)

# ----- Plotting helpers for unstructured / body-fitted geometries ------------

def _build_structured_triangulation(coords_xy: np.ndarray, grid_shape):
    # zcollapse-ok: triangulates a 2-D slice grid already extracted by callers (coords_xy is planar)
    """
    Build a matplotlib Triangulation that follows the (Ny, Nx) logical
    connectivity of a structured body-fitted mesh. Each quad cell is
    split into two triangles.

    Args:
        coords_xy:  [N, 2] flattened (row-major, j varies fastest along Nx)
        grid_shape: (Ny, Nx)
    Returns:
        mtri.Triangulation
    """
    # Row-major (C-order) flattening: index = j*Nx + i. Verified against
    # ElasticityDataset (meshgrid indexing='ij' + ravel) and
    # AirfoilCGridDataset (CX shape (N,Ny,Nx) reshaped via reshape(N,-1)).
    # Empirical check: 2*(Ny-1)*(Nx-1) non-degenerate unit-area triangles.
    Ny, Nx = int(grid_shape[0]), int(grid_shape[1])
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    idx = np.arange(Ny * Nx).reshape(Ny, Nx)
    v00 = idx[:-1, :-1].ravel()
    v10 = idx[1:, :-1].ravel()
    v01 = idx[:-1, 1:].ravel()
    v11 = idx[1:, 1:].ravel()

    tris = np.concatenate(
        [np.stack([v00, v10, v11], axis=1),
         np.stack([v00, v11, v01], axis=1)],
        axis=0,
    )
    return mtri.Triangulation(x, y, triangles=tris)

def midplane_slice(coords_np, axis: int = 2):
    """
    Pick a single plane out of a volumetric point cloud for 2-D plotting.

    Projecting every z-level of a 125^3 cube onto (x, y) superposes ~125
    coincident points per location, so the triangulation renders a smear
    rather than a field. This returns the plane nearest the median of
    ``axis`` so every model's figures show the same cross-section.

    Returns:
        (is_3d, mask, coords_xy_plot, triang) where ``mask`` indexes the
        selected plane (``slice(None)`` for genuinely 2-D data) and
        ``triang`` is None when the caller should build its own.
    """
    coords_np = np.asarray(coords_np)
    if coords_np.shape[-1] <= axis or not bool(np.ptp(coords_np[:, axis]) > 1e-6):
        return False, slice(None), coords_np[:, :2], None
    vals = coords_np[:, axis]
    # Select the middle *level* exactly. A median-plus-tolerance test picks
    # two planes whenever the level count is even, which silently doubles the
    # points handed to the triangulation.
    levels = np.unique(vals)
    mask = vals == levels[len(levels) // 2]
    xy = coords_np[mask][:, :2]
    return True, mask, xy, mtri.Triangulation(xy[:, 0], xy[:, 1])


# ═════════ §5. Sparse-condition construction ═════════

def build_sparse_condition(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    cond_fields: Union[int, Sequence[int]],
    n_obs_min: Union[int, Sequence[int]],
    n_obs_max: Union[int, Sequence[int]],
    valid_mask: Optional[torch.Tensor] = None,
    coords_2d: Optional[torch.Tensor] = None,
    Ny: Optional[int] = None,
    Nx: Optional[int] = None,
):
    """
    Generalized sparse conditioning.

    Args:
        coords_full: [B, N, D]
        fields_full: [B, N, C]
        cond_fields: int or list[int], e.g. 2 or [0, 2]
        n_obs_min: int or list[int], per conditioned field
        n_obs_max: int or list[int], per conditioned field
        valid_mask: optional [B, N] or [N] bool/float mask restricting sensor
            placement to a subset of points (e.g. near the stress
            concentration around a hole for elasticity, or near the airfoil
            surface). Points where the mask is 0/False are never sampled.
            If a batch item has fewer valid points than n_obs_max for a
            field, sampling is capped at the number of valid points (and
            the unused slots stay zero-masked).
        coords_2d, Ny, Nx: optional grid-aware sampling. If coords_2d [B, N, 2]
            in [0, 1] and (Ny, Nx) are provided, the sampler guarantees that
            each sampled sensor lands in a distinct latent grid cell for its
            field. Use for irregular meshes where the Geo-FNO-style deformer
            can collapse multiple mesh points into the same cell. If the pool
            (after valid_mask) contains fewer than n_obs distinct cells for a
            field, m is capped at the number of reachable cells.

    Returns:
        obs_coords:    [B, M, D]
        obs_values:    [B, M, 1]
        obs_mask:      [B, M]
        obs_indices:   [B, M]
        obs_field_ids: [B, M]   # which field each sensor belongs to
    """
    cond_fields = _to_int_list(cond_fields)
    if len(cond_fields) == 0:
        raise ValueError("cond_fields must contain at least one field index.")

    n_obs_min = _broadcast_per_field(n_obs_min, cond_fields, "n_obs_min")
    n_obs_max = _broadcast_per_field(n_obs_max, cond_fields, "n_obs_max")

    for a, b in zip(n_obs_min, n_obs_max):
        if b < a:
            raise ValueError(f"Each n_obs_max must be >= n_obs_min, got {a} and {b}.")

    bsz, n_pts, coord_dim = coords_full.shape
    device = coords_full.device

    grid_aware = coords_2d is not None
    if grid_aware:
        if Ny is None or Nx is None:
            raise ValueError("Grid-aware sampling requires both Ny and Nx.")
        if coords_2d.shape[:2] != (bsz, n_pts):
            raise ValueError(
                f"coords_2d shape {tuple(coords_2d.shape)} incompatible with "
                f"coords_full shape {tuple(coords_full.shape)}."
            )
        ix_all = (coords_2d[..., 0] * (Nx - 1)).long().clamp(0, Nx - 1)
        iy_all = (coords_2d[..., 1] * (Ny - 1)).long().clamp(0, Ny - 1)
        cells_all = iy_all * Nx + ix_all   # [B, N]
        n_cells_total = Ny * Nx

    max_obs = sum(n_obs_max)

    obs_coords = torch.zeros(
        bsz, max_obs, coord_dim, device=device, dtype=coords_full.dtype
    )
    obs_values = torch.zeros(
        bsz, max_obs, 1, device=device, dtype=fields_full.dtype
    )
    obs_mask = torch.zeros(
        bsz, max_obs, device=device, dtype=coords_full.dtype
    )
    obs_indices = torch.zeros(
        bsz, max_obs, device=device, dtype=torch.long
    )
    obs_field_ids = torch.full(
        (bsz, max_obs), -1, device=device, dtype=torch.long
    )

    if valid_mask is not None:
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0).expand(bsz, -1)
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    for b in range(bsz):
        if valid_mask is not None:
            valid_idx_b = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
        else:
            valid_idx_b = None

        cursor = 0
        for fld, nmin, nmax in zip(cond_fields, n_obs_min, n_obs_max):
            if grid_aware:
                if valid_idx_b is not None:
                    pool = valid_idx_b
                else:
                    pool = torch.arange(n_pts, device=device)
                V = int(pool.numel())
                if V == 0:
                    continue
                # Random permutation, then keep only the first-seen point per
                # unique grid cell so each sensor lands in a distinct cell.
                perm = pool[torch.randperm(V, device=device)]
                perm_cells = cells_all[b, perm]                       # [V]
                positions = torch.arange(V, device=device)
                min_pos = torch.full(
                    (n_cells_total,), V, device=device, dtype=torch.long,
                )
                min_pos.scatter_reduce_(0, perm_cells, positions, reduce="amin")
                is_first = positions == min_pos[perm_cells]           # [V]
                first_points = perm[is_first]                         # [U]
                pool_size = int(first_points.numel())
                eff_max = min(nmax, pool_size)
                eff_min = min(nmin, eff_max)
                if eff_max <= 0:
                    continue
                m = int(torch.randint(
                    low=eff_min, high=eff_max + 1, size=(1,), device=device,
                ).item())
                idx = first_points[:m].sort().values
            elif valid_idx_b is not None:
                pool_size = int(valid_idx_b.numel())
                eff_max = min(nmax, pool_size)
                eff_min = min(nmin, eff_max)
                if eff_max <= 0:
                    continue
                m = int(torch.randint(low=eff_min, high=eff_max + 1, size=(1,), device=device).item())
                perm = torch.randperm(pool_size, device=device)[:m]
                idx = valid_idx_b[perm].sort().values
            else:
                m = int(torch.randint(low=nmin, high=nmax + 1, size=(1,), device=device).item())
                idx = torch.randperm(n_pts, device=device)[:m].sort().values

            obs_coords[b, cursor:cursor + m] = coords_full[b, idx]
            obs_values[b, cursor:cursor + m, 0] = fields_full[b, idx, fld]
            obs_mask[b, cursor:cursor + m] = 1.0
            obs_indices[b, cursor:cursor + m] = idx
            obs_field_ids[b, cursor:cursor + m] = fld

            cursor += m

    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


def nearest_fill_grid(value_grid: torch.Tensor,
                      mask_grid: torch.Tensor) -> torch.Tensor:
    """Per-(batch, field) nearest-neighbor Voronoi fill of a sparse grid.

    Every unobserved cell is filled with the value of the closest observed
    cell under L2 distance on grid indices. Used by the 'interp' sparse
    conditioning mode: instead of feeding the denoiser a mostly-zero sparse
    field + mask, feed a dense Voronoi-interpolated field + mask so the
    network has an immediate inductive bias for smooth infill (Shu et al.,
    arXiv:2409.00230).

    Channels/samples with zero observations are left as-is (all zeros). The
    mask is NOT modified — callers should still pass the original binary
    mask alongside so the network can distinguish interpolated guesses
    from actual measurements.

    Implementation: scipy's exact Euclidean distance transform on CPU.
    Cheap at typical grid sizes (~1 ms per field per sample at 100x400);
    move to a GPU variant if it becomes a training bottleneck.
    """
    from scipy.ndimage import distance_transform_edt
    dev = value_grid.device
    dtype = value_grid.dtype
    vg = value_grid.detach().cpu().numpy()
    mg = mask_grid.detach().cpu().numpy() > 0.5
    filled = vg.copy()
    B, C = vg.shape[:2]
    for b in range(B):
        for c in range(C):
            m = mg[b, c]
            if not m.any() or m.all():
                continue
            inds = distance_transform_edt(
                ~m, return_distances=False, return_indices=True)
            filled[b, c] = vg[b, c][tuple(inds)]
    return torch.from_numpy(filled).to(device=dev, dtype=dtype)


# ═════════ §6. Metrics & logging ═════════


# ═════════ §7. Visualization ═════════

def _save_single_field_plot(
    true_f=None, pred_f=None, coords_xy=None, sensor_coords=None,
    field_name="field", epoch=0, save_dir=".", file_prefix=None,
    triang=None, body_polygon=None,
    metric_true_f=None, metric_pred_f=None,
    dpi=300, cmap_field="coolwarm", cmap_err="inferno",
    contour_levels=20, contour_linewidth=0.5, contour_alpha=0.5,
    **kwargs,
):
    """
    Module-level 3-panel plot (Ground truth / Reconstruction / |Error|).

    Compatible with both the original keyword-only call from
    visualize_reconstruction and the positional calls in the baselines,
    which pass extra ``triang`` / ``body_polygon`` kwargs for body-fitted
    plotting. If ``triang`` is not supplied, a Delaunay triangulation is
    built from ``coords_xy`` as before.
    """
    # Superset of the earlier keyword-only signature; adds optional
    # triang / body_polygon kwargs used by baseline scripts
    # (train_latentfm_baseline.py, train_sit_baseline.py,
    # train_senseiver_baseline.py, evaluate_ffm.py, evaluate_s3gm.py).
    x_plot = coords_xy[:, 0]
    y_plot = coords_xy[:, 1]
    true_f = np.asarray(true_f, dtype=np.float64)
    pred_f = np.asarray(pred_f, dtype=np.float64)

    # Points on a complete, uniformly spaced tensor-product grid (Kolmogorov
    # 256^2, the cylinder grid export 200x400, a JHU z-midplane 125^2) are
    # rendered as a bicubic raster: a Delaunay triangulation of a regular grid
    # leaves facet / aliasing artefacts, worst for single generative samples
    # whose chunk-wise speckle it sharpens.  Decided from the coordinates, not
    # from the caller: body-fitted or curvilinear meshes fall through to the
    # triangulation path and keep any ``triang`` the caller supplied.
    grid = regular_grid_from_points(coords_xy)
    if grid is None and triang is None:
        triang = mtri.Triangulation(x_plot, y_plot)

    err = np.abs(true_f - pred_f)
    # When a slice is plotted, the reported error is still the full-volume
    # one so it stays comparable across models; the slice value is shown
    # alongside it for context.
    _slice_l2 = _normalized_l2(true_f, pred_f)
    if metric_true_f is not None and metric_pred_f is not None:
        l2_error = _normalized_l2(metric_true_f, metric_pred_f)
        _l2_label = (f"Normalized L2 = {l2_error:.3e} (volume)"
                     f"    |    {_slice_l2:.3e} (slice)")
    else:
        l2_error = _slice_l2
        _l2_label = f"Normalized L2 = {l2_error:.3e}"

    # Percentile clip, matching qualitative_jhu.py. Raw min/max lets a few
    # outliers compress the mid-range, which makes truth and reconstruction
    # both look smoother -- and more alike -- than they are. This is the
    # single change that most affects whether these figures can be trusted.
    _stack = np.concatenate([np.asarray(true_f).ravel(), np.asarray(pred_f).ravel()])
    field_min, field_max = (float(v) for v in np.percentile(_stack, [1, 99]))

    positive_err = err[err > 0]
    err_min = float(positive_err.min()) if positive_err.size > 0 else 0.0
    err_max = float(err.max()) if err.size > 0 else 1.0

    fig = plt.figure(figsize=(5.8, 12))
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[1.0, 0.045],
        wspace=0.06,
        hspace=0.20,
    )
    ax_true = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[2, 0])
    cax_field = fig.add_subplot(gs[0:2, 1])
    cax_err = fig.add_subplot(gs[2, 1])

    if grid is not None:
        # Regular grid: bicubic raster, pixel centres on the point coordinates
        # (so the sensor overlay below lands on the samples it annotates).
        X, Y = grid.meshgrid()
        im_true = imshow_points(ax_true, grid, true_f, cmap=cmap_field,
                                vmin=field_min, vmax=field_max)
        im_pred = imshow_points(ax_pred, grid, pred_f, cmap=cmap_field,
                                vmin=field_min, vmax=field_max)
        im_err = imshow_points(ax_err, grid, err, cmap=cmap_err,
                               vmin=err_min, vmax=err_max)
        if contour_levels is not None:
            for ax, f in ((ax_true, true_f), (ax_pred, pred_f)):
                ax.contour(X, Y, grid.image(f), levels=contour_levels,
                           colors="white", linewidths=contour_linewidth,
                           alpha=contour_alpha)
        for ax in (ax_true, ax_pred, ax_err):
            ax.set_xlim(grid.extent[0], grid.extent[1])
            ax.set_ylim(grid.extent[2], grid.extent[3])
    else:
        im_true = ax_true.tricontourf(
            triang, true_f, levels=100, cmap=cmap_field,
            vmin=field_min, vmax=field_max,
        )
        if contour_levels is not None:
            ax_true.tricontour(
                triang, true_f, levels=contour_levels, colors="white",
                linewidths=contour_linewidth, alpha=contour_alpha,
            )

        im_pred = ax_pred.tricontourf(
            triang, pred_f, levels=100, cmap=cmap_field,
            vmin=field_min, vmax=field_max,
        )
        if contour_levels is not None:
            ax_pred.tricontour(
                triang, pred_f, levels=contour_levels, colors="white",
                linewidths=contour_linewidth, alpha=contour_alpha,
            )

        im_err = ax_err.tricontourf(
            triang, err, levels=100, cmap=cmap_err,
            vmin=err_min, vmax=err_max, extend="both",
        )

    # Optional body polygon overlay (airfoil etc.)
    if body_polygon is not None and len(body_polygon) > 0:
        for ax in (ax_true, ax_pred, ax_err):
            ax.add_patch(Polygon(body_polygon, closed=True,
                                 facecolor="white", edgecolor="black",
                                 linewidth=0.8, zorder=3))

    # Overlay the sensor locations on the ground-truth panel. For volumetric
    # data these are only the sensors lying on the displayed plane, so the
    # count shown is a slice of the full sensor set, not all of it.
    if sensor_coords is not None and len(sensor_coords) > 0:
        _sc = np.asarray(sensor_coords)
        ax_true.scatter(_sc[:, 0], _sc[:, 1], s=9, marker="o",
                        facecolors="none", edgecolors="#00ff88",
                        linewidths=0.7, zorder=5)

    ax_true.set_title("Ground truth", fontsize=13)
    ax_pred.set_title("Reconstruction", fontsize=13)
    ax_err.set_title("|Error|", fontsize=13)

    for ax in (ax_true, ax_pred, ax_err):
        ax.set_aspect("equal")
        ax.set_anchor("W")
        ax.set_xticks([])
        ax.set_yticks([])

    # Auto-zoom around the airfoil: the C-grid far field is ~10× the chord,
    # which makes the airfoil a single pixel at full extent. Frame the view
    # to a neighbourhood of the body so the field structure near the surface
    # is actually visible.
    if body_polygon is not None and len(body_polygon) > 0:
        bp = np.asarray(body_polygon)
        bx_min, bx_max = float(bp[:, 0].min()), float(bp[:, 0].max())
        by_min, by_max = float(bp[:, 1].min()), float(bp[:, 1].max())
        chord = max(bx_max - bx_min, by_max - by_min, 1e-6)
        pad_x = 1.0 * chord
        pad_y = 0.75 * chord
        cx = 0.5 * (bx_min + bx_max)
        cy = 0.5 * (by_min + by_max)
        half_w = 0.5 * (bx_max - bx_min) + pad_x
        half_h = 0.5 * (by_max - by_min) + pad_y
        for ax in (ax_true, ax_pred, ax_err):
            ax.set_xlim(cx - half_w, cx + half_w)
            ax.set_ylim(cy - half_h, cy + half_h)

    cbar_field = fig.colorbar(im_true, cax=cax_field)
    cbar_field.set_label(field_name)
    cbar_err = fig.colorbar(im_err, cax=cax_err)
    cbar_err.set_label(f"|{field_name} - û|")

    fig.suptitle(
        f"{field_name}    |    {_l2_label}",
        y=0.96, fontsize=14,
    )

    prefix = file_prefix if file_prefix is not None else f"epoch_{epoch:04d}"
    filename = os.path.join(save_dir, f"{prefix}_field_{field_name}.png")
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return l2_error





# ═════════ §8. Training-resume utilities ═════════









# ═════════════════════════════════════════════════════════════════════════════
# §9. Shared grid ↔ point-cloud utilities used by multiple baselines
#     (SiT, S3GM, and any future grid-based method). These live here rather
#     than in a method-specific script because they are pure data-layout
#     helpers — no loss, no sampler, no training loop.
# ═════════════════════════════════════════════════════════════════════════════



def compute_pad_size(Num_y: int, Num_x: int, factor: int) -> Tuple[int, int]:
    """Pad (Num_y, Num_x) up to the nearest multiple of `factor`.

    Callers pass the factor their architecture requires:
      - SiT patch tokenizer:  factor = patch_size
      - S3GM UNet:            factor = 2 ** (len(ch_mult) - 1)
    """
    H_pad = math.ceil(Num_y / factor) * factor
    W_pad = math.ceil(Num_x / factor) * factor
    return H_pad, W_pad


def pointcloud_to_grid_padded(
    x: torch.Tensor,
    Num_y: int,
    Num_x: int,
    H_pad: int,
    W_pad: int,
    grid_order: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[B, N, C] -> [B, C, H_pad, W_pad] with zero padding if needed.

    Distinct from `pointcloud_to_grid` above (which returns unpadded
    [B, C, Num_y, Num_x]); this variant produces the padded tensor that
    fully-convolutional / patch-tokenized architectures require. If
    grid_order is supplied, it maps row-major grid cells to original point
    indices and is applied before reshape, matching the FFM-FNO path.
    """
    B = x.shape[0]
    if grid_order is not None:
        x = x[:, grid_order.to(device=x.device), :]
    g = x.reshape(B, Num_y, Num_x, -1).permute(0, 3, 1, 2).contiguous()
    if H_pad > Num_y or W_pad > Num_x:
        g = F.pad(g, (0, W_pad - Num_x, 0, H_pad - Num_y), mode="constant", value=0)
    return g


def grid_to_pointcloud(
    x_grid: torch.Tensor,
    Num_y: int,
    Num_x: int,
    point_to_grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[B, C, H_pad, W_pad] -> [B, N, C], cropping padding back off.

    If point_to_grid is supplied, it maps original point indices to row-major
    grid cells and is used to return the point cloud in original dataset order.
    """
    x_crop = x_grid[:, :, :Num_y, :Num_x]
    B, C, H, W = x_crop.shape
    x = x_crop.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
    if point_to_grid is not None:
        x = x[:, point_to_grid.to(device=x.device), :]
    return x


def grid3d_to_pointcloud(x_grid: torch.Tensor, Num_z: int, Num_y: int, Num_x: int) -> torch.Tensor:
    """[B, C, D_pad, H_pad, W_pad] -> [B, N, C], cropping padding back off."""
    x_crop = x_grid[:, :, :Num_z, :Num_y, :Num_x]
    B, C, D, H, W = x_crop.shape
    return x_crop.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C).contiguous()


def build_obs_grid_mask(
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: torch.Tensor,
    n_fields: int,
    n_pts: int,
    Num_y: int,
    Num_x: int,
    H_pad: int,
    W_pad: int,
    point_to_grid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert PhyCoFlow-style sparse obs tensors into dense grid maps
    (observation values + binary mask), padded to (H_pad, W_pad).
    """
    B = obs_values.shape[0]
    device = obs_values.device
    dtype = obs_values.dtype

    value_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)
    mask_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)

    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        idx = obs_indices[b, valid].long()
        if point_to_grid is not None:
            idx = point_to_grid.to(device=device)[idx]
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        value_flat[b, fld, idx] = val
        mask_flat[b, fld, idx] = 1.0

    obs_value_grid = value_flat.reshape(B, n_fields, Num_y, Num_x)
    obs_mask_grid = mask_flat.reshape(B, n_fields, Num_y, Num_x)

    if H_pad > Num_y or W_pad > Num_x:
        obs_value_grid = F.pad(obs_value_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)
        obs_mask_grid = F.pad(obs_mask_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)

    return obs_value_grid, obs_mask_grid


def build_obs_grid_mask3d(
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: torch.Tensor,
    n_fields: int,
    n_pts: int,
    Num_z: int,
    Num_y: int,
    Num_x: int,
    D_pad: int,
    H_pad: int,
    W_pad: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert sparse obs tensors into dense 3D grid maps, padded to (D,H,W)."""
    B = obs_values.shape[0]
    device = obs_values.device
    dtype = obs_values.dtype

    value_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)
    mask_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)

    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        idx = obs_indices[b, valid].long()
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        value_flat[b, fld, idx] = val
        mask_flat[b, fld, idx] = 1.0

    obs_value_grid = value_flat.reshape(B, n_fields, Num_z, Num_y, Num_x)
    obs_mask_grid = mask_flat.reshape(B, n_fields, Num_z, Num_y, Num_x)

    if D_pad > Num_z or H_pad > Num_y or W_pad > Num_x:
        pad = (0, W_pad - Num_x, 0, H_pad - Num_y, 0, D_pad - Num_z)
        obs_value_grid = F.pad(obs_value_grid, pad, value=0)
        obs_mask_grid = F.pad(obs_mask_grid, pad, value=0)

    return obs_value_grid, obs_mask_grid


def scatter_sensors_to_nodes(obs_values, obs_mask, obs_field_ids, obs_indices,
                             B, N, n_fields, device, dtype):
    """Build per-node sparse conditioning tensors for point-token based models."""
    value = torch.zeros(B, N, n_fields, device=device, dtype=dtype)
    mask = torch.zeros(B, N, n_fields, device=device, dtype=dtype)
    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        idx = obs_indices[b, valid].long()
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        value[b, idx, fld] = val
        mask[b, idx, fld] = 1.0
    return value, mask


def nearest_sensor_fill_nodes(node_coords, obs_coords, obs_values, obs_mask,
                              obs_field_ids, n_fields, sigma=0.05, chunk=8192):
    """Voronoi-style conditioning evaluated at arbitrary node locations.

    Point-space analog of ``nearest_fill_grid`` for token subsets: with a
    random node subsample, sensors are generally not members of the node set,
    so exact scatter would silently discard the conditioning. Each node
    receives, per conditioned field, the value of its nearest valid sensor of
    that field plus a soft support weight exp(-d^2 / (2 sigma^2)) that plays
    the role of the binary mask channel (support -> 1 at a sensor, -> 0 far
    from all sensors of that field).

    node_coords [B, N, D]; obs_* as produced by build_sparse_condition.
    Returns (value [B, N, n_fields], support [B, N, n_fields]).
    """
    B, N, _ = node_coords.shape
    device, dtype = node_coords.device, node_coords.dtype
    value = torch.zeros(B, N, n_fields, device=device, dtype=dtype)
    support = torch.zeros(B, N, n_fields, device=device, dtype=dtype)
    inv2s2 = 1.0 / (2.0 * float(sigma) ** 2)
    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        oc = obs_coords[b, valid]
        ov = obs_values[b, valid, 0]
        of = obs_field_ids[b, valid].long()
        for f in of.unique().tolist():
            sel = of == f
            oc_f = oc[sel]
            ov_f = ov[sel]
            for s in range(0, N, chunk):
                d = torch.cdist(node_coords[b, s:s + chunk], oc_f)
                dmin, imin = d.min(dim=1)
                value[b, s:s + chunk, f] = ov_f[imin]
                support[b, s:s + chunk, f] = torch.exp(-dmin.pow(2) * inv2s2)
    return value, support
