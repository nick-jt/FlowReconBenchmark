
'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training
'''

import os
import csv
import inspect
import torch
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.gridspec as gridspec

import h5py
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Union

from methods.dmfgen.obs_consistency import build_smooth_observation_maps, observation_consistency_metrics
from protocol import held_out_subset

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

def validate_regular_grid_compatibility(
    dataset: Dataset,
    Num_x: Optional[int],
    Num_y: Optional[int],
    decimals: int = 6,
    atol: float = 1e-5,
) -> Dict[str, object]:
    """
    Validate that a point-cloud dataset can be interpreted as a Num_x by Num_y regular grid.

    Required behavior for the FNO branch:
      - Num_x and Num_y must be provided in YAML / args
      - Num_x * Num_y must match the number of points
      - the coordinate set must contain exactly Num_x unique x-values
        and Num_y unique y-values (up to rounding)
      - every (x, y) tensor-product grid cell must exist exactly once

    The FNO backbone can now apply a coordinate-derived row-major permutation
    internally, so non-row-major HDF5 point order is reported but not rejected.
    Nonuniform x/y spacing is also reported because the FNO then operates on
    index-space grid positions, while physical coordinates remain available to
    the point-cloud baselines.

    Raises ValueError if the dataset is not compatible with the requested grid.
    Returns a small diagnostics dict for launch-time logging.
    """
    if Num_x is None or Num_y is None:
        raise ValueError(
            "FNO backbone requires Num_x and Num_y to be explicitly provided in YAML / args."
        )

    Num_x = int(Num_x)
    Num_y = int(Num_y)

    if Num_x <= 0 or Num_y <= 0:
        raise ValueError(f"Num_x and Num_y must be positive, got Num_x={Num_x}, Num_y={Num_y}.")

    expected_points = Num_x * Num_y
    if int(dataset.num_points) != expected_points:
        raise ValueError(
            f"Grid mismatch: dataset has {dataset.num_points} points, but "
            f"Num_x * Num_y = {Num_x} * {Num_y} = {expected_points}."
        )

    coords = dataset.coords.cpu()
    x = torch.round(coords[:, 0] * (10 ** decimals)) / (10 ** decimals)
    y = torch.round(coords[:, 1] * (10 ** decimals)) / (10 ** decimals)

    unique_x = int(torch.unique(x).numel())
    unique_y = int(torch.unique(y).numel())

    if unique_x != Num_x or unique_y != Num_y:
        raise ValueError(
            "[x] Grid compatibility check failed. "
            f"Dataset unique counts are ({unique_x}, {unique_y}) in (x, y), "
            f"but requested (Num_x, Num_y)=({Num_x}, {Num_y})."
        )

    unique_x_vals = torch.unique(x, sorted=True)
    unique_y_vals = torch.unique(y, sorted=True)

    def _spacing_is_regular(values: torch.Tensor) -> bool:
        if values.numel() <= 2:
            return True
        diffs = values[1:] - values[:-1]
        return bool(torch.allclose(diffs, diffs.median().expand_as(diffs), atol=atol, rtol=1e-4))

    diagnostics = []
    if not _spacing_is_regular(unique_x_vals):
        diagnostics.append("unique x coordinates are not approximately regularly spaced")
    if not _spacing_is_regular(unique_y_vals):
        diagnostics.append("unique y coordinates are not approximately regularly spaced")

    x_grid = x.reshape(Num_y, Num_x)
    y_grid = y.reshape(Num_y, Num_x)

    x_first_row = x_grid[0]
    y_first_col = y_grid[:, 0]

    x_row_matches_all = bool(torch.allclose(
        x_grid,
        x_first_row.view(1, Num_x).expand(Num_y, Num_x),
        atol=atol,
        rtol=1e-4,
    ))
    y_col_matches_all = bool(torch.allclose(
        y_grid,
        y_first_col.view(Num_y, 1).expand(Num_y, Num_x),
        atol=atol,
        rtol=1e-4,
    ))
    y_constant_by_row = bool((y_grid.max(dim=1).values - y_grid.min(dim=1).values <= atol).all())
    x_constant_by_col = bool((x_grid.max(dim=0).values - x_grid.min(dim=0).values <= atol).all())

    x_row_ascending = bool(torch.allclose(x_first_row, unique_x_vals, atol=atol, rtol=1e-4))
    x_row_descending = bool(torch.allclose(x_first_row, torch.flip(unique_x_vals, dims=[0]), atol=atol, rtol=1e-4))
    y_col_ascending = bool(torch.allclose(y_first_col, unique_y_vals, atol=atol, rtol=1e-4))
    y_col_descending = bool(torch.allclose(y_first_col, torch.flip(unique_y_vals, dims=[0]), atol=atol, rtol=1e-4))

    row_major = (
        x_row_matches_all
        and y_col_matches_all
        and y_constant_by_row
        and x_constant_by_col
        and (x_row_ascending or x_row_descending)
        and (y_col_ascending or y_col_descending)
    )

    x_rank = torch.bucketize(x, unique_x_vals)
    y_rank = torch.bucketize(y, unique_y_vals)
    # bucketize can return insertion positions for exact rounded values; after
    # the unique-count check above, every rounded coordinate must map in range.
    if bool((x_rank >= Num_x).any() or (y_rank >= Num_y).any()):
        raise ValueError(
            "FNO regular-grid validation failed: coordinate values could not be "
            "mapped back to the detected unique x/y grid values."
        )
    point_to_grid = y_rank.long() * Num_x + x_rank.long()
    complete_tensor_product = int(torch.unique(point_to_grid).numel()) == expected_points

    if not complete_tensor_product:
        raise ValueError(
            "FNO regular-grid validation failed: the dataset may be a valid point cloud, "
            "but it is not a complete tensor-product grid. FNO needs exactly one "
            f"point for every ({Num_y}, {Num_x}) grid cell. Detected unique counts "
            f"are (x={unique_x}, y={unique_y}), but unique (x, y) cells are "
            f"{int(torch.unique(point_to_grid).numel())} out of {expected_points}. "
            "Use a point-cloud backbone or implement a dataset-specific gridding/interpolation step."
        )

    if not row_major:
        diagnostics.append(
            "flattened point order is not row-major; FNO will apply a coordinate-derived grid permutation"
        )

    x_diffs = unique_x_vals[1:] - unique_x_vals[:-1] if unique_x_vals.numel() > 1 else torch.ones(1)
    y_diffs = unique_y_vals[1:] - unique_y_vals[:-1] if unique_y_vals.numel() > 1 else torch.ones(1)
    order = torch.argsort(point_to_grid)
    return {
        "Num_x": Num_x,
        "Num_y": Num_y,
        "num_points": expected_points,
        "unique_x": unique_x,
        "unique_y": unique_y,
        "complete_tensor_product": True,
        "row_major": row_major,
        "requires_permutation": not row_major,
        "x_spacing_min": float(x_diffs.min().item()),
        "x_spacing_median": float(x_diffs.median().item()),
        "x_spacing_max": float(x_diffs.max().item()),
        "y_spacing_min": float(y_diffs.min().item()),
        "y_spacing_median": float(y_diffs.median().item()),
        "y_spacing_max": float(y_diffs.max().item()),
        "spacing_regular": len([d for d in diagnostics if "spaced" in d]) == 0,
        "diagnostics": diagnostics,
        "first_row_original_indices": [int(v) for v in order[: min(8, order.numel())].tolist()],
        "first_original_to_grid_indices": [int(v) for v in point_to_grid[: min(8, point_to_grid.numel())].tolist()],
    }

def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    cmin = coords.min(dim=0).values
    cmax = coords.max(dim=0).values
    scale = (cmax - cmin).clamp_min(1e-8)
    return (coords - cmin) / scale

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
        # sensor_pool: None = sensors anywhere on the point set (canonical);
        # "surface" = sensors restricted to the H5's `surface_indices` pool
        # (cylinder surface-to-field task). Exposed per sample as
        # `valid_sensor_mask` [N] bool for build_sparse_condition(valid_mask=).
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

        # Exact symmetry augmentation, train split only. JHU_AUGMENT is a
        # comma-separated list: "octahedral" (grid-safe, also given to the
        # baselines), "so3" (continuous rotation, point-cloud only), "pdatum"
        # (pressure offset), "reflect_y" (FireBench lateral reflection).
        self.augment = set()
        self._grid_shape = None
        self._aug_rng = np.random.default_rng(seed + 977)
        self._p_idx = [i for i, n in enumerate(self.field_names)
                       if str(n).lower() in ("p", "pressure")]
        requested = {s.strip() for s in os.environ.get("JHU_AUGMENT", "").split(",") if s.strip()}
        if requested and split == "train":
            side = int(round(self.num_points ** (1.0 / 3.0)))
            cubic = side ** 3 == self.num_points
            if cubic:
                self._grid_shape = (side, side, side)
            env_grid = os.environ.get("AUG_GRID_SHAPE", "")
            if env_grid:
                gs = tuple(int(v) for v in env_grid.split(","))
                if int(np.prod(gs)) == self.num_points:
                    self._grid_shape = gs
                else:
                    print(f"[dataset] AUG_GRID_SHAPE={gs} does not match "
                          f"{self.num_points} points; ignored")
            for name in requested:
                if name in ("translate",):
                    self.augment.add(name); continue
                if name in ("octahedral", "octahedral_proper", "so3") and not cubic:
                    print(f"[dataset] '{name}' needs a cubic grid "
                          f"({self.num_points} points); skipped")
                    continue
                if name == "reflect_y" and self._grid_shape is None:
                    print("[dataset] 'reflect_y' needs AUG_GRID_SHAPE; skipped")
                    continue
                if name == "pdatum" and not self._p_idx:
                    print("[dataset] 'pdatum' found no pressure channel; skipped")
                    continue
                self.augment.add(name)
            if self.augment:
                print(f"[dataset] symmetry augmentation ON (train split): "
                      f"{sorted(self.augment)}")
        # Back-compat flag used by earlier runs/checks.
        self.augment_octahedral = "octahedral" in self.augment

        self.stats_path = stats_path or str(Path(self.h5_path).with_suffix(".stats.pt"))
        self.mean, self.std = self._load_or_compute_stats(train_indices=np.sort(all_indices[:n_train]))

    def _require_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _load_or_compute_stats(self, train_indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        stats_path = Path(self.stats_path)
        if stats_path.exists():
            obj = torch.load(stats_path, map_location="cpu", weights_only=False)
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
        coords = self.coords
        if self.augment:
            import methods.dmfgen.augment_symmetry as aug
            r = self._aug_rng
            if "octahedral" in self.augment or "octahedral_proper" in self.augment:
                x = aug.octahedral_augment(
                    x, self._grid_shape, vel_idx=(0, 1, 2), rng=r,
                    proper_only="octahedral_proper" in self.augment)
            if "reflect_y" in self.augment:
                x = aug.reflect_axis_augment(x, self._grid_shape, axis=1,
                                             sign_flip_idx=(1,), rng=r)
            if "so3" in self.augment:
                coords, x = aug.so3_augment(coords, x, vel_idx=(0, 1, 2), rng=r)
            if "translate" in self.augment:
                coords = aug.translate_augment(coords, scale=1.0, rng=r)
            if "pdatum" in self.augment:
                x = aug.scalar_offset_augment(x, idx=self._p_idx, scale=1.0, rng=r)
        out = {
            "coords": coords,                       # shared tensor unless SO(3) moved the points
            "coords_raw": self.coords_raw,          # immutable shared tensor, stacked in collate
            "fields": x,
            "time_index": torch.tensor(t_idx, dtype=torch.long),
            "physical_time": self.times[t_idx],
        }
        if self.valid_sensor_mask is not None:
            out["valid_sensor_mask"] = self.valid_sensor_mask
        return out

class MetricsLogger:
    def __init__(self, base_dir: str, Demo_Num: int, timestamp: str):
        """
        Initializes the logger, creates the timestamped directory, 
        and sets up the CSV file with headers.
        """
        # Create timestamped directory: Loss_YYYYMMDD_HHMMSS
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_dir, f"Loss_DemoN{Demo_Num}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.save_dir, "losses.csv")
        self.plot_path = os.path.join(self.save_dir, "loss_curve.png")
        
        # Initialize CSV with headers
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss",
                             "epoch_time_s", "peak_gpu_mem_mb", "cumul_train_time_s"])

        self._cumul_train_time_s = 0.0

        # Store history for dynamic plotting
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

    def _append_history(self, epoch: int, train_loss: float, val_loss: float = None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def log_csv(self, epoch: int, train_loss: float, val_loss: float = None,
                epoch_time_s: float = None, peak_gpu_mem_mb: float = None):
        """Append one row to the CSV log and update in-memory history."""
        self._append_history(epoch=epoch, train_loss=train_loss, val_loss=val_loss)

        if epoch_time_s is not None:
            self._cumul_train_time_s += epoch_time_s

        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_loss if val_loss is not None else "",
                f"{epoch_time_s:.2f}" if epoch_time_s is not None else "",
                f"{peak_gpu_mem_mb:.0f}" if peak_gpu_mem_mb is not None else "",
                f"{self._cumul_train_time_s:.2f}",
            ])

    def plot_history(self):
        """Render and save the loss curve using current in-memory history."""
        if not self.epochs:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Train Loss', marker='o', color='blue', markersize=4)

        v_epochs = [e for e, v in zip(self.epochs, self.val_losses) if v is not None]
        v_losses = [v for v in self.val_losses if v is not None]

        if v_losses:
            plt.plot(v_epochs, v_losses, label='Validation Loss', marker='s', color='orange', markersize=5)

        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Conditional Point-Cloud FFM Training Progress')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.plot_path)
        plt.close()

def create_recon_dir(base_dir: str, Demo_Num: int, timestamp: str) -> str:
    """Creates a timestamped directory for saving evaluation plots."""
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, "Recon", f"demo_N{Demo_Num}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path

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

def build_sparse_condition(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    cond_fields: Union[int, Sequence[int]],
    n_obs_min: Union[int, Sequence[int]],
    n_obs_max: Union[int, Sequence[int]],
    return_counts: bool = False,
    valid_mask: Optional[torch.Tensor] = None,
):
    """
    Generalized sparse conditioning.

    valid_mask (optional, [N] or [B, N] bool): restrict sensor placement to a
    sub-pool of the point set (surface-to-field task: the wall-adjacent cell
    ring). Draw = randperm over the pool under the SAME seeding contract; the
    count m is capped at the pool size. Default None reproduces the canonical
    draw bit for bit (no extra RNG consumption).

    Args:
        coords_full: [B, N, D]
        fields_full: [B, N, C]
        cond_fields: int or list[int], e.g. 2 or [0, 2]
        n_obs_min: int or list[int], per conditioned field
        n_obs_max: int or list[int], per conditioned field

    Returns:
        obs_coords:    [B, M, D]
        obs_values:    [B, M, 1]
        obs_mask:      [B, M]
        obs_indices:   [B, M]
        obs_field_ids: [B, M]   # which field each sensor belongs to
        obs_counts:    [B]      # optional Python list when return_counts=True
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

    obs_counts: list[int] = []

    if valid_mask is not None:
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0).expand(bsz, -1)
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    for b in range(bsz):
        cursor = 0
        pool_b = (torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
                  if valid_mask is not None else None)
        for fld, nmin, nmax in zip(cond_fields, n_obs_min, n_obs_max):
            m = int(torch.randint(low=nmin, high=nmax + 1, size=(1,)).item())
            if pool_b is None:
                idx = torch.randperm(n_pts, device=device)[:m].sort().values
            else:
                m = min(m, int(pool_b.numel()))
                idx = pool_b[torch.randperm(pool_b.numel(), device=device)[:m]].sort().values

            obs_coords[b, cursor:cursor + m] = coords_full[b, idx]
            obs_values[b, cursor:cursor + m, 0] = fields_full[b, idx, fld]
            obs_mask[b, cursor:cursor + m] = 1.0
            obs_indices[b, cursor:cursor + m] = idx
            obs_field_ids[b, cursor:cursor + m] = fld

            cursor += m

        obs_counts.append(cursor)

    if return_counts:
        return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids, obs_counts

    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


def build_sparse_condition_from_pool(
    pool_coords: torch.Tensor,
    pool_values: torch.Tensor,
    pool_field_ids: torch.Tensor,
    n_obs_min: int | Sequence[int],
    n_obs_max: int | Sequence[int],
):
    """Sparse conditioning drawn from a dedicated observation pool.

    Unlike ``build_sparse_condition``, the observation candidates live on
    their own point set (e.g. a body surface) with their own observable
    channels, decoupled from the query/generation point set.

    Args:
        pool_coords:    [B, Ns, D]  candidate sensor locations
        pool_values:    [B, Ns, Cp] observable channels at those locations
        pool_field_ids: [B, Cp]     field id of each pool channel
        n_obs_min/max:  int or per-channel list; each channel c draws
                        m_c ~ U{min_c, max_c} independent locations.

    Returns the standard padded tuple
        (obs_coords [B,M,D], obs_values [B,M,1], obs_mask [B,M],
         obs_indices [B,M] (pool indices), obs_field_ids [B,M])
    """
    bsz, n_pool, coord_dim = pool_coords.shape
    n_ch = pool_values.shape[-1]
    device = pool_coords.device

    channels = list(range(n_ch))
    n_obs_min = _broadcast_per_field(n_obs_min, channels, "n_obs_min")
    n_obs_max = _broadcast_per_field(n_obs_max, channels, "n_obs_max")
    max_obs = sum(n_obs_max)

    obs_coords = torch.zeros(bsz, max_obs, coord_dim, device=device, dtype=pool_coords.dtype)
    obs_values = torch.zeros(bsz, max_obs, 1, device=device, dtype=pool_values.dtype)
    obs_mask = torch.zeros(bsz, max_obs, device=device, dtype=pool_coords.dtype)
    obs_indices = torch.zeros(bsz, max_obs, device=device, dtype=torch.long)
    obs_field_ids = torch.full((bsz, max_obs), -1, device=device, dtype=torch.long)

    for b in range(bsz):
        cursor = 0
        for c, nmin, nmax in zip(channels, n_obs_min, n_obs_max):
            m = int(torch.randint(low=nmin, high=nmax + 1, size=(1,)).item())
            idx = torch.randperm(n_pool, device=device)[:m].sort().values
            obs_coords[b, cursor:cursor + m] = pool_coords[b, idx]
            obs_values[b, cursor:cursor + m, 0] = pool_values[b, idx, c]
            obs_mask[b, cursor:cursor + m] = 1.0
            obs_indices[b, cursor:cursor + m] = idx
            obs_field_ids[b, cursor:cursor + m] = pool_field_ids[b, c]
            cursor += m

    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


def append_extra_tokens(
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_field_ids: torch.Tensor,
    extra_coords: torch.Tensor,
    extra_values: torch.Tensor,
    extra_field_ids: torch.Tensor,
):
    """Append always-valid tokens (e.g. flow-condition parameters) to an
    observation tuple. ``extra_values`` is [B, P, 1]; indices are set to 0
    and must not be used for clamping."""
    bsz, n_extra = extra_field_ids.shape
    ones = torch.ones(bsz, n_extra, device=obs_mask.device, dtype=obs_mask.dtype)
    zeros = torch.zeros(bsz, n_extra, device=obs_indices.device, dtype=obs_indices.dtype)
    return (
        torch.cat([obs_coords, extra_coords], dim=1),
        torch.cat([obs_values, extra_values], dim=1),
        torch.cat([obs_mask, ones], dim=1),
        torch.cat([obs_indices, zeros], dim=1),
        torch.cat([obs_field_ids, extra_field_ids], dim=1),
    )


def _normalized_l2(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))


def _save_single_field_plot(
    *,
    true_f: np.ndarray,
    pred_f: np.ndarray,
    coords_xy: np.ndarray,
    sensor_coords: Optional[np.ndarray],
    field_name: str,
    epoch: int,
    save_dir: str,
    file_prefix: Optional[str] = None,
    triang=None,
    metric_true_f: Optional[np.ndarray] = None,
    metric_pred_f: Optional[np.ndarray] = None,

    dpi: int = 300,
    cmap_field: str = "coolwarm", # "viridis",
    cmap_err: str = "inferno",

    contour_levels: Optional[int] = 20,
    contour_linewidth: float = 0.5,
    contour_alpha: float = 0.5,
):
    """
    Save one high-quality 3-panel plot:
        Ground truth | Reconstruction | |Error|
    while preserving the same per-field saving logic and L2 evaluation logic.
    """
    x_plot = coords_xy[:, 0]
    y_plot = coords_xy[:, 1]
    if triang is None:
        triang = mtri.Triangulation(x_plot, y_plot)

    err = np.abs(true_f - pred_f)
    # A z-midplane slice is what gets plotted for volumetric data, but the
    # reported error is the full-volume one so it stays comparable with the
    # baselines; the slice value is shown alongside it.
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

    im_true = ax_true.tricontourf(
        triang, true_f, levels=100, cmap=cmap_field,
        vmin=field_min, vmax=field_max
    )
    if contour_levels is not None:
        ax_true.tricontour(
            triang, true_f, levels=contour_levels, colors="white",
            linewidths=contour_linewidth, alpha=contour_alpha
        )

    im_pred = ax_pred.tricontourf(
        triang, pred_f, levels=100, cmap=cmap_field,
        vmin=field_min, vmax=field_max
    )
    if contour_levels is not None:
        ax_pred.tricontour(
            triang, pred_f, levels=contour_levels, colors="white",
            linewidths=contour_linewidth, alpha=contour_alpha
        )

    im_err = ax_err.tricontourf(
        triang, err, levels=100, cmap=cmap_err,
        vmin=err_min, vmax=err_max, extend="both"
    )

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

    cbar_field = fig.colorbar(im_true, cax=cax_field)
    cbar_field.set_label(field_name)

    cbar_err = fig.colorbar(im_err, cax=cax_err)
    cbar_err.set_label(f"|{field_name} - û|")

    fig.suptitle(
        f"{field_name}    |    {_l2_label}",
        y=0.96,
        fontsize=14,
    )

    prefix = file_prefix if file_prefix is not None else f"epoch_{epoch:04d}"
    filename = os.path.join(save_dir, f"{prefix}_field_{field_name}.png")
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return l2_error


def _safe_json_float_dict(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _sensor_arrays_from_payload(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recon = payload["recon"].detach().cpu()
    obs_values = payload["obs_values"].detach().cpu()
    obs_mask = payload["obs_mask"].detach().cpu()
    obs_indices = payload["obs_indices"].detach().cpu().long()
    obs_field_ids = payload["obs_field_ids"].detach().cpu().long()

    values = obs_values[..., 0] if obs_values.ndim == 3 else obs_values
    bsz, n_obs = obs_mask.shape
    batch_idx = torch.arange(bsz).view(-1, 1).expand(bsz, n_obs)
    generated = recon[batch_idx, obs_indices, obs_field_ids.clamp_min(0)]
    valid = (obs_mask > 0) & (obs_field_ids >= 0)
    return (
        values[valid].numpy(),
        generated[valid].numpy(),
        obs_field_ids[valid].numpy(),
    )


def _save_figure_all_formats(fig, base_path: str, dpi: int = 250) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base_path}.{ext}", dpi=dpi, bbox_inches="tight")


def save_sensor_parity_plot(payload: dict, senconsis_dir: str) -> None:
    observed, generated, field_ids = _sensor_arrays_from_payload(payload)
    field_names = list(payload.get("field_names", []))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if observed.size == 0:
        ax.text(0.5, 0.5, "No sensors", transform=ax.transAxes, ha="center", va="center")
    else:
        for fld in np.unique(field_ids):
            mask = field_ids == fld
            label = field_names[int(fld)] if 0 <= int(fld) < len(field_names) else f"field_{int(fld)}"
            ax.scatter(observed[mask], generated[mask], s=18, alpha=0.72, label=label)
        lo = float(np.nanmin([observed.min(), generated.min()]))
        hi = float(np.nanmax([observed.max(), generated.max()]))
        pad = 0.03 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle=":", linewidth=1.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        if len(np.unique(field_ids)) > 1:
            ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Observed sensor value")
    ax.set_ylabel("Generated value at sensor")
    ax.set_title("Sensor consistency")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure_all_formats(fig, os.path.join(senconsis_dir, "obs_consistency_parity"))
    plt.close(fig)


def save_sensor_residual_plot(payload: dict, senconsis_dir: str) -> None:
    observed, generated, field_ids = _sensor_arrays_from_payload(payload)
    residual = generated - observed
    field_names = list(payload.get("field_names", []))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    if residual.size == 0:
        ax.text(0.5, 0.5, "No sensors", transform=ax.transAxes, ha="center", va="center")
    else:
        unique_fields = np.unique(field_ids)
        data = [residual[field_ids == fld] for fld in unique_fields]
        labels = [
            field_names[int(fld)] if 0 <= int(fld) < len(field_names) else f"field_{int(fld)}"
            for fld in unique_fields
        ]
        ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.4)
    ax.set_ylabel("Generated - observed")
    ax.set_title("Sensor consistency residuals")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure_all_formats(fig, os.path.join(senconsis_dir, "obs_consistency_residuals"))
    plt.close(fig)


def save_smooth_mask_plot(
    payload: dict,
    senconsis_dir: str,
    sigma: float,
    chunk_size: int = 8192,
) -> None:
    coords = payload["coords"].detach()
    obs_coords = payload["obs_coords"].detach()
    obs_values = payload["obs_values"].detach()
    obs_mask = payload["obs_mask"].detach()
    obs_field_ids = payload["obs_field_ids"].detach()
    field_names = list(payload.get("field_names", []))
    n_fields = int(payload["recon"].shape[-1])
    value_map, mask_map = build_smooth_observation_maps(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_fields=n_fields,
        sigma=sigma,
        chunk_size=chunk_size,
    )
    coords_xy = np.asarray(payload["coords_xy"])
    mask_np = mask_map[0].detach().cpu().numpy()
    value_np = value_map[0].detach().cpu().numpy()
    # z-collapse fix (audit 2026-08-29): payload["coords_xy"] is the FULL
    # volume projected onto (x, y). For 3-D data that superposes every z-plane
    # in the triangulation, rendering a smear instead of a cross-section.
    # Take the z-midplane slice, same pattern as helpers_baseline.midplane_slice.
    coords_full = payload["coords"].detach().cpu().numpy()
    if coords_full.ndim == 3:
        coords_full = coords_full[0]
    if (coords_full.shape[0] == coords_xy.shape[0]
            and coords_full.shape[-1] >= 3
            and bool(np.ptp(coords_full[:, 2]) > 1e-6)):
        z_vals = coords_full[:, 2]
        z_levels = np.unique(z_vals)
        slice_mask = z_vals == z_levels[len(z_levels) // 2]
        coords_xy = coords_xy[slice_mask]
        mask_np = mask_np[slice_mask]
        value_np = value_np[slice_mask]
    triang = mtri.Triangulation(coords_xy[:, 0], coords_xy[:, 1])

    for c in range(n_fields):
        if not np.any(mask_np[:, c] > 0):
            continue
        name = field_names[c] if c < len(field_names) else f"field_{c}"
        for kind, arr, cmap in (
            ("mask", mask_np[:, c], "viridis"),
            ("interp", value_np[:, c], "coolwarm"),
        ):
            fig, ax = plt.subplots(figsize=(6.2, 4.8))
            im = ax.tricontourf(triang, arr, levels=80, cmap=cmap)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Sensor consistency {kind}: {name}")
            fig.colorbar(im, ax=ax, shrink=0.75)
            fig.tight_layout()
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
            _save_figure_all_formats(fig, os.path.join(senconsis_dir, f"obs_consistency_{kind}_{safe_name}"))
            plt.close(fig)


def save_obs_consistency_comparison(rows: list[dict], senconsis_dir: str) -> None:
    os.makedirs(senconsis_dir, exist_ok=True)
    csv_path = os.path.join(senconsis_dir, "obs_consistency_comparison.csv")
    json_path = os.path.join(senconsis_dir, "obs_consistency_comparison.json")
    fieldnames = ["mode", "relative_l2", "obs_rel_l2_SenConsis", "obs_count_SenConsis_total"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    modes = [row["mode"] for row in rows]
    sensor_vals = [row.get("obs_rel_l2_SenConsis", np.nan) for row in rows]
    full_vals = [row.get("relative_l2", np.nan) for row in rows]
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.36
    ax.bar(x - width / 2, sensor_vals, width=width, label="obs_rel_l2_SenConsis")
    if np.any(np.isfinite(full_vals)):
        ax.bar(x + width / 2, full_vals, width=width, label="relative_l2")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Relative L2")
    ax.set_title("Sensor consistency comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure_all_formats(fig, os.path.join(senconsis_dir, "obs_consistency_comparison_bar"))
    plt.close(fig)

@torch.no_grad()
def reconstruct_snapshot(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    snapshot_index: int,
    cond_fields: Union[int, Sequence[int]] = (2,),
    n_obs_list: Union[int, Sequence[int]] = 256,
    n_steps: int = 100,
    ode_solver: Optional[str] = None,
):
    """
    Reconstruct one snapshot under arbitrary sparse conditioning.

    This utility is intended for evaluation scripts (e.g. physical coherence
    auditing) so they can reuse the exact same reconstruction logic without
    duplicating code from visualize_reconstruction().

    Returns normalized truth and reconstruction tensors so downstream metrics
    can decide whether to work in normalized or physical-unit space.
    """
    import inspect

    model.eval()

    cond_fields = _to_int_list(cond_fields)
    n_obs_list = _broadcast_per_field(n_obs_list, cond_fields, "n_obs_list")

    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)   # [1, N, D]
    truth = sample["fields"].unsqueeze(0).to(device)    # [1, N, C]

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs_list,
        n_obs_max=n_obs_list,   # exact sensor counts for evaluation
    )

    # Compatibility wrapper: support both newer generalized checkpoints and
    # older single-field checkpoints if needed.
    sig = inspect.signature(model.sample)
    sample_kwargs = dict(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        n_steps=n_steps,
        clamp_indices=obs_indices,
    )

    if "obs_field_ids" in sig.parameters:
        sample_kwargs["obs_field_ids"] = obs_field_ids
    elif "cond_field_idx" in sig.parameters:
        unique = torch.unique(obs_field_ids[obs_mask.bool()])
        if unique.numel() != 1:
            raise ValueError(
                "Loaded checkpoint expects single-field conditioning (cond_field_idx), "
                "but reconstruct_snapshot received multiple conditioned fields."
            )
        sample_kwargs["cond_field_idx"] = unique.view(1).to(obs_field_ids.device)

    if "ode_solver" in sig.parameters and ode_solver is not None:
        sample_kwargs["ode_solver"] = ode_solver

    recon = model.sample(**sample_kwargs)

    return {
        "coords": coords,
        "truth": truth,                  # normalized truth
        "recon": recon,                  # normalized reconstruction
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": obs_indices,
        "obs_field_ids": obs_field_ids,
    }

@torch.no_grad()
def visualize_reconstruction(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    epoch: int,
    device: torch.device,
    save_dir: str,
    cond_fields: Union[int, Sequence[int]] = (2,),
    n_obs: Union[int, Sequence[int]] = 256,
    n_steps: int = 32,
    ode_solver: Optional[str] = None,
    snapshot_index: int = 0,
    file_tag: Optional[str] = None,
    save_metrics_json: bool = True,
    return_payload: bool = False,
    obs_consistency_mode: str = "default_hard",
    obs_consistency_strength: float = 1.0,
    obs_consistency_sigma: float = 0.05,
    obs_consistency_schedule_power: float = 2.0,
    obs_consistency_final_clamp: bool = True,
    save_obs_consistency_plots: bool = False,
    obs_consistency_compare_modes: Optional[Sequence[str]] = None,
    obs_consistency_chunk_size: int = 8192,
    sparse_condition: Optional[dict] = None,
    field_names: Optional[Sequence[str]] = None,
):
    """
    Reconstruct full fields from arbitrary sparse sensors and save improved plots.

    Example:
        cond_fields=[0, 2], n_obs=[128, 256]

    Returns
    -------
    metrics : dict
        Per-field normalized L2 errors.
    """
    import inspect

    model.eval()

    cond_fields = _to_int_list(cond_fields)
    n_obs = _broadcast_per_field(n_obs, cond_fields, "n_obs")

    sample = dataset[snapshot_index]

    # Normalized coordinates go into the model.
    coords = sample["coords"].unsqueeze(0).to(device)   # [1, N, D]
    # Original coordinates are used only for plotting.
    coords_raw = sample["coords_raw"].unsqueeze(0)

    truth = sample["fields"].unsqueeze(0).to(device)    # [1, N, C]

    if sparse_condition is None:
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=truth,
            cond_fields=cond_fields,
            n_obs_min=n_obs,
            n_obs_max=n_obs,   # exact sensor counts for visualization
        )
    else:
        obs_coords = sparse_condition["obs_coords"].to(device)
        obs_values = sparse_condition["obs_values"].to(device)
        obs_mask = sparse_condition["obs_mask"].to(device)
        obs_indices = sparse_condition["obs_indices"].to(device)
        obs_field_ids = sparse_condition["obs_field_ids"].to(device)

    sample_kwargs = dict(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=n_steps,
        clamp_indices=obs_indices,
        obs_consistency_mode=obs_consistency_mode,
        obs_consistency_strength=obs_consistency_strength,
        obs_consistency_sigma=obs_consistency_sigma,
        obs_consistency_schedule_power=obs_consistency_schedule_power,
        obs_consistency_final_clamp=obs_consistency_final_clamp,
        obs_consistency_chunk_size=obs_consistency_chunk_size,
    )
    sig = inspect.signature(model.sample)
    if "ode_solver" in sig.parameters and ode_solver is not None:
        sample_kwargs["ode_solver"] = ode_solver

    recon = model.sample(**sample_kwargs)

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)

    recon_phys = recon_phys[0].cpu().numpy()   # [N, C]
    truth_phys = truth_phys[0].cpu().numpy()   # [N, C]

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()

    # coords_np = coords[0].cpu().numpy()
    coords_np = coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]

    # For 3D volumetric datasets (e.g. JHU 125³), plotting all points projected
    # onto XY superposes all Z-planes and produces a smeared blob. Extract the
    # Z-midplane slice so the visualization matches the baseline's cross-section.
    is_3d = coords_np.shape[-1] >= 3 and bool(np.ptp(coords_np[:, 2]) > 1e-6)
    if is_3d:
        z_vals = coords_np[:, 2]
        _z_levels = np.unique(z_vals)
        slice_mask = z_vals == _z_levels[len(_z_levels) // 2]
        coords_xy_plot = coords_xy[slice_mask]
        slice_triang = mtri.Triangulation(coords_xy_plot[:, 0], coords_xy_plot[:, 1])
    else:
        slice_mask = slice(None)
        coords_xy_plot = coords_xy
        slice_triang = None

    field_names = tuple(field_names if field_names is not None else getattr(dataset, "field_names", FIELD_NAMES))
    metrics = {}

    for c, name in enumerate(field_names):
        true_f = truth_phys[:, c][slice_mask]
        pred_f = recon_phys[:, c][slice_mask]

        # Only overlay sensors belonging to this field.
        sensor_coords = None
        field_sensor_mask = (obs_field_ids_cpu == c)
        if np.any(field_sensor_mask):
            sensor_indices = obs_indices_cpu[field_sensor_mask]
            if is_3d:
                on_slice = slice_mask[sensor_indices]
                sensor_coords = coords_xy[sensor_indices[on_slice]] if np.any(on_slice) else None
            else:
                sensor_coords = coords_xy[sensor_indices]

        l2_error = _save_single_field_plot(
            true_f=true_f,
            pred_f=pred_f,
            metric_true_f=truth_phys[:, c],
            metric_pred_f=recon_phys[:, c],
            coords_xy=coords_xy_plot,
            sensor_coords=sensor_coords,
            field_name=name,
            epoch=epoch,
            save_dir=save_dir,
            file_prefix=file_tag,
            triang=slice_triang,
        )
        metrics[name] = l2_error

    # SenConsis = sensor consistency between generated values and sparse
    # observed values. Added scalar metrics are relative L2 at sensor entries.
    obs_metrics = observation_consistency_metrics(
        recon=recon,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_indices=obs_indices,
        obs_field_ids=obs_field_ids,
        field_names=field_names,
    )
    metrics.update(obs_metrics)

    if save_metrics_json:
        prefix = file_tag if file_tag is not None else f"epoch_{epoch:04d}"
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "n_steps": int(n_steps),
            "ode_solver": ode_solver,
            "obs_consistency_mode": obs_consistency_mode,
            "ode_solver": ode_solver,
            "metrics": metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

    payload = {
        "coords": coords.detach().cpu(),
        "coords_xy": coords_xy,
        "truth": truth.detach().cpu(),
        "target": truth.detach().cpu(),
        "recon": recon.detach().cpu(),
        "truth_phys": truth_phys,
        "recon_phys": recon_phys,
        "obs_coords": obs_coords.detach().cpu(),
        "obs_values": obs_values.detach().cpu(),
        "obs_mask": obs_mask.detach().cpu(),
        "obs_indices": obs_indices.detach().cpu(),
        "obs_field_ids": obs_field_ids.detach().cpu(),
        "obs_indices_valid": obs_indices_cpu,
        "obs_field_ids_valid": obs_field_ids_cpu,
        "field_names": list(field_names),
        "snapshot_index": int(snapshot_index),
        "cond_fields": [int(v) for v in cond_fields],
        "n_obs": [int(v) for v in n_obs],
        "n_steps": int(n_steps),
        "ode_solver": ode_solver,
        "obs_consistency_mode": obs_consistency_mode,
    }

    if save_obs_consistency_plots:
        senconsis_dir = os.path.join(save_dir, "SenConsis")
        os.makedirs(senconsis_dir, exist_ok=True)
        with open(os.path.join(senconsis_dir, "obs_consistency_metrics.json"), "w") as f:
            json.dump(_safe_json_float_dict(obs_metrics), f, indent=2)
        save_sensor_parity_plot(payload, senconsis_dir)
        save_sensor_residual_plot(payload, senconsis_dir)
        if obs_consistency_mode == "endpoint_smooth":
            save_smooth_mask_plot(
                payload,
                senconsis_dir,
                sigma=obs_consistency_sigma,
                chunk_size=obs_consistency_chunk_size,
            )

    if return_payload:
        return metrics, payload

    return metrics

# -------------------------------------------
