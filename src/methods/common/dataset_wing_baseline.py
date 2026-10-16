"""Expose SHIFT-WING to the baseline machinery in model_baseline.py.

`build_dataset` there only ever built TurbulentCombustionH5Dataset, so no
baseline could see the wing at all. This wrapper presents ShiftWingDataset
through the attribute surface the baseline adapters actually touch
(`field_names`, `mean`, `std`, `num_points`, `num_fields`, `grid_shape`,
`airfoil_body_indices`) and returns per-item tensors under the keys they read.

Two wing-specific facts the adapters must respect:

  * `grid_shape` is None. The wing is an unstructured CFD mesh, so anything
    that rasterizes onto a Cartesian array (ConvAE/latent-FM, GeoFNO, the
    voxel diffusion baselines) cannot consume this and should not be pointed
    at it. Only coordinate/point-based models apply.
  * Observations come from the SURFACE POOL, not from random volume points.
    Sampling sensors uniformly inside the volume would hand a baseline
    information our model never receives and invert the comparison, so the
    pool is carried through on every item and conditioning must be built from
    it (see `build_sparse_condition_from_pool` in helpers.py).
"""

from typing import Dict

import torch
from torch.utils.data import Dataset

from methods.dmfgen.dataset_shiftwing import ShiftWingDataset


class ShiftWingBaselineDataset(Dataset):
    """ShiftWingDataset behind the interface the baseline adapters expect."""

    def __init__(self, processed_root: str, split: str = "train"):
        self.inner = ShiftWingDataset(processed_root, split=split)

        self.field_names = tuple(self.inner.field_names)
        self.num_fields = int(self.inner.num_fields)
        self.mean = self.inner.mean
        self.std = self.inner.std
        self.n_obs_field_types = int(self.inner.n_obs_field_types)

        # Unstructured: signals every grid-based code path to bail out rather
        # than silently reshaping an aircraft mesh into a cube.
        self.grid_shape = None
        self.airfoil_body_indices = None

        sample = self.inner[0]
        self.num_points = int(sample["coords"].shape[0])
        self.num_pool_points = int(sample["obs_pool_coords"].shape[0])

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s = self.inner[i]
        # coords_raw duplicates coords: the wing is already stored in
        # normalized coordinates and has no separate physical-unit array.
        return {
            "coords": s["coords"],
            "coords_raw": s["coords"],
            "fields": s["fields"],
            "obs_pool_coords": s["obs_pool_coords"],
            "obs_pool_values": s["obs_pool_values"],
            "obs_pool_field_ids": s["obs_pool_field_ids"],
            "param_coords": s["param_coords"],
            "param_values": s["param_values"],
            "param_field_ids": s["param_field_ids"],
            "time_index": torch.tensor(i, dtype=torch.long),
        }


def build_wing_dataset(cfg: dict, split: str) -> ShiftWingBaselineDataset:
    """Mirror of model_baseline.build_dataset for `dataset: shiftwing` configs."""
    shared = cfg["shared"]["data"] if "shared" in cfg else cfg
    root = shared.get("processed_root") or shared.get("data_path")
    if root is None:
        raise ValueError("shiftwing config needs data.processed_root")
    split = {"validation": "val", "test": "val"}.get(split, split)
    return ShiftWingBaselineDataset(root, split=split)
