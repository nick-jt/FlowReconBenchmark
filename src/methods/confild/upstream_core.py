from __future__ import annotations
from paths import DATA_DIR, THIRD_PARTY_DIR

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import h5py
import numpy as np
import torch


DEFAULT_CONFILD_ROOT = Path(
    f"{THIRD_PARTY_DIR}/CoNFiLD"
)
DEFAULT_DATA = Path(
    f"{DATA_DIR}/JHU_4cubes_stride100.h5"
)

PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_upstream_decoder(confild_root: str | Path = DEFAULT_CONFILD_ROOT):
    root = str(Path(confild_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ConditionalNeuralField.cnf.nf_networks import SIRENAutodecoder_film

    return SIRENAutodecoder_film


def normalize_coords(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lo = coords.amin(dim=0)
    hi = coords.amax(dim=0)
    normalized = (coords - lo) / (hi - lo).clamp_min(1e-12) * 2.0 - 1.0
    return normalized, lo, hi


@dataclass
class FieldStatistics:
    """Upstream field scaling plus benchmark-wide z-score statistics."""

    # [1, channel] -- GLOBAL over all points and all training snapshots.
    #
    # Upstream reduces only over time (Normalizer_ts(dim=0) on [t, N, c]),
    # giving PER-POINT statistics. That is right for their cases, where the
    # flow is pinned by walls and geometry so per-point structure is a property
    # of the domain and transfers to held-out times. It does not hold here.
    # Measured on this dataset: the per-point range reproduces within a single
    # cube (even/odd split-half r = 0.42-0.59, against a positive control of
    # 0.80-0.96 for the per-point time mean) but does NOT transfer between
    # cubes (r = -0.26..+0.39, scattered about zero, across all six cube
    # pairings including every pairing with the held-out cube 3). So per-point
    # statistics fitted on cubes 0-2 encode structure specific to those
    # realisations and carry no information about cube 3; using them to
    # denormalise cube-3 reconstructions injects a spatially structured
    # multiplicative error of 11-21% (CV of the per-point range is 0.10-0.19)
    # that is uncorrelated with anything real in the held-out data.
    #
    # Class (ii) required adaptation. It also makes the normalisation exactly
    # invariant under the mandated octahedral group, which removes an entire
    # class of index-transform defects.
    minimum: torch.Tensor  # [1, channel]
    maximum: torch.Tensor  # [1, channel]
    benchmark_mean: torch.Tensor  # [channel]
    benchmark_std: torch.Tensor  # [channel]

    def _at(self, point_indices):
        """Global statistics broadcast over points; point_indices is inert.

        Kept in the signature so older per-point checkpoints still load and
        normalise correctly if one is ever read back.
        """
        if point_indices is None or self.minimum.shape[0] == 1:
            return self.minimum, self.maximum
        return self.minimum[point_indices], self.maximum[point_indices]

    def normalize(
        self, fields: torch.Tensor, point_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        minimum, maximum = self._at(point_indices)
        return (fields - minimum) / (maximum - minimum).clamp_min(1e-12) * 2.0 - 1.0

    def denormalize(
        self, fields: torch.Tensor, point_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        minimum, maximum = self._at(point_indices)
        return (fields + 1.0) * 0.5 * (maximum - minimum) + minimum

    def benchmark_normalize(self, fields: torch.Tensor) -> torch.Tensor:
        return (fields - self.benchmark_mean) / self.benchmark_std.clamp_min(1e-12)

    def cpu(self) -> "FieldStatistics":
        return FieldStatistics(
            minimum=self.minimum.cpu(),
            maximum=self.maximum.cpu(),
            benchmark_mean=self.benchmark_mean.cpu(),
            benchmark_std=self.benchmark_std.cpu(),
        )

    def to(self, device: torch.device | str) -> "FieldStatistics":
        return FieldStatistics(
            minimum=self.minimum.to(device),
            maximum=self.maximum.to(device),
            benchmark_mean=self.benchmark_mean.to(device),
            benchmark_std=self.benchmark_std.to(device),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "minimum": self.minimum.cpu(),
            "maximum": self.maximum.cpu(),
            "benchmark_mean": self.benchmark_mean.cpu(),
            "benchmark_std": self.benchmark_std.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "FieldStatistics":
        return cls(**{key: value.float() for key, value in state.items()})


class PackedJHUCubes:
    """Read the packed four-cube HDF5 without reinterpreting its leading axes."""

    def __init__(
        self,
        path: str | Path,
        train_count: int = 150,
        val_start: int = 150,
        val_count: int = 50,
    ) -> None:
        self.path = Path(path)
        self.train_indices = np.arange(train_count, dtype=np.int64)
        self.val_indices = np.arange(val_start, val_start + val_count, dtype=np.int64)
        with h5py.File(self.path, "r") as handle:
            shape = tuple(handle["fields"].shape)
            if len(shape) != 6 or shape[0] != 1 or shape[3:5] != (1, 1):
                raise ValueError(f"Unexpected packed JHU field shape: {shape}")
            self.num_snapshots = int(shape[1])
            self.num_points = int(shape[2])
            self.num_fields = int(shape[-1])
            raw_coords = torch.from_numpy(
                handle["coordinates"][:, 0, 0, :].astype(np.float32)
            )
            self.layout = str(handle.attrs.get("layout", ""))
        if self.val_indices[-1] >= self.num_snapshots:
            raise ValueError("Requested validation range exceeds the HDF5 snapshot count")
        self.coords, self.coord_min, self.coord_max = normalize_coords(raw_coords)
        side = round(self.num_points ** (1.0 / 3.0))
        if side**3 != self.num_points:
            raise ValueError(f"Expected a cubic grid, got {self.num_points} points")
        self.grid_shape = (side, side, side)

    def read_snapshot(self, index: int) -> torch.Tensor:
        with h5py.File(self.path, "r") as handle:
            array = handle["fields"][0, int(index), :, 0, 0, :].astype(np.float32)
        return torch.from_numpy(array)

    def snapshots(self, indices: Sequence[int]) -> Iterator[torch.Tensor]:
        with h5py.File(self.path, "r") as handle:
            fields = handle["fields"]
            for index in indices:
                yield torch.from_numpy(
                    fields[0, int(index), :, 0, 0, :].astype(np.float32)
                )

    def compute_train_statistics(self) -> FieldStatistics:
        minimum = None
        maximum = None
        total = torch.zeros(self.num_fields, dtype=torch.float64)
        total_sq = torch.zeros(self.num_fields, dtype=torch.float64)
        count = 0
        for fields in self.snapshots(self.train_indices):
            # Global per-channel extremes over BOTH points and snapshots.
            batch_min, batch_max = fields.amin(dim=0), fields.amax(dim=0)
            minimum = batch_min if minimum is None else torch.minimum(minimum, batch_min)
            maximum = batch_max if maximum is None else torch.maximum(maximum, batch_max)
            total += fields.sum(dim=0, dtype=torch.float64)
            total_sq += fields.double().square().sum(dim=0)
            count += fields.shape[0]
        assert minimum is not None and maximum is not None
        # Match TurbulentCombustionH5Dataset exactly: it stores the mean as
        # float32 before using it in the population-variance calculation.
        mean = (total / count).float()
        variance = (total_sq / count - mean.double().square()).clamp_min(1e-12).float()
        return FieldStatistics(
            minimum=minimum.reshape(1, -1).float(),
            maximum=maximum.reshape(1, -1).float(),
            benchmark_mean=mean,
            benchmark_std=variance.sqrt(),
        )


def octahedral_transform(
    fields: torch.Tensor,
    grid_shape: Sequence[int],
    group_index: int,
    velocity_channels: Sequence[int] = (0, 1, 2),
) -> torch.Tensor:
    """Apply one of 48 signed axis permutations to a physical vector field."""
    if not 0 <= group_index < 48:
        raise ValueError(f"group_index must be in [0, 48), got {group_index}")
    if len(grid_shape) != 3 or int(np.prod(grid_shape)) != fields.shape[0]:
        raise ValueError("grid_shape does not match flattened field size")
    permutation = PERMUTATIONS[group_index // 8]
    flip_code = group_index % 8
    flip_axes = [axis for axis in range(3) if (flip_code >> axis) & 1]
    channels = fields.shape[-1]

    transformed = fields.reshape(*grid_shape, channels).permute(
        permutation[0], permutation[1], permutation[2], 3
    )
    component_order = list(range(channels))
    velocity_channels = list(velocity_channels)
    for axis in range(3):
        component_order[velocity_channels[axis]] = velocity_channels[permutation[axis]]
    transformed = transformed[..., component_order]
    if flip_axes:
        transformed = torch.flip(transformed, dims=flip_axes)
        signs = torch.ones(channels, dtype=transformed.dtype, device=transformed.device)
        for axis in flip_axes:
            signs[velocity_channels[axis]] = -1.0
        transformed = transformed * signs
    return transformed.reshape(-1, channels).contiguous()


def octahedral_gather(
    fields: torch.Tensor,
    grid_shape: Sequence[int],
    group_index: int,
    point_ids: torch.Tensor,
    velocity_channels: Sequence[int] = (0, 1, 2),
) -> torch.Tensor:
    """`octahedral_transform(fields, grid_shape, g)[point_ids]` without materialising.

    Exactly equivalent to transforming the whole 125^3 field and then indexing,
    but costs O(len(point_ids)) instead of O(1.95M). The group action is a
    relabelling of grid points plus a signed permutation of the velocity
    components, so it can be pushed onto the sampled indices instead. This is
    what makes the mandated 48-element octahedral family affordable: the naive
    path transforms a 7.8M-element tensor per item and then throws away 96.6%
    of it.
    """
    if not 0 <= group_index < 48:
        raise ValueError(f"group_index must be in [0, 48), got {group_index}")
    grid = [int(value) for value in grid_shape]
    permutation = PERMUTATIONS[group_index // 8]
    flip_code = group_index % 8
    flips = [(flip_code >> axis) & 1 for axis in range(3)]
    channels = fields.shape[-1]

    out_shape = [grid[permutation[axis]] for axis in range(3)]
    point_ids = point_ids.long()
    index = [
        torch.div(point_ids, out_shape[2] * out_shape[1], rounding_mode="floor"),
        torch.div(point_ids, out_shape[2], rounding_mode="floor") % out_shape[1],
        point_ids % out_shape[2],
    ]
    for axis in range(3):
        if flips[axis]:
            index[axis] = (out_shape[axis] - 1) - index[axis]
    source = [None, None, None]
    for axis in range(3):
        source[permutation[axis]] = index[axis]
    flat = (source[0] * grid[1] + source[1]) * grid[2] + source[2]

    component_order = list(range(channels))
    velocity_channels = list(velocity_channels)
    for axis in range(3):
        component_order[velocity_channels[axis]] = velocity_channels[permutation[axis]]
    gathered = fields[flat][:, component_order]
    flip_axes = [axis for axis in range(3) if flips[axis]]
    if flip_axes:
        signs = torch.ones(channels, dtype=gathered.dtype)
        for axis in flip_axes:
            signs[velocity_channels[axis]] = -1.0
        gathered = gathered * signs
    return gathered


def item_to_snapshot_group(item_id: int, n_group: int) -> tuple[int, int]:
    if n_group not in (1, 8, 48):
        raise ValueError("n_group must be one of 1, 8, or 48")
    return divmod(int(item_id), n_group)


def batches(permutation: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, permutation.numel(), batch_size):
        yield permutation[start : start + batch_size]


def decoder_gradient_is_clear(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in model.parameters())


def build_latent_windows(
    latents: torch.Tensor,
    n_snapshots: int,
    n_groups: int,
    cube_length: int,
    window_length: int,
    stride: int = 1,
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """Arrange lumped codes as upstream [time, latent] diffusion images."""
    if latents.shape[0] != n_snapshots * n_groups:
        raise ValueError("Latent table shape does not match snapshots x groups")
    if n_snapshots % cube_length:
        raise ValueError("n_snapshots must be divisible by cube_length")
    if not 1 <= window_length <= cube_length:
        raise ValueError("window_length must be between 1 and cube_length")
    table = latents.reshape(n_snapshots, n_groups, latents.shape[-1])
    windows = []
    manifest = []
    for cube in range(n_snapshots // cube_length):
        cube_start = cube * cube_length
        for group in range(n_groups):
            for offset in range(0, cube_length - window_length + 1, stride):
                windows.append(
                    table[
                        cube_start + offset : cube_start + offset + window_length,
                        group,
                    ]
                )
                manifest.append(
                    {
                        "cube": cube,
                        "group": group,
                        "start_snapshot": cube_start + offset,
                        "window_length": window_length,
                    }
                )
    return torch.stack(windows).unsqueeze(1), manifest


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def config_dict(namespace) -> dict:
    result = vars(namespace).copy()
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
    return result
