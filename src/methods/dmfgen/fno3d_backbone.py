"""
3-D FNO backbone for the point-cloud rectified-flow framework.

Why this file exists
--------------------
The existing `Model.FNO` backbone is strictly 2-D: it reshapes the point cloud
to `[B, C, Num_y, Num_x]`, derives its grid permutation from `coords[0, :, :2]`
(the z axis is dropped), blurs with `F.conv2d`, and passes a 2-tuple `n_modes`
to `neuralop.models.FNO`, which fixes the spectral convolution at `rfft2`.
On the JHU 125^3 cutout (1,953,125 points, 125 unique values on *each* of the
three axes) no `(Num_x, Num_y)` pair can satisfy
`helpers.validate_regular_grid_compatibility`, so the 2-D branch exits before
the first step.  This module is the faithful 3-D extension.

Fidelity
--------
The spectral core, lifting/projection MLPs, mode truncation, skip connections
and positional embedding are **upstream `neuralop.models.FNO` verbatim**
(neuraloperator 2.0.0, Li et al. 2021 / Kossaifi et al. 2024) -- exactly as the
2-D backbone uses them.  The only change is `n_modes` becoming a 3-tuple, which
switches upstream's `SpectralConv` from `rfft2` to `rfftn` over three axes.
Everything else in this file is the same wrapper the 2-D backbone already had,
lifted from 2-D to 3-D:

  * identical input-channel contract  (4*n_fields + 1):
        current state x_t                -> C
        scalar time channel              -> 1
        normalized observed values       -> C
        support-weighted observed values -> C
        soft observation support maps    -> C
  * identical three-group conditioning rasterization,
  * identical optional Gaussian splatting of one-cell sensor impulses
    (here separable, so a k^3 kernel costs 3k taps instead of k^3),
  * identical point-order <-> row-major-grid permutation, now derived from all
    three coordinate axes instead of the first two.

Cost that is inherent to a grid-spectral backbone (documented, not a defect):
sparse point observations must be *rasterized onto the dense mesh*, and the
model must evaluate the *entire* 125^3 grid every step -- the point-cloud
interface and the Monte-Carlo query subsampling are both forfeited by
construction.  `requires_full_grid = True` makes the trainer disable query
subsampling for this backbone.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models import FNO as NeuralOpFNO
from neuralop.layers.spectral_convolution import SpectralConv

from methods.dmfgen.Model import PointCloudFFM


# ---------------------------------------------------------------------------
# FIDELITY FIX for the pinned dependency (class (i)).
#
# neuraloperator 2.0.0's `SpectralConv.forward` applies `fftshift` on BOTH the
# forward and the inverse path when `order > 1`.  On an even-length axis
# `fftshift` is its own inverse, so this is harmless for the usual 64/128-point
# benchmark grids.  On an ODD-length axis it is not, and our grid is 125^3.
# Measured on the installed version: a full-band SpectralConv with identity
# channel weights (which must be the identity map) has max relative error
# 2.0e+00 at N=125 versus 2.5e-07 at N=124.
#
# Upstream fixed this at HEAD (github.com/neuraloperator/neuraloperator,
# 00b7d86) by using `ifftshift` on the inverse path.  We restore that behaviour
# here rather than upgrading the shared virtualenv.  Everything else in the
# method is upstream 2.0.0 verbatim, taken from the installed source at import
# time so it cannot silently drift.
# ---------------------------------------------------------------------------


class SpectralConvOddSafe(SpectralConv):
    """`neuralop` SpectralConv with the upstream inverse-shift fix."""

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            # When x is real in spatial domain, the last half of the last dim is redundant.
            # See :ref:`fft_shift_explanation` for discussion of the FFT shift.
            dims_to_fft_shift = fft_dims[:-1]

        if self.order > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype
        )

        # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        # if contraction is separable, weights have shape (channels, modes_x, ...)
        # otherwise they have shape (in_channels, out_channels, modes_x, ...)
        if self.separable:
            slices_w = [slice(None)]  # channels
        else:
            slices_w = [slice(None), slice(None)]  # in_channels, out_channels
        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            # The last mode already has redundant half removed in real FFT
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        slices_w = tuple(slices_w)
        weight = self.weight[slices_w]

        ### Pick the first n_modes modes of FFT signal along each dim

        # if separable conv, weight tensor only has one channel dim
        if self.separable:
            weight_start_idx = 1
        # otherwise drop first two dims (in_channels, out_channels)
        else:
            weight_start_idx = 2

        slices_x = [slice(None), slice(None)]  # Batch_size, channels

        for all_modes, kept_modes in zip(fft_size, list(weight.shape[weight_start_idx:])):
            # After fft-shift, the 0th frequency is located at n // 2 in each direction
            # We select n_modes modes around the 0th frequency (kept at index n//2) by grabbing indices
            # n//2 - n_modes//2  to  n//2 + n_modes//2       if n_modes is even
            # n//2 - n_modes//2  to  n//2 + n_modes//2 + 1   if n_modes is odd
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2

            # this slice represents the desired indices along each dim
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)

        slices_x = tuple(slices_x)
        out_fft[slices_x] = self._contract(
            x[slices_x], weight, separable=self.separable
        )

        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            # FIDELITY FIX (restores upstream HEAD): the forward path applies
            # fftshift, so the inverse path must apply ifftshift. On ODD-length
            # axes -- our 125^3 grid -- fftshift is not its own inverse, so
            # neuralop 2.0.0 leaves a spurious circular shift in the spectrum.
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])

        if self.complex_data:
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(
                out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm
            )

        if self.bias is not None:
            x = x + self.bias

        return x



__all__ = ["FNO3D", "FNO3DFFM", "validate_regular_grid_compatibility_3d"]


def validate_regular_grid_compatibility_3d(
    dataset,
    Num_x: Optional[int],
    Num_y: Optional[int],
    Num_z: Optional[int],
    decimals: int = 6,
    atol: float = 1e-5,
) -> dict:
    """3-D analogue of `helpers.validate_regular_grid_compatibility`.

    Kept here rather than in `helpers.py` so the shared module is untouched.
    Same contract: Num_x*Num_y*Num_z must equal the point count, each axis must
    carry exactly the requested number of distinct values, and the tensor
    product must be complete (every cell present exactly once).
    """
    if Num_x is None or Num_y is None or Num_z is None:
        raise ValueError(
            "FNO3D backbone requires Num_x, Num_y and Num_z to be explicitly "
            "provided in YAML / args."
        )
    Num_x, Num_y, Num_z = int(Num_x), int(Num_y), int(Num_z)
    if min(Num_x, Num_y, Num_z) <= 0:
        raise ValueError(
            f"Num_x/Num_y/Num_z must be positive, got ({Num_x}, {Num_y}, {Num_z})."
        )

    expected = Num_x * Num_y * Num_z
    if int(dataset.num_points) != expected:
        raise ValueError(
            f"Grid mismatch: dataset has {dataset.num_points} points, but "
            f"Num_x*Num_y*Num_z = {Num_x}*{Num_y}*{Num_z} = {expected}."
        )

    coords = dataset.coords.cpu()
    scale = 10 ** decimals
    rounded = [torch.round(coords[:, d] * scale) / scale for d in range(3)]
    uniques = [torch.unique(r, sorted=True) for r in rounded]
    got = tuple(int(u.numel()) for u in uniques)
    want = (Num_x, Num_y, Num_z)
    if got != want:
        raise ValueError(
            "[x] 3-D grid compatibility check failed. Dataset unique counts are "
            f"{got} in (x, y, z), but requested (Num_x, Num_y, Num_z)={want}."
        )

    ranks = [torch.searchsorted(uniques[d], rounded[d].contiguous()) for d in range(3)]
    flat = (ranks[0].long() * Num_y + ranks[1].long()) * Num_z + ranks[2].long()
    if int(torch.unique(flat).numel()) != expected:
        raise ValueError(
            "FNO3D could not infer a complete tensor-product grid from coords: "
            "duplicate or missing (x, y, z) cells."
        )

    diagnostics = []
    for d, name in enumerate("xyz"):
        v = uniques[d]
        if v.numel() > 2:
            diffs = v[1:] - v[:-1]
            if not bool(torch.allclose(diffs, diffs.median().expand_as(diffs),
                                       atol=atol, rtol=1e-4)):
                diagnostics.append(f"unique {name} coordinates are not regularly spaced")
    row_major = bool(torch.equal(flat, torch.arange(expected, dtype=flat.dtype)))
    if not row_major:
        diagnostics.append("dataset point order is not row-major; an internal "
                           "permutation will be applied")
    return {
        "Num_x": Num_x, "Num_y": Num_y, "Num_z": Num_z,
        "row_major": row_major, "diagnostics": diagnostics,
    }


class SpectralConvOddSafeIsland(SpectralConvOddSafe):
    """`SpectralConvOddSafe` that confines the fp32 requirement to itself.

    The FFT is the only op that cannot run in bf16, so the fp32 island is placed
    here instead of around the whole FNO. The surrounding pointwise layers then
    keep bf16 tensor cores, matching the precision regime of every other method
    in the comparison.
    """

    def forward(self, x, output_shape=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float(), output_shape=output_shape)


class FNO3D(nn.Module):
    """Grid-based 3-D FNO backbone with the generalized sparse-conditioning API.

    Input contract (identical to the point-cloud backbones, so the wrapper,
    sampler and evaluation path are shared):
        t             : [B]
        x_t           : [B, N, C]
        coords        : [B, N, 3]
        obs_coords    : [B, M, 3]
        obs_values    : [B, M, 1]
        obs_mask      : [B, M]
        obs_field_ids : [B, M]
        obs_indices   : [B, M] optional; derived from obs_coords when absent
    Output:
        velocity      : [B, N, C]

    N may exceed Num_x*Num_y*Num_z when the trainer appends the contiguous
    spectral-loss block to the query set.  The FNO is evaluated once on the
    full grid and the appended tail is gathered from that same prediction, so
    the spectral term is exact and costs no extra forward pass.
    """

    def __init__(
        self,
        n_fields: int,
        Num_x: int,
        Num_y: int,
        Num_z: int,
        n_modes_x: int = 16,
        n_modes_y: int = 16,
        n_modes_z: int = 16,
        hidden_channels: int = 27,
        n_layers: int = 4,
        use_grid_positional_embedding: bool = True,
        condition_blur: bool = False,
        condition_blur_kernel: int = 5,
        condition_blur_sigma: float = 1.0,
        domain_padding: Optional[float] = None,
        spectral_fp32_island: bool = False,
    ) -> None:
        super().__init__()

        # cuFFT has no bf16 kernel. Either the WHOLE FNO runs in fp32
        # (spectral_fp32_island=False, the conservative default), or only the
        # FFT/contraction inside the spectral conv does and the pointwise
        # lifting / projection / skip / channel-MLP layers -- which dominate
        # wall time at 1.95M spatial positions -- stay in bf16 like the rest of
        # the pipeline (spectral_fp32_island=True).
        self.spectral_fp32_island = bool(spectral_fp32_island)
        self.n_fields = int(n_fields)
        self.Num_x, self.Num_y, self.Num_z = int(Num_x), int(Num_y), int(Num_z)
        self.n_grid = self.Num_x * self.Num_y * self.Num_z
        self.condition_blur = bool(condition_blur)
        self.condition_blur_kernel = int(condition_blur_kernel)
        self.condition_blur_sigma = float(condition_blur_sigma)

        if self.condition_blur_kernel < 1 or self.condition_blur_kernel % 2 == 0:
            raise ValueError(
                f"condition_blur_kernel must be a positive odd integer, got "
                f"{self.condition_blur_kernel}."
            )
        if self.condition_blur_sigma <= 0.0:
            raise ValueError(
                f"condition_blur_sigma must be > 0, got {self.condition_blur_sigma}."
            )

        # Non-persistent caches: rebuilt after device moves / checkpoint loads,
        # so old checkpoints still load with strict=True.
        self.register_buffer("_blur_kernel_cache", torch.empty(0), persistent=False)
        self.register_buffer("_grid_order_cache", torch.empty(0, dtype=torch.long),
                             persistent=False)
        self.register_buffer("_point_to_grid_cache", torch.empty(0, dtype=torch.long),
                             persistent=False)
        self._axis_values: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._identity_order: bool = False

        in_channels = 4 * self.n_fields + 1

        self.fno = NeuralOpFNO(
            # tensor layout is [B, C, Num_x, Num_y, Num_z]; upstream halves the
            # LAST entry internally because the real FFT is redundant there.
            n_modes=(int(n_modes_x), int(n_modes_y), int(n_modes_z)),
            in_channels=in_channels,
            out_channels=self.n_fields,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
            positional_embedding="grid" if use_grid_positional_embedding else None,
            domain_padding=domain_padding,
            conv_module=(SpectralConvOddSafeIsland if self.spectral_fp32_island
                         else SpectralConvOddSafe),
        )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Strip neuralop's non-tensor ``_metadata`` entry.

        ``neuralop.models.base_model.BaseModel.state_dict`` injects an
        *unprefixed* ``_metadata`` dict of init kwargs into whatever destination
        it is handed. When the FNO is a submodule that dict lands in the parent
        state dict, where it breaks any consumer that assumes tensor values
        (the trainer's EMA shadow) and shows up as an unexpected key on a
        strict load. The init kwargs are already recorded in the run config.
        """
        sd = super().state_dict(destination=destination, prefix=prefix,
                                keep_vars=keep_vars)
        if isinstance(sd, dict):
            sd.pop("_metadata", None)
        return sd

    # ------------------------------------------------------------------
    # Grid <-> point-cloud bookkeeping
    # ------------------------------------------------------------------
    def _build_axis_values(self, coords: torch.Tensor, decimals: int = 6):
        scale = 10 ** decimals
        vals = []
        for d, n in enumerate((self.Num_x, self.Num_y, self.Num_z)):
            v = torch.unique(torch.round(coords[:, d] * scale) / scale, sorted=True)
            if v.numel() != n:
                raise ValueError(
                    "FNO3D could not infer the requested grid from coords: detected "
                    f"{v.numel()} unique values on axis {d}, expected {n}."
                )
            vals.append(v.contiguous())
        return tuple(vals)

    def _coords_to_flat(self, coords: torch.Tensor, decimals: int = 6) -> torch.Tensor:
        """Map arbitrary coordinates on the mesh to row-major flat cell ids."""
        assert self._axis_values is not None
        scale = 10 ** decimals
        ranks = []
        for d in range(3):
            v = self._axis_values[d].to(coords.device)
            q = torch.round(coords[..., d] * scale) / scale
            idx = torch.searchsorted(v, q.contiguous().reshape(-1))
            idx = idx.clamp_(0, v.numel() - 1)
            # nearest of {idx-1, idx} guards against float round-trip drift
            lo = (idx - 1).clamp_min(0)
            pick = torch.where(
                (q.reshape(-1) - v[lo]).abs() <= (v[idx] - q.reshape(-1)).abs(), lo, idx)
            ranks.append(pick.reshape(q.shape).long())
        return (ranks[0] * self.Num_y + ranks[1]) * self.Num_z + ranks[2]

    def _get_grid_permutation(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """grid_order[g] = point index of grid cell g;  point_to_grid[p] = cell of point p."""
        cached_order = self._grid_order_cache
        cached_p2g = self._point_to_grid_cache
        if (cached_order.numel() == self.n_grid
                and cached_p2g.numel() == self.n_grid
                and cached_order.device == coords.device):
            return cached_order, cached_p2g

        base = coords[0, : self.n_grid, :3].detach()
        if self._axis_values is None or self._axis_values[0].device != base.device:
            self._axis_values = self._build_axis_values(base)
        point_to_grid = self._coords_to_flat(base).contiguous()
        if int(torch.unique(point_to_grid).numel()) != self.n_grid:
            raise ValueError(
                "FNO3D could not infer a complete tensor-product grid from coords. "
                "The coordinate set has duplicate or missing (x, y, z) cells."
            )
        grid_order = torch.argsort(point_to_grid).contiguous()
        self._identity_order = bool(torch.equal(
            point_to_grid, torch.arange(self.n_grid, device=point_to_grid.device)))
        self._grid_order_cache = grid_order
        self._point_to_grid_cache = point_to_grid
        return grid_order, point_to_grid

    def _pointcloud_to_grid(self, x: torch.Tensor, grid_order: torch.Tensor) -> torch.Tensor:
        """[B, n_grid, C] (point order) -> [B, C, Num_x, Num_y, Num_z]."""
        bsz, n_pts, n_ch = x.shape
        if n_pts != self.n_grid:
            raise ValueError(
                f"FNO3D expected N = Num_x*Num_y*Num_z = {self.n_grid}, got {n_pts}.")
        if not self._identity_order:
            x = x[:, grid_order, :]
        x = x.reshape(bsz, self.Num_x, self.Num_y, self.Num_z, n_ch)
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def _grid_to_pointcloud(self, x_grid: torch.Tensor,
                            point_to_grid: torch.Tensor) -> torch.Tensor:
        """[B, C, Num_x, Num_y, Num_z] -> [B, n_grid, C] (point order)."""
        bsz, n_ch = x_grid.shape[0], x_grid.shape[1]
        x = x_grid.permute(0, 2, 3, 4, 1).contiguous().reshape(bsz, self.n_grid, n_ch)
        if not self._identity_order:
            x = x[:, point_to_grid, :]
        return x

    # ------------------------------------------------------------------
    # Conditioning rasterization
    # ------------------------------------------------------------------
    def _get_blur_kernel(self, dtype, device) -> torch.Tensor:
        k = self._blur_kernel_cache
        if k.numel() > 0 and k.dtype == dtype and k.device == device:
            return k
        radius = self.condition_blur_kernel // 2
        c = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k1 = torch.exp(-0.5 * (c / self.condition_blur_sigma) ** 2)
        k1 = k1 / k1.sum().clamp_min(1e-12)
        k1 = k1.view(1, 1, -1).expand(self.n_fields, 1, -1).contiguous()
        self._blur_kernel_cache = k1
        return k1

    def _separable_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise separable 3-D Gaussian: three 1-D passes, k^3 -> 3k taps."""
        k1 = self._get_blur_kernel(x.dtype, x.device)
        ksz = self.condition_blur_kernel
        pad = ksz // 2
        g = self.n_fields
        for axis in range(3):
            shape = [1, 1, 1]
            shape[axis] = ksz
            w = k1.view(g, 1, *shape)
            p = [0, 0, 0]
            p[axis] = pad
            x = F.conv3d(x, w, padding=tuple(p), groups=g)
        return x

    def _build_condition_maps(self, obs_values, obs_mask, obs_field_ids,
                              obs_flat, dtype, device):
        """Rasterize sparse observations to dense [B, C, Nx, Ny, Nz] maps."""
        bsz = obs_values.shape[0]
        value_maps = torch.zeros(bsz, self.n_fields, self.n_grid, dtype=dtype, device=device)
        mask_maps = torch.zeros(bsz, self.n_fields, self.n_grid, dtype=dtype, device=device)

        valid = obs_mask.bool()
        if valid.any():
            b_idx = torch.arange(bsz, device=device).unsqueeze(1).expand_as(valid)[valid]
            f_idx = obs_field_ids.long()[valid]
            g_idx = obs_flat.long()[valid]
            v = obs_values[..., 0][valid].to(dtype)
            value_maps.index_put_((b_idx, f_idx, g_idx), v, accumulate=False)
            mask_maps.index_put_((b_idx, f_idx, g_idx),
                                 torch.ones_like(v), accumulate=False)

        shape = (bsz, self.n_fields, self.Num_x, self.Num_y, self.Num_z)
        value_maps = value_maps.reshape(shape)
        mask_maps = mask_maps.reshape(shape)

        if not self.condition_blur:
            # A point observation is both the normalized and the support-weighted
            # value; the binary mask carries support.
            return value_maps, value_maps, mask_maps

        blurred_mask = self._separable_blur(mask_maps)
        blurred_num = self._separable_blur(value_maps)
        blurred_norm = blurred_num / blurred_mask.clamp_min(1e-6)
        blurred_norm = torch.where(blurred_mask > 0, blurred_norm,
                                   torch.zeros_like(blurred_norm))
        return blurred_norm, blurred_num, blurred_mask

    # ------------------------------------------------------------------
    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, n_pts, _ = x_t.shape
        if n_pts < self.n_grid:
            raise ValueError(
                f"FNO3D requires the full grid: N={n_pts} < {self.n_grid}. "
                "Set requires_full_grid so the trainer disables query subsampling.")

        grid_order, point_to_grid = self._get_grid_permutation(coords)

        x_head = x_t[:, : self.n_grid, :]
        x_grid = self._pointcloud_to_grid(x_head, grid_order)
        t_map = t.view(bsz, 1, 1, 1, 1).expand(
            bsz, 1, self.Num_x, self.Num_y, self.Num_z).to(x_grid.dtype)

        # Sensor cells: prefer the explicit indices when the caller has them,
        # otherwise derive them from obs_coords (the wrapper's training_loss and
        # sample() do not forward obs_indices).
        if obs_indices is not None:
            obs_flat = point_to_grid[obs_indices.long()]
        else:
            obs_flat = self._coords_to_flat(obs_coords)

        norm_maps, weighted_maps, support_maps = self._build_condition_maps(
            obs_values=obs_values, obs_mask=obs_mask, obs_field_ids=obs_field_ids,
            obs_flat=obs_flat, dtype=x_grid.dtype, device=x_grid.device)

        fno_in = torch.cat([x_grid, t_map, norm_maps, weighted_maps, support_maps], dim=1)

        # cuFFT has no bf16/fp16 kernel: torch.fft.rfftn raises
        # "Unsupported dtype BFloat16" under the trainer's bf16 autocast. The
        # spectral core therefore always runs in fp32. This is a property of
        # grid-spectral backbones, not a choice -- the rest of the training
        # pipeline is left byte-identical to the reference protocol.
        if self.spectral_fp32_island:
            vel_grid = self.fno(fno_in)
        else:
            with torch.autocast(device_type=fno_in.device.type, enabled=False):
                vel_grid = self.fno(fno_in.float())
        vel_grid = vel_grid.to(x_t.dtype)
        vel = self._grid_to_pointcloud(vel_grid, point_to_grid)

        if n_pts > self.n_grid:
            # Spectral-loss block appended by the trainer: gather it from the
            # grid prediction we already computed (exact, no second forward).
            tail_flat = self._coords_to_flat(coords[:, self.n_grid:, :])
            flat_pred = vel_grid.permute(0, 2, 3, 4, 1).reshape(bsz, self.n_grid, -1)
            tail = torch.gather(
                flat_pred, 1, tail_flat.unsqueeze(-1).expand(-1, -1, flat_pred.shape[-1]))
            vel = torch.cat([vel, tail], dim=1)
        return vel


class FNO3DFFM(PointCloudFFM):
    """Rectified-flow wrapper for the 3-D FNO backbone.

    Deliberately inherits `training_loss`, `sample`, `simulate` and
    `target_vector_field` from `PointCloudFFM` unchanged, so the objective
    (including logit-normal t and the binned spectral loss), the ODE sampler
    and the observation-consistency handling are *identical* to the model this
    ablates against.  Only `requires_full_grid` differs, which is what makes
    the trainer feed the whole 125^3 grid instead of a Monte-Carlo subset.
    """

    def __init__(self, model: nn.Module, prior: nn.Module, sigma_min: float = 1e-4):
        super().__init__(model=model, prior=prior, sigma_min=sigma_min)
        self.requires_full_grid = True


class FNO3DRegression(FNO3DFFM):
    """Deterministic counterpart of `FNO3DFFM`: the same backbone, inputs and
    full-grid handling, trained by MSE regression on the field instead of flow
    matching, so the pair differs only in the training objective.

    The state and time channels are fed as zeros, which turns the velocity
    network into a direct map from the sensor maps to the field. `sample` is a
    single forward pass; `n_steps` and the source prior are unused.
    """

    deterministic = True

    def predict(self, coords, obs_coords, obs_values, obs_mask, obs_field_ids):
        bsz, n_pts, _ = coords.shape
        x_zero = torch.zeros(bsz, n_pts, self.model.n_fields, device=coords.device, dtype=coords.dtype)
        t_zero = torch.zeros(bsz, device=coords.device, dtype=coords.dtype)
        return self.model(t_zero, x_zero, coords, obs_coords, obs_values, obs_mask, obs_field_ids)

    def training_loss(self, x1, coords, obs_coords, obs_values, obs_mask, obs_field_ids,
                      obs_indices=None, compute_metrics=True, spectral_block_shape=None,
                      spectral_weight=0.0, spectral_bins=12, spectral_window=False):
        if spectral_weight > 0.0:
            raise ValueError("FNO3DRegression trains on MSE alone; set spectral_weight: 0.")
        loss = F.mse_loss(self.predict(coords, obs_coords, obs_values, obs_mask, obs_field_ids), x1)
        return loss, ({"loss": float(loss.detach().cpu())} if compute_metrics else {})

    @torch.no_grad()
    def sample(self, coords, obs_coords, obs_values, obs_mask, obs_field_ids, n_steps=1,
               clamp_indices=None, obs_consistency_mode="default_hard",
               obs_consistency_final_clamp=True, **unused):
        x = self.predict(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        if obs_consistency_final_clamp and obs_consistency_mode != "none" and clamp_indices is not None:
            from methods.dmfgen.obs_consistency import scatter_observed_values
            x = scatter_observed_values(x=x, obs_values=obs_values, obs_mask=obs_mask,
                                        obs_indices=clamp_indices, obs_field_ids=obs_field_ids,
                                        strength=1.0)
        return x
