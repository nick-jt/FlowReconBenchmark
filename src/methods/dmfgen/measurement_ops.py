"""Realistic measurement operators applied to sparse observation tuples.

All operators act on the padded observation buffers produced by
``helpers.build_sparse_condition``:

    obs_coords    [B, M, D]
    obs_values    [B, M, 1]
    obs_mask      [B, M]      (1.0 = valid sensor, 0.0 = padded/removed)
    obs_field_ids [B, M]      (-1 on padded slots)

Removal-type operators (occlusion, field dropout) only zero ``obs_mask``;
the model excludes masked sensors from both the attention encoder
(key_padding_mask) and the kNN gather (masked distances), so no reshaping
is required. Noise perturbs ``obs_values`` in-place on valid slots only.

Fields are assumed z-scored, so noise sigmas are in units of the
per-channel standard deviation.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch


def apply_sensor_noise(
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Additive Gaussian sensor noise with a per-sample noise level.

    A noise level sigma_b ~ U[sigma_min, sigma_max] is drawn per batch
    element, mimicking campaigns whose sensing quality varies, and the model
    is trained amortized over that range.
    """
    if sigma_max <= 0.0:
        return obs_values
    bsz = obs_values.shape[0]
    sigma = torch.empty(bsz, 1, 1, device=obs_values.device, dtype=obs_values.dtype)
    sigma.uniform_(sigma_min, sigma_max, generator=generator)
    noise = torch.randn(
        obs_values.shape, device=obs_values.device, dtype=obs_values.dtype,
        generator=generator,
    )
    return obs_values + sigma * noise * obs_mask.unsqueeze(-1)


def apply_occlusion(
    obs_coords: torch.Tensor,
    obs_mask: torch.Tensor,
    prob: float,
    kind: str = "slab",
    frac_min: float = 0.1,
    frac_max: float = 0.35,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Remove all sensors inside a random contiguous region.

    ``slab``: an axis-aligned slab of relative thickness U[frac_min, frac_max]
    along a random axis (an opaque obstruction or a dead measurement plane).
    ``ball``: a sphere with relative radius U[frac_min, frac_max] of the
    bounding-box diagonal.

    Each batch element is occluded independently with probability ``prob``.
    Returns a new obs_mask.
    """
    if prob <= 0.0:
        return obs_mask
    bsz, n_obs, dim = obs_coords.shape
    device = obs_coords.device
    new_mask = obs_mask.clone()

    lo = obs_coords.amin(dim=1)  # [B, D]  (valid slots dominate; padded zeros are inside the domain anyway)
    hi = obs_coords.amax(dim=1)
    ext = (hi - lo).clamp_min(1e-6)

    apply = torch.rand(bsz, device=device, generator=generator) < prob
    for b in range(bsz):
        if not bool(apply[b]):
            continue
        frac = float(
            torch.empty(1, device=device).uniform_(frac_min, frac_max, generator=generator)
        )
        if kind == "ball":
            center = lo[b] + torch.rand(dim, device=device, generator=generator) * ext[b]
            radius = frac * ext[b].norm()
            inside = (obs_coords[b] - center).norm(dim=-1) < radius
        else:  # slab
            axis = int(torch.randint(dim, (1,), device=device, generator=generator))
            width = frac * ext[b, axis]
            start = lo[b, axis] + torch.rand(1, device=device, generator=generator) * (
                ext[b, axis] - width
            )
            c = obs_coords[b, :, axis]
            inside = (c >= start) & (c <= start + width)
        new_mask[b] = torch.where(inside, torch.zeros_like(new_mask[b]), new_mask[b])
    return new_mask


def apply_field_dropout(
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    prob: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomly drop entire observed channels (keeping at least one).

    With probability ``prob`` per batch element, a random nonempty proper
    subset of the observed field ids is removed. Training with this operator
    amortizes the model over all observed-channel patterns, enabling
    cross-variable inference at test time (e.g. velocity-only observations
    of a thermochemical state).
    """
    if prob <= 0.0:
        return obs_mask
    bsz = obs_mask.shape[0]
    device = obs_mask.device
    new_mask = obs_mask.clone()
    for b in range(bsz):
        if float(torch.rand(1, device=device, generator=generator)) >= prob:
            continue
        present = obs_field_ids[b][obs_mask[b] > 0].unique()
        present = present[present >= 0]
        if present.numel() <= 1:
            continue
        n_keep = int(
            torch.randint(1, int(present.numel()), (1,), device=device, generator=generator)
        )
        keep = present[torch.randperm(present.numel(), device=device, generator=generator)[:n_keep]]
        keep_slot = (obs_field_ids[b].unsqueeze(-1) == keep.view(1, -1)).any(dim=-1)
        new_mask[b] = new_mask[b] * keep_slot.to(new_mask.dtype)
    return new_mask


def apply_measurement_operators(
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    cfg: Optional[Dict],
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the configured operator chain; returns (obs_values, obs_mask).

    ``cfg`` keys (all optional):
        noise_sigma_min / noise_sigma_max : float, z-score units
        occlusion_prob                    : float
        occlusion_kind                    : "slab" | "ball"
        occlusion_frac_min / _max         : float
        field_dropout_prob                : float
    """
    if not cfg:
        return obs_values, obs_mask

    obs_mask = apply_occlusion(
        obs_coords,
        obs_mask,
        prob=float(cfg.get("occlusion_prob", 0.0)),
        kind=str(cfg.get("occlusion_kind", "slab")),
        frac_min=float(cfg.get("occlusion_frac_min", 0.1)),
        frac_max=float(cfg.get("occlusion_frac_max", 0.35)),
        generator=generator,
    )
    obs_mask = apply_field_dropout(
        obs_mask,
        obs_field_ids,
        prob=float(cfg.get("field_dropout_prob", 0.0)),
        generator=generator,
    )
    obs_values = apply_sensor_noise(
        obs_values,
        obs_mask,
        sigma_min=float(cfg.get("noise_sigma_min", 0.0)),
        sigma_max=float(cfg.get("noise_sigma_max", 0.0)),
        generator=generator,
    )
    return obs_values, obs_mask
