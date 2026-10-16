"""Exact symmetry augmentations for the three datasets.

Every transform here is a symmetry the underlying physics genuinely possesses,
so an augmented sample is another exact draw from the same distribution -- no
approximation, unlike cropping-and-rescaling (which changes the Reynolds
number) or additive noise.

Available symmetries by dataset:

  JHU isotropic turbulence (homogeneous + isotropic, cubic grid)
    - octahedral: the 48 signed axis permutations. Grid-safe, so grid-based
      baselines can use it too; this is the one we apply to every method for
      a matched comparison.
    - SO(3): arbitrary continuous rotation. Valid only because the model works
      on point clouds in ambient space -- rotating a voxel grid would require
      interpolation and smear the spectrum, so grid-based methods cannot use
      this without corrupting their data. Infinite augmentations, no
      resampling error.
    - pressure datum: incompressible pressure is defined up to an additive
      constant, so shifting the p channel is exact.

  FireBench wildfire LES (mean wind along x, line ignition across y)
    - lateral reflection y -> -y with v -> -v. The only valid spatial
      symmetry: the ground plane breaks z, and the wind direction breaks x.

  SHIFT-WING (left-right symmetric airframe at zero sideslip)
    - spanwise reflection y -> -y with Uy -> -Uy (and tau_y -> -tau_y on the
      surface pool).

All functions operate on [N, F] field tensors plus, where the transform moves
points, an [N, D] coordinate tensor.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

_PERMS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
)


# Parity of each permutation in _PERMS (+1 even, -1 odd).
_PERM_PARITY: Tuple[int, ...] = (1, -1, -1, 1, 1, -1)


def octahedral_augment(
    fields: torch.Tensor,
    grid_shape: Sequence[int],
    vel_idx: Sequence[int] = (0, 1, 2),
    rng: Optional[np.random.Generator] = None,
    proper_only: bool = False,
) -> torch.Tensor:
    """Random element of the octahedral group, applied on-grid.

    Coordinates are left untouched: the field is reshaped to its cubic grid,
    the spatial axes are permuted/flipped, and the same signed permutation is
    applied to the velocity components, so the result stays consistent with the
    original coordinate array.

    With `proper_only`, the 24 rotations (determinant +1) are used instead of
    the full 48-element group. The improper elements are reflections, and
    mirror symmetry is only a symmetry of the flow when the mean helicity
    vanishes -- helicity is a pseudo-scalar and changes sign under reflection.
    Restricting to proper rotations is the safe choice when that is unverified.
    """
    if rng is None:
        rng = np.random.default_rng()
    gx, gy, gz = (int(s) for s in grid_shape)
    if not (gx == gy == gz):
        raise ValueError(f"octahedral augmentation needs a cubic grid, got {grid_shape}")
    n_f = fields.shape[-1]
    vel = [int(v) for v in vel_idx]
    if len(vel) != 3:
        raise ValueError(f"vel_idx must name three components, got {vel_idx}")

    pi = int(rng.integers(len(_PERMS)))
    perm = _PERMS[pi]
    flips = rng.integers(0, 2, size=3).astype(bool)
    if proper_only:
        # det = parity(perm) * (-1)^n_flips must be +1, so n_flips parity has
        # to match the permutation's. Toggling one bit maps the 8 draws
        # uniformly onto the 4 admissible flip patterns.
        want_odd = _PERM_PARITY[pi] < 0
        if bool(flips.sum() % 2) != want_odd:
            flips[0] = ~flips[0]

    g = fields.reshape(gx, gy, gz, n_f)
    g = g.permute(perm[0], perm[1], perm[2], 3)
    comp = list(range(n_f))
    for a in range(3):
        comp[vel[a]] = vel[perm[a]]
    g = g[..., comp]

    flip_dims = [a for a in range(3) if flips[a]]
    if flip_dims:
        g = torch.flip(g, dims=flip_dims)
        sign = torch.ones(n_f, dtype=g.dtype, device=g.device)
        for a in flip_dims:
            sign[vel[a]] = -1.0
        g = g * sign

    return g.reshape(gx * gy * gz, n_f).contiguous()


def translate_augment(
    coords: torch.Tensor,
    scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Shift the whole point cloud by a random offset.

    Exact for a homogeneous flow, and the cheapest way to remove absolute
    position as a usable signal: a model that can read off where in the domain
    a query sits can memorize "field 37 has value v at (0.3, 0.7, 0.2)", which
    is retrieval rather than inference. Sensors and queries move together, so
    the reconstruction problem is unchanged.
    """
    if rng is None:
        rng = np.random.default_rng()
    off = torch.from_numpy(rng.uniform(-scale, scale, size=coords.shape[-1])).to(coords.dtype)
    return coords + off


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform sample from SO(3) via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))          # fix the QR sign ambiguity
    if np.linalg.det(q) < 0:             # reflect to land in SO(3), not O(3)
        q[:, 0] = -q[:, 0]
    return q


def so3_augment(
    coords: torch.Tensor,
    fields: torch.Tensor,
    vel_idx: Sequence[int] = (0, 1, 2),
    center: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate a point cloud and its vector field by a random element of SO(3).

    Exact for homogeneous isotropic turbulence and, unlike a grid rotation,
    interpolation-free: the sample points move with the field, so the values
    are the original DNS values at their (rotated) locations. Points are
    rotated about `center` so the cloud stays roughly in place; corners of the
    cube leave the unit box, which the coordinate encodings handle since they
    are defined on all of R^3.
    """
    if rng is None:
        rng = np.random.default_rng()
    vel = [int(v) for v in vel_idx]
    if len(vel) != 3:
        raise ValueError(f"vel_idx must name three components, got {vel_idx}")

    R = torch.from_numpy(_random_rotation(rng)).to(dtype=coords.dtype)
    new_coords = (coords - center) @ R.T + center

    new_fields = fields.clone()
    u = fields[:, vel].to(R.dtype)
    new_fields[:, vel] = (u @ R.T).to(fields.dtype)
    return new_coords.contiguous(), new_fields


def scalar_offset_augment(
    fields: torch.Tensor,
    idx: Sequence[int],
    scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Shift channels whose absolute datum is physically arbitrary.

    For incompressible flow only the pressure gradient is determined, so a
    constant offset on p is an exact symmetry. Fields are already normalized
    when this runs, so `scale` is in units of the field's standard deviation.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = fields.clone()
    for i in idx:
        out[:, int(i)] += float(rng.normal(0.0, scale))
    return out


def reflect_axis_augment(
    fields: torch.Tensor,
    grid_shape: Sequence[int],
    axis: int,
    sign_flip_idx: Sequence[int],
    rng: Optional[np.random.Generator] = None,
    p: float = 0.5,
) -> torch.Tensor:
    """Reflect a structured grid about one axis, negating the named channels.

    Used where only a single spatial symmetry survives: FireBench's lateral
    direction (mean wind fixes x, gravity fixes z) and the wing's span.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() >= p:
        return fields
    gs = [int(s) for s in grid_shape]
    n_f = fields.shape[-1]
    g = fields.reshape(*gs, n_f)
    g = torch.flip(g, dims=[int(axis)])
    if sign_flip_idx:
        sign = torch.ones(n_f, dtype=g.dtype, device=g.device)
        for i in sign_flip_idx:
            sign[int(i)] = -1.0
        g = g * sign
    return g.reshape(int(np.prod(gs)), n_f).contiguous()


