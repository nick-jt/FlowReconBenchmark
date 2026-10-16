"""Regular-grid detection and smooth raster rendering for point-ordered fields.

Every reconstruction in this project is stored as a point cloud ``[N, C]`` with
coordinates ``[N, 2|3]``, even when the points are the nodes of a regular
tensor-product grid (2D Kolmogorov 256x256, the cylinder grid export 200x400,
a JHU z-midplane 125x125).  Drawing such a field through a Delaunay
triangulation (``tripcolor`` / ``tricontourf``) leaves visible facet and
aliasing artefacts, worst for single generative samples whose chunk-wise
speckle is sharpened by the gouraud shading.  Rendering the same values with
``imshow(origin="lower", interpolation="bicubic", extent=...)`` is both faster
and cleaner.

This module is deliberately numpy-only (no torch / h5py) so the paper figure
scripts and the baseline visualizers can share it.

Gridness is decided from the COORDINATES alone, never from a model name: the
points must form a complete tensor product of their unique x and y values with
near-uniform spacing along each axis.  Any point ordering is accepted (x
fastest, y fastest, or an arbitrary permutation) because each value is placed
by its ``(iy, ix)`` rank, so the result is never transposed relative to the
physical axes.

The imshow ``extent`` is padded by half a cell so the pixel CENTRES land on
the point coordinates; scatter overlays of sensor positions therefore sit
exactly on the samples they annotate.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "RegularGrid",
    "regular_grid_from_points",
    "nearest_fill",
    "imshow_points",
]


class RegularGrid:
    """A detected (ny, nx) tensor-product grid over a point-ordered field."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray,
                 ix: np.ndarray, iy: np.ndarray):
        self.xs = xs                      # unique x, ascending
        self.ys = ys                      # unique y, ascending
        self.ix = ix                      # [N] column of each point
        self.iy = iy                      # [N] row of each point
        self.nx = int(xs.size)
        self.ny = int(ys.size)

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def dx(self) -> float:
        return float(self.xs[1] - self.xs[0]) if self.nx > 1 else 1.0

    @property
    def dy(self) -> float:
        return float(self.ys[1] - self.ys[0]) if self.ny > 1 else 1.0

    @property
    def extent(self):
    # zcollapse-ok: planar grid helper, callers slice volumes first
        """imshow extent with pixel centres on the point coordinates."""
        return [float(self.xs[0]) - 0.5 * self.dx, float(self.xs[-1]) + 0.5 * self.dx,
                float(self.ys[0]) - 0.5 * self.dy, float(self.ys[-1]) + 0.5 * self.dy]

    @property
    def point_extent(self):
        """[xmin, xmax, ymin, ymax] of the points themselves (for axis limits)."""
        return [float(self.xs[0]), float(self.xs[-1]),
                float(self.ys[0]), float(self.ys[-1])]

    def image(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
    # zcollapse-ok: planar (ny, nx) raster of an already 2-D point set
        """Point-ordered ``[N]`` (or ``[N, C]``) -> ``[ny, nx]`` (or ``[ny, nx, C]``),
        rows = y, columns = x, so ``imshow(origin="lower")`` shows x to the
        right and y up."""
        v = np.asarray(values)
        out = np.full((self.ny, self.nx) + v.shape[1:], fill, dtype=np.float64)
        out[self.iy, self.ix] = v
        return out

    def meshgrid(self):
        """(X, Y) arrays of shape (ny, nx) for ``ax.contour``."""
        return np.meshgrid(self.xs, self.ys)


def _uniform(values: np.ndarray, rtol: float) -> bool:
    if values.size < 3:
        return True
    d = np.diff(values)
    med = float(np.median(d))
    if med <= 0:
        return False
    return bool(np.all(np.abs(d - med) <= rtol * med))


def regular_grid_from_points(coords_xy: np.ndarray, rtol: float = 1e-3,
                             decimals: Optional[int] = None) -> Optional[RegularGrid]:
    """Return a :class:`RegularGrid` if ``coords_xy[:, :2]`` is a complete,
    near-uniform tensor-product grid, else ``None``.

    ``decimals`` rounds the coordinates before the uniqueness test (float32
    grids exported through different code paths can differ in the last bits);
    the default picks a tolerance of ~1e-6 of the coordinate range.
    """
    c = np.asarray(coords_xy, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 4 or c.shape[1] < 2:
        return None
    x, y = c[:, 0], c[:, 1]
    if decimals is None:
        span = max(float(np.ptp(x)), float(np.ptp(y)), 1e-12)
        decimals = int(max(0, np.ceil(-np.log10(span * 1e-6))))
    xr = np.round(x, decimals)
    yr = np.round(y, decimals)
    xs, ix = np.unique(xr, return_inverse=True)
    ys, iy = np.unique(yr, return_inverse=True)
    n = c.shape[0]
    if xs.size < 2 or ys.size < 2 or xs.size * ys.size != n:
        return None
    # complete tensor product: every (iy, ix) cell hit exactly once
    cell = iy.astype(np.int64) * xs.size + ix.astype(np.int64)
    if np.unique(cell).size != n:
        return None
    if not (_uniform(xs, rtol) and _uniform(ys, rtol)):
        return None
    return RegularGrid(xs, ys, ix.astype(np.int64), iy.astype(np.int64))


def nearest_fill(img: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Replace ``img[invalid]`` by the nearest valid pixel (plotting-only
    inpainting).  Used to fill a solid body before bicubic interpolation so
    the smoothing kernel does not blend the body's zeros into the flow at the
    wall; the body is then overdrawn with its patch."""
    from scipy.ndimage import distance_transform_edt
    inv = np.asarray(invalid, dtype=bool)
    if not inv.any() or inv.all():
        return img
    idx = distance_transform_edt(inv, return_distances=False, return_indices=True)
    return img[tuple(idx)]


def imshow_points(ax, grid: RegularGrid, values: np.ndarray, *,
                  interpolation: str = "bicubic", invalid: Optional[np.ndarray] = None,
                  **imshow_kw):
    """``ax.imshow`` of a point-ordered field on a detected regular grid.

    ``invalid`` is an optional boolean image mask of pixels to inpaint
    (nearest neighbour) before interpolation, e.g. the interior of a body.
    Sets origin lower, the half-cell extent and equal aspect unless overridden.
    """
    # zcollapse-ok: draws a 2-D grid; volumetric callers pass a z-slice (midplane_slice)
    img = grid.image(values)
    if invalid is not None:
        img = nearest_fill(img, invalid)
    kw = dict(origin="lower", interpolation=interpolation, extent=grid.extent,
              aspect="equal")
    kw.update(imshow_kw)
    return ax.imshow(img, **kw)
