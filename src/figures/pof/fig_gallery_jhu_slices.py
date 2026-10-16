"""JHU gallery with 3D orthoslices -- an alternative to jhu_compact.

Each panel shows the 125^3 cube with the octant nearest the viewer cut away:
three outer faces and, inside the notch, the x, y and z midplanes. Sensors
are marked on the midplane faces only (all visible faces would hold ~470 per
channel and hide the truth field). The columns, rows, color scales and panel
numbers (single-member relative L2 over the whole cube) are those of
jhu_compact in fig_gallery_allbaselines.py; only the panel rendering differs.
Faces are affine-transformed rasters drawn back to front, so the PDF stays small.

    PYTHONPATH=src python -m figures.pof.fig_gallery_jhu_slices
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from figures.pof.fig_gallery_allbaselines import (
    COMPACT_COLUMNS, COMPACT_RC, F_SUB, INK, INK_2, Gallery, coords_of, make_blocks)
from figures.pof.pof_style import use_style

STEM = "gallery_jhu_allbaselines_slices"
AZ, EL = np.deg2rad(45.0), np.deg2rad(30.0)
# orthographic projection of unit-cube coordinates (x, y, z) to panel (u, v);
# the viewer sits in the +x, +y, +z octant, z points up
PROJ = np.array([[-np.sin(AZ), np.cos(AZ), 0.0],
                 [-np.sin(EL) * np.cos(AZ), -np.sin(EL) * np.sin(AZ), np.cos(EL)]])
_UV = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ PROJ.T
PAD = 0.22                                         # room under the cube for the error label
XLIM = (_UV[:, 0].min(), _UV[:, 0].max())
YLIM = (_UV[:, 1].min() - PAD, _UV[:, 1].max())
EDGE = dict(color=INK_2, lw=0.3, rasterized=True)


def faces(g):
    """Visible faces back to front as (normal axis k, data index along k,
    plane position along k, first in-plane index): the outer faces at the far
    cells, then the notch faces at the midplane index g // 2."""
    c = g // 2
    return [(k, g - 1, 1.0, 0) for k in range(3)] + [(k, c, c / g, c) for k in range(3)]


def cube_of(d, vals):
    """[N] point values -> [g, g, g] array indexed (ix, iy, iz). Every JHU dump
    stores z fastest; the coordinates are checked so a reordered dump fails."""
    g = int(round(vals.size ** (1 / 3)))
    c = coords_of(d)
    ijk = np.rint((c - c.min(0)) / (c[1, 2] - c[0, 2])).astype(np.int64)
    if not np.array_equal(ijk @ [g * g, g, 1], np.arange(g ** 3)):
        raise ValueError("JHU dump points are not in (x, y, z) C order")
    return vals.reshape(g, g, g)


def draw_cube(ax, d, vals, vmin, vmax, cmap):
    vol = cube_of(d, vals)
    g = vol.shape[0]
    for n, (k, idx, pos, lo) in enumerate(faces(g)):
        a, b = [i for i in range(3) if i != k]
        p0, e1, e2 = np.full(3, lo / g), np.zeros(3), np.zeros(3)
        p0[k], e1[a], e2[b] = pos, 1 - lo / g, 1 - lo / g
        # unit square (s, t) -> p0 + s e1 + t e2, then projected
        m = np.vstack([np.column_stack([PROJ @ e1, PROJ @ e2, PROJ @ p0]), [0, 0, 1]])
        face = np.take(vol, idx, axis=k)[lo:, lo:]                 # [i_a, i_b]
        im = ax.imshow(face.T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="bicubic",
                       transform=Affine2D(m) + ax.transData, rasterized=True, zorder=2 * n)
        q = np.array([p0, p0 + e1, p0 + e1 + e2, p0 + e2, p0]) @ PROJ.T
        ax.plot(q[:, 0], q[:, 1], zorder=2 * n + 1, **EDGE)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
    ax.set_axis_off()
    return im


def sensor_uv(ref, j):
    """Projected positions of the field-j sensors on the three midplane faces."""
    g = int(round(ref["truth"].shape[0] ** (1 / 3)))
    c = g // 2
    idx = np.asarray(ref["sensor_indices"])[np.asarray(ref["sensor_field_ids"]) == j]
    ijk = np.stack([idx // (g * g), (idx // g) % g, idx % g], axis=1)
    pts = []
    for k in range(3):
        p = (ijk[(ijk[:, k] == c) & (ijk >= c).all(1)] + 0.5) / g
        p[:, k] = c / g
        pts.append(p)
    return (np.concatenate(pts) @ PROJ.T).T


def triad(ax, origin=(-0.55, -0.84), length=0.17):
    """Projected axis directions, drawn in the empty lower-left corner of ax."""
    o = np.asarray(origin)
    for name, e in zip("xyz", np.eye(3)):
        dv = length * PROJ @ e
        ax.annotate("", xy=o + dv, xytext=o, arrowprops=dict(
            arrowstyle="-|>", lw=0.5, color=INK_2, mutation_scale=4, shrinkA=0, shrinkB=0))
        ax.text(*(o + 1.5 * dv), f"${name}$", fontsize=F_SUB, color=INK_2,
                ha="center", va="center")


class CubeGallery(Gallery):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.draw = draw_cube
        self.aspect = (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])

    def truth_panel(self, ax, j, observed):
        im = super().truth_panel(ax, j, False)     # field only: the base marks 2D z-midplane sensors
        if observed:
            u, v = sensor_uv(self.ref, j)
            ax.scatter(u, v, s=self.sensor_ms, c="white", edgecolors=INK, linewidths=0.25,
                       marker="o", rasterized=True, zorder=20)
        if j == self.chans[-1][1]:
            triad(ax)
        return im


if __name__ == "__main__":
    use_style()
    with plt.rc_context(COMPACT_RC):
        make_blocks("jhu", per_block=len(COMPACT_COLUMNS["jhu"]), cols=COMPACT_COLUMNS["jhu"],
                    stem=STEM, sensor_ms=1.4, truth_title="truth", gallery=CubeGallery)
