"""All-baselines reconstruction galleries -- one figure per dataset.

Methods are ordered generative learned -> deterministic learned -> classical in
every gallery. Layouts (paper widths, fonts >= 7 pt):
  cylinder, cylinder_surface : one column wide, one METHOD per row (truth first),
                               one CHANNEL per column, a colorbar under each column
  kolmogorov                 : full width, 2 x 5 panels (truth + 9 methods), one
                               channel label and one colorbar for both rows
  jhu                        : full width, two stacked blocks of 4 channel rows,
                               truth + 5 methods per block
Compact variants (paper revision; the layouts above are unchanged):
  cylinder_merged            : one column, volume and surface sensing side by side
                               for u_y and p, one colorbar per channel pair
  kolmogorov_compact         : one column, 3 x 3 panels (truth + 8 methods)
  jhu_compact                : full width, one block, truth + 7 methods

Every column conditions on the sensors its TABLE ROW scored:
  Kolmogorov  test-half frame 128 (a fleet-scored frame), 655 vorticity
              sensors, all dumps idx_sum 21466837.
  Cylinder    test-half frame 150 (first held-out Re 250 frame, fleet-scored),
              238 sensors per field on u_x,u_y; every method is mesh-native and
              shares the one draw (idx_sum 5569341). Latent FM and S3GM have no
              cylinder row (grid-locked).
  JHU         held-out cube, test-half frame 152 (val index 2), z-midplane;
              19,531 sensors on each of U_x, U_z (qual draw, seed 102), injected
              identically except in Gen4Turb's own 120^3 crop. Dumps in
              field_dumps/frame152/.

Generative columns show ONE posterior sample (never an ensemble mean);
deterministic columns their single prediction. The per-panel number is that
displayed field's relative L2 error for that channel, in its dump's own
standardized units, over the whole field (the whole cube for JHU) -- a
single-sample error, so it sits above the table's ensemble-mean relative L2.

Color (dataviz reference palette): signed fields use a blue <-> red diverging
map around a neutral gray midpoint, symmetric about zero; one-sided fields a
single-hue blue ramp. Scales are anchored on truth (2-98th percentile), shared
along each row. A per-panel error table is written next to each figure.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from paths import REPO_ROOT

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex

from figures.pof.pof_style import use_style
from figures.regular_grid import regular_grid_from_points, imshow_points

WT = REPO_ROOT
FD = WT / "outputs/pof" / "field_dumps"
OUT = WT / "outputs" / "paper" / "figures"

INK, INK_2, MUTED, RULE = "#262624", "#52514e", "#898781", "#c9c8c2"
MID = "#f0efec"                                   # neutral diverging midpoint
BLUE_ARM = ["#0d366b", "#1c5cab", "#2a78d6", "#86b6ef", "#cde2fb"]   # dark -> light


def _mix(c, other, t):
    a, b = np.array(to_rgb(c)), np.array(to_rgb(other))
    return to_hex((1 - t) * a + t * b)


# The palette specifies the diverging pair (blue <-> red) and the midpoint but
# no red ramp, so the red arm is stepped from categorical red #e34948 to mirror
# the blue arm's lightness positions (light, light, base, dark, dark).
RED = "#e34948"
RED_ARM = [_mix(RED, "#ffffff", 0.78), _mix(RED, "#ffffff", 0.45), RED,
           _mix(RED, "#000000", 0.28), _mix(RED, "#000000", 0.52)]      # light -> dark
DIVERGING = LinearSegmentedColormap.from_list("div", BLUE_ARM + [MID] + RED_ARM)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
for cm in (DIVERGING, SEQUENTIAL):
    cm.set_bad(MID)

# (column title, npz, key) -- one order for every dataset
COLUMNS = {
    "kolmogorov": [
        ("DMF-Gen", "kolm_dmfgen.npz", "pred_sample"),
        ("SiT-point", "kolm_sit.npz", "pred_sample"),
        ("latent FM", "kolm_latent_fm.npz", "pred_sample"),
        ("S3GM", "kolm_s3gm.npz", "pred_sample"),
        ("Senseiver", "kolm_senseiver.npz", "pred_mean"),
        ("MLP-RBF", "kolm_mlprbf.npz", "pred_mean"),
        ("FNO", "kolm_geofno.npz", "pred_mean"),
        ("IDW", "kolm_classical.npz", "pred_idw"),
        ("gappy POD ($r{=}160$)", "kolm_classical.npz", "pred_gappy_pod_r160"),
    ],
    "cylinder": [
        ("DMF-Gen", "cyl_dmfgen_b128.npz", "pred_sample"),
        ("SiT-point", "cyl_sit.npz", "pred_sample"),
        ("Senseiver", "cyl_senseiver.npz", "pred_mean"),
        ("MLP-RBF", "cyl_mlprbf.npz", "pred_mean"),
        ("Geo-FNO", "cyl_geofno.npz", "pred_mean"),
        ("IDW", "cyl_classical.npz", "pred_idw"),
        ("gappy POD ($r{=}20$)", "cyl_classical.npz", "pred_gappy_pod_r20"),
    ],
    # surface-to-field task: the same frame, sensors confined to the wall ring
    "cylinder_surface": [
        ("DMF-Gen", "cylsurf_dmfgen_b128.npz", "pred_sample"),
        ("SiT-point", "cylsurf_sit.npz", "pred_sample"),
        ("Senseiver", "cylsurf_senseiver.npz", "pred_mean"),
        ("Geo-FNO", "cylsurf_geofno.npz", "pred_mean"),
        ("IDW", "cylsurf_classical.npz", "pred_idw"),
        ("gappy POD ($r{=}5$)", "cylsurf_classical.npz", "pred_gappy_pod_r5"),
    ],
    "jhu": [
        ("DMF-Gen", "frame152/jhu_dmfgen.npz", "pred_sample"),
        ("SiT-point", "frame152/jhu_sit.npz", "pred_sample"),
        ("latent FM", "frame152/jhu_latent_fm.npz", "pred_sample"),
        ("FNO3D-FM", "frame152/jhu_fno3d.npz", "pred_sample"),
        ("CoNFiLD (published prior)", "frame152/jhu_confild.npz", "pred_sample"),
        ("Gen4Turb ($120^3$, own draw)", "frame152/jhu_gen4turb.npz", "pred_sample"),
        ("S3GM (norm. guidance)", "frame152/jhu_s3gm.npz", "pred_sample"),
        ("Senseiver", "frame152/jhu_senseiver.npz", "pred_mean"),
        ("IDW", "frame152/jhu_classical.npz", "pred_idw"),
        ("gappy POD ($r{=}80$)", "frame152/jhu_classical.npz", "pred_gappy_pod_r80"),
    ],
}
# Compact variants: short titles (ranks go in the caption), fewer methods.
COMPACT_COLUMNS = {
    "kolmogorov": [
        ("DMF-Gen", "kolm_dmfgen_cq_range.npz", "pred_sample"),
        ("SiT-point", "kolm_sit.npz", "pred_sample"),
        ("latent FM", "kolm_latent_fm.npz", "pred_sample"),
        ("S3GM", "kolm_s3gm.npz", "pred_sample"),
        ("Senseiver", "kolm_senseiver.npz", "pred_mean"),
        ("FNO", "kolm_geofno.npz", "pred_mean"),
        ("IDW", "kolm_classical.npz", "pred_idw"),
        ("gappy POD", "kolm_classical.npz", "pred_gappy_pod_r160"),
    ],
    "jhu": [
        ("DMF-Gen", "frame152/jhu_dmfgen.npz", "pred_sample"),
        ("SiT-point", "frame152/jhu_sit.npz", "pred_sample"),
        ("latent FM", "frame152/jhu_latent_fm.npz", "pred_sample"),
        ("CoNFiLD", "frame152/jhu_confild.npz", "pred_sample"),
        ("Gen4Turb", "frame152/jhu_gen4turb.npz", "pred_sample"),
        ("S3GM", "frame152/jhu_s3gm.npz", "pred_sample"),
        ("IDW", "frame152/jhu_classical.npz", "pred_idw"),
    ],
}
# merged cylinder gallery: (row label, volume (npz, key), surface (npz, key))
CYL_MERGED_ROWS = [
    ("gappy POD\n($r$ 20 | 5)", ("cyl_classical.npz", "pred_gappy_pod_r20"),
     ("cylsurf_classical.npz", "pred_gappy_pod_r5")),
    ("IDW", ("cyl_classical.npz", "pred_idw"), ("cylsurf_classical.npz", "pred_idw")),
    ("SiT-point", ("cyl_sit.npz", "pred_sample"), ("cylsurf_sit.npz", "pred_sample")),
    ("Geo-FNO", ("cyl_geofno.npz", "pred_mean"), ("cylsurf_geofno.npz", "pred_mean")),
    ("Senseiver", ("cyl_senseiver.npz", "pred_mean"), ("cylsurf_senseiver.npz", "pred_mean")),
    ("DMF-Gen", ("cyl_dmfgen_b128.npz", "pred_sample"), ("cylsurf_dmfgen_b128.npz", "pred_sample")),
]
CYL_ROI_COMPACT = (-2.0, 12.0, -4.0, 4.0)
# (label, channel index, observed?)
CHANNELS = {
    "kolmogorov": [(r"$\omega$", 0, True)],
    "cylinder": [(r"$u_x$", 0, True), (r"$u_y$", 1, True), (r"$p$", 2, False)],
    "cylinder_surface": [(r"$u_x$", 0, True), (r"$u_y$", 1, True), (r"$p$", 2, True)],
    "jhu": [(r"$U_x$", 0, True), (r"$U_y$", 1, False), (r"$U_z$", 2, True), (r"$p$", 3, False)],
}
OBS_WORD = {"cylinder_surface": ("tapped", "tapped")}        # (observed, unobserved) wording
REFERENCE = {"kolmogorov": "kolm_classical.npz", "cylinder": "cyl_dmfgen_b128.npz",
             "cylinder_surface": "cylsurf_classical.npz", "jhu": "frame152/jhu_sit.npz"}
STEM = {"kolmogorov": "gallery_kolmogorov_allbaselines_paper",
        "cylinder": "gallery_cylinder_allbaselines_paper",
        "cylinder_surface": "recon_gallery_cylinder_surface",
        "jhu": "gallery_jhu_allbaselines_paper"}
COMPACT_STEM = {"kolmogorov": "gallery_kolmogorov_allbaselines_compact",
                "cylinder_merged": "gallery_cylinder_merged_compact",
                "jhu": "gallery_jhu_allbaselines_compact"}
CYL_ROI, CYL_R = (-3.0, 17.0, -5.0, 5.0), 0.5
COL_W, FULL_W = 3.4, 7.0
F_TITLE, F_LABEL, F_SUB, F_NUM, F_CB = 8.0, 8.0, 7.0, 7.0, 6.5
# Why a slot is empty, when the reason is known -- "not dumped" hides the
# difference between "not run yet" and "cannot be run here".
PENDING_REASON = {}
G = 125


def load(name):
    p = FD / name
    return dict(np.load(p, allow_pickle=False)) if p.is_file() else None


def to_physical(d, arr, dataset):
    """JHU dumps are stored physical; the 2D dumps in their own z-units."""
    a = np.asarray(arr, dtype=np.float64)
    if dataset == "jhu":
        return a
    return a * np.asarray(d["norm_std"], dtype=np.float64).ravel() + \
        np.asarray(d["norm_mean"], dtype=np.float64).ravel()


def rel_l2_channel(d, key, j, dataset, ref):
    """Relative L2 for channel j in standardized units (table convention)."""
    if dataset == "jhu":
        src = d if "norm_mean" in d else ref        # Gen4Turb carries its own crop's stats
        m = np.asarray(src["norm_mean"], np.float64)[j]
        s = np.asarray(src["norm_std"], np.float64)[j]
        pr = (np.asarray(d[key], np.float64)[:, j] - m) / s
        tr = (np.asarray(d["truth"], np.float64)[:, j] - m) / s
    else:
        pr = np.asarray(d[key], np.float64)[:, j]
        tr = np.asarray(d["truth"], np.float64)[:, j]
    return float(np.linalg.norm(pr - tr) / (np.linalg.norm(tr) + 1e-12))


def row_scale(v):
    lo, hi = np.nanpercentile(v, [2, 98])
    if lo < 0 < hi and min(abs(lo), abs(hi)) / max(abs(lo), abs(hi)) > 0.10:
        m = float(max(abs(lo), abs(hi)))
        return -m, m, DIVERGING
    return float(lo), float(hi), SEQUENTIAL


def coords_of(d):
    return np.asarray(d["coords_raw"] if "coords_raw" in d else d["coords"], np.float64)


# ---------------------------------------------------------------- renderers --
# Regular-grid panels (Kolmogorov, JHU midplane) are bicubic rasters:
# nearest/gouraud rendering sharpened the chunk-wise speckle of single
# generative samples into visible facets.  Gridness is decided from the dump's
# coordinates, never from the column name.
def draw_kolm(ax, d, vals, vmin, vmax, cmap):
    # zcollapse-ok: Kolmogorov is genuinely 2-D
    c = coords_of(d)
    side = int(round(np.sqrt(vals.size)))
    a = vals.reshape(side, side)
    fast_x = np.ptp(c[:side, 0]) > np.ptp(c[:side, 1])
    img = a if fast_x else a.T
    return ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="bicubic", rasterized=True)


_TRI = {}


def draw_cyl(ax, d, vals, vmin, vmax, cmap, roi=CYL_ROI):
    # zcollapse-ok: cylinder data is genuinely 2-D (mesh or grid export)
    c = coords_of(d)[:, :2]
    x0, x1, y0, y1 = roi
    grid = regular_grid_from_points(c)
    if grid is not None:                           # grid export
        X, Y = grid.meshgrid()
        im = imshow_points(ax, grid, vals, cmap=cmap, vmin=vmin, vmax=vmax,
                           invalid=(X ** 2 + Y ** 2) < CYL_R ** 2,
                           rasterized=True)
    else:                                           # body-fitted mesh
        key = (c.shape[0], float(c[:, 0].sum()))
        if key not in _TRI:
            t = mtri.Triangulation(c[:, 0], c[:, 1])
            cx = c[t.triangles].mean(axis=1)
            t.set_mask((cx ** 2).sum(1) < CYL_R ** 2)   # Delaunay spans the body
            _TRI[key] = t
        im = ax.tripcolor(_TRI[key], vals, shading="gouraud", cmap=cmap,
                          vmin=vmin, vmax=vmax, rasterized=True)
    ax.add_patch(plt.Circle((0, 0), CYL_R, fc=MID, ec=MUTED, lw=0.5, zorder=3))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    return im


def draw_jhu(ax, d, vals, vmin, vmax, cmap):
    # zcollapse-ok: takes the fixed z-midplane slice [:, :, zc] below; the grid
    # side comes from the dump (Gen4Turb's crop is 120^3, the rest 125^3)
    g = int(round(vals.size ** (1 / 3)))
    zc = g // 2
    sl = vals.reshape(g, g, g)[:, :, zc]
    return ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="bicubic", rasterized=True)


DRAW = {"kolmogorov": draw_kolm, "cylinder": draw_cyl, "cylinder_surface": draw_cyl,
        "jhu": draw_jhu}
PANEL_ASPECT = {"kolmogorov": 1.0, "cylinder": 0.5, "cylinder_surface": 0.5, "jhu": 1.0}


def sensor_xy(dataset, ref, j):
    idx = np.asarray(ref["sensor_indices"]).astype(np.int64)
    fid = np.asarray(ref["sensor_field_ids"]).astype(np.int64)
    pi = idx[fid == j]
    if dataset.startswith("cylinder"):
        c = coords_of(ref)
        return c[pi, 0], c[pi, 1], 1.6
    if dataset == "kolmogorov":
        c = coords_of(ref); side = int(round(np.sqrt(c.shape[0])))
        r, q = divmod(pi, side)
        fast_x = np.ptp(c[:side, 0]) > np.ptp(c[:side, 1])
        # 655 sensors on a ~1 in panel: markers must stay near-point-sized or
        # they hide the field they are meant to annotate
        return (q, r, 0.6) if fast_x else (r, q, 0.6)
    zc = G // 2                                     # jhu: sensors in the shown plane
    pi = pi[pi % G == zc]
    return pi // (G * G), (pi // G) % G, 2.4


def style(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color(RULE)


def two_line(title):
    return title.replace(" (", "\n(")


def obs_word(dataset, observed):
    w = OBS_WORD.get(dataset, ("observed", "unobserved"))
    return w[0] if observed else w[1]


class Gallery:
    """Shared panel drawing for every layout; the per-panel error table is
    collected as the panels are drawn."""

    def __init__(self, dataset, cols=None, stem=None, roi=None, sensor_ms=None):
        self.ds = dataset
        self.ref = load(REFERENCE[dataset])
        if self.ref is None:
            raise SystemExit(f"[{dataset}] reference dump {REFERENCE[dataset]} missing")
        self.cols = COLUMNS[dataset] if cols is None else cols
        self.chans = CHANNELS[dataset]
        self.stem = STEM.get(dataset) if stem is None else stem
        self.roi = CYL_ROI if roi is None else roi
        self.draw_kw = {"roi": self.roi} if dataset.startswith("cylinder") else {}
        self.sensor_ms = sensor_ms
        self.num_size = F_NUM
        self.draw, self.aspect = DRAW[dataset], PANEL_ASPECT[dataset]
        self.truth = to_physical(self.ref, self.ref["truth"], dataset)
        self.scale = {j: row_scale(self.truth[:, j]) for _, j, _ in self.chans}
        self.cache, self.table = {}, []

    def truth_panel(self, ax, j, observed):
        vmin, vmax, cmap = self.scale[j]
        im = self.draw(ax, self.ref, self.truth[:, j], vmin, vmax, cmap, **self.draw_kw)
        if observed:
            sx, sy, ms = sensor_xy(self.ds, self.ref, j)
            ms = ms if self.sensor_ms is None else self.sensor_ms
            ax.scatter(sx, sy, s=ms, c="white", edgecolors=INK,
                       linewidths=0.18 if self.ds == "kolmogorov" else 0.3,
                       marker="o", rasterized=True, zorder=4)
        style(ax)
        return im

    def method_panel(self, ax, col, lab, j, observed):
        title, fname, key = col
        vmin, vmax, cmap = self.scale[j]
        if fname not in self.cache:
            self.cache[fname] = load(fname)
        d = self.cache[fname]
        if d is None or key not in d:
            ax.set_facecolor("#f6f5f2")
            ax.text(0.5, 0.5, PENDING_REASON.get(fname, "not dumped"),
                    transform=ax.transAxes, linespacing=1.3,
                    ha="center", va="center", fontsize=F_SUB, color=MUTED)
            if self.ds.startswith("cylinder"):
                ax.set_xlim(*self.roi[:2]); ax.set_ylim(*self.roi[2:]); ax.set_aspect("equal")
            style(ax)
            self.table.append({"channel": lab.strip("$"), "method": title.split(" (")[0],
                               "observed": observed, "rel_l2": None})
            return None
        vals = to_physical(d, d[key], self.ds)[:, j]
        im = self.draw(ax, d, vals, vmin, vmax, cmap, **self.draw_kw)
        err = rel_l2_channel(d, key, j, self.ds, self.ref)
        self.table.append({"channel": lab.strip("$"), "method": title.split(" (")[0],
                           "observed": observed, "rel_l2": round(err, 4)})
        ax.text(0.97, 0.05, f"{err:.2f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=self.num_size, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
        style(ax)
        return im

    def colorbar(self, fig, cax, im, j, orientation="vertical"):
        vmin, vmax, cmap = self.scale[j]
        cb = fig.colorbar(im, cax=cax, orientation=orientation)
        ticks = [vmin, 0.0, vmax] if cmap is DIVERGING else [vmin, vmax]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.2g}" for t in ticks])
        cb.ax.tick_params(labelsize=F_CB, length=2, pad=1, colors=INK_2)
        cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(RULE)
        return cb

    def write(self, fig, W, H):
        write_figure(fig, self.stem, self.table, ["channel", "method", "observed", "rel_l2"],
                     self.ds, W, H)


def write_figure(fig, stem, rows, fields, tag, W, H):
    """Save the gallery (png 220 dpi, pdf 300 dpi) and its per-panel error table."""
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    fig.savefig(OUT / f"{stem}.pdf", dpi=300)
    plt.close(fig)
    with open(OUT / f"{stem}_rell2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    done = sum(1 for t in rows if t["rel_l2"] is not None)
    print(f"[{tag}] wrote {stem}.png/.pdf  ({done}/{len(rows)} panels, {W:.1f}x{H:.1f} in)")


# ------------------------------------------------------------------ layouts --
def make_vertical(dataset):
    """One column wide: rows = truth then one method each, columns = channels,
    a horizontal colorbar under each channel column."""
    g = Gallery(dataset)
    n_r, n_c = 1 + len(g.cols), len(g.chans)
    lab_w, top, cb_h, bot = 0.70, 0.36, 0.09, 0.26
    pw = (COL_W - lab_w - 0.06) / (n_c + 0.06 * (n_c - 1))
    ph = pw * PANEL_ASPECT[dataset]
    W = COL_W
    H = top + n_r * ph + 0.05 * (n_r - 1) + cb_h + bot
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(n_r + 1, n_c, height_ratios=[1] * n_r + [cb_h / ph],
                          left=lab_w / W, right=1 - 0.06 / W,
                          top=1 - top / H, bottom=bot / H, wspace=0.06, hspace=0.07)
    ims = {}
    for c_i, (lab, j, observed) in enumerate(g.chans):
        ax = fig.add_subplot(gs[0, c_i])
        ims[j] = g.truth_panel(ax, j, observed)
        ax.set_title(f"{lab} {obs_word(dataset, observed)}", fontsize=F_TITLE, color=INK, pad=3)
        if c_i == 0:
            b = ax.get_position()
            fig.text(0.04 / W, b.y0 + b.height / 2, "truth +\nsensors" if dataset != "cylinder_surface"
                     else "truth +\ntaps", fontsize=F_LABEL, color=INK, va="center", ha="left")
        for r, col in enumerate(g.cols, start=1):
            ax = fig.add_subplot(gs[r, c_i])
            g.method_panel(ax, col, lab, j, observed)
            if c_i == 0:
                b = ax.get_position()
                fig.text(0.04 / W, b.y0 + b.height / 2, two_line(col[0]), fontsize=F_LABEL,
                         color=INK, va="center", ha="left", linespacing=1.15)
        cax = fig.add_subplot(gs[n_r, c_i])
        b = cax.get_position()
        cax.set_position([b.x0 + 0.1 * b.width, b.y0, 0.8 * b.width, b.height])
        g.colorbar(fig, cax, ims[j], j, orientation="horizontal")
    g.write(fig, W, H)


def make_grid(dataset, n_r=2, n_c=5, W=FULL_W, lab_w=0.62, row_gap=0.30, hspace=0.28,
              cols=None, stem=None):
    """n_r x n_c panels: truth then the methods in order, one colorbar for all
    rows. With a label margin (lab_w > 0.3) the channel name sits on the left;
    otherwise it heads the colorbar."""
    g = Gallery(dataset, cols=cols, stem=stem)
    lab, j, observed = g.chans[0]
    panels = [None] + list(g.cols)
    assert len(panels) == n_r * n_c
    cb_w, top, bot = 0.40, 0.30, 0.06
    pw = (W - lab_w - cb_w - 0.12) / (n_c + 0.05 * (n_c - 1))
    ph = pw * PANEL_ASPECT[dataset]
    H = top + n_r * (ph + row_gap) + bot
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(n_r, n_c + 1, width_ratios=[1] * n_c + [0.06],
                          left=lab_w / W, right=1 - 0.30 / W,
                          top=1 - top / H, bottom=bot / H, wspace=0.05, hspace=hspace)
    im = None
    for i, col in enumerate(panels):
        ax = fig.add_subplot(gs[i // n_c, i % n_c])
        if col is None:
            im = g.truth_panel(ax, j, observed)
            ax.set_title("truth + sensors", fontsize=F_TITLE, color=INK, pad=3)
        else:
            g.method_panel(ax, col, lab, j, observed)
            ax.set_title(col[0], fontsize=F_TITLE, color=INK, pad=3)
    if lab_w > 0.3:
        fig.text(0.10 / W, (gs.top + gs.bottom) / 2 + 0.08 / H, lab, fontsize=F_LABEL + 2,
                 color=INK, va="center", ha="left")
        fig.text(0.10 / W, (gs.top + gs.bottom) / 2 - 0.10 / H, obs_word(dataset, observed),
                 fontsize=F_SUB, color=INK_2, va="top", ha="left")
    cax = fig.add_subplot(gs[:, n_c])
    b = cax.get_position()
    cax.set_position([b.x0, b.y0 + 0.10 * b.height, b.width, 0.80 * b.height])
    g.colorbar(fig, cax, im, j)
    if lab_w <= 0.3:
        cax.set_title(lab, fontsize=F_LABEL + 1, color=INK, pad=4)
    g.write(fig, W, H)


def make_grid2x5(dataset):
    return make_grid(dataset)


def make_blocks(dataset, per_block=5, cols=None, stem=None, sensor_ms=None,
                truth_title="truth + sensors", gallery=Gallery):
    """Full width, stacked blocks of (truth + per_block methods) columns, one
    channel per row inside each block, a colorbar per row."""
    g = gallery(dataset, cols=cols, stem=stem, sensor_ms=sensor_ms)
    blocks = [g.cols[i:i + per_block] for i in range(0, len(g.cols), per_block)]
    n_c = 1 + per_block
    n_ch = len(g.chans)
    n_r = n_ch * len(blocks)
    lab_w, cb_w, top, bot, block_gap = 0.70, 0.40, 0.30, 0.06, 0.34
    pw = (FULL_W - lab_w - cb_w - 0.12) / (n_c + 0.05 * (n_c - 1))
    ph = pw * g.aspect
    W = FULL_W
    H = top + n_r * ph + 0.05 * (n_r - 1) + block_gap * (len(blocks) - 1) + bot
    fig = plt.figure(figsize=(W, H))
    hr = []
    for bi in range(len(blocks)):
        hr += [1.0] * n_ch
        if bi < len(blocks) - 1:
            hr.append(block_gap / ph)
    gs = fig.add_gridspec(len(hr), n_c + 1, width_ratios=[1] * n_c + [0.07],
                          height_ratios=hr,
                          left=lab_w / W, right=1 - 0.30 / W,
                          top=1 - top / H, bottom=bot / H, wspace=0.05, hspace=0.07)
    for bi, cols in enumerate(blocks):
        r0 = bi * (n_ch + 1)
        for ci, (lab, j, observed) in enumerate(g.chans):
            r = r0 + ci
            ax = fig.add_subplot(gs[r, 0])
            im = g.truth_panel(ax, j, observed)
            if ci == 0:
                ax.set_title(truth_title, fontsize=F_TITLE, color=INK, pad=3)
            b = ax.get_position()
            fig.text(0.10 / W, b.y0 + b.height * 0.5, lab, fontsize=F_LABEL + 1,
                     color=INK, va="center", ha="left")
            fig.text(0.10 / W, b.y0 + b.height * 0.5 - 0.14 / H, obs_word(dataset, observed),
                     fontsize=F_SUB, color=INK_2, va="top", ha="left",
                     style="normal" if observed else "italic")
            for c_i, col in enumerate(cols, start=1):
                ax = fig.add_subplot(gs[r, c_i])
                g.method_panel(ax, col, lab, j, observed)
                if ci == 0:
                    ax.set_title(two_line(col[0]), fontsize=F_TITLE, color=INK, pad=3)
            cax = fig.add_subplot(gs[r, n_c])
            b = cax.get_position()
            cax.set_position([b.x0, b.y0 + 0.14 * b.height, b.width, 0.72 * b.height])
            g.colorbar(fig, cax, im, j)
    g.write(fig, W, H)


def make_cylinder_merged():
    """One column: the volume and surface sensing tasks side by side on the same
    held-out frame. Column groups are channels (u_y, p); inside each group the
    volume task is left and the surface task right, sharing one colorbar (both
    dumps carry the identical truth, mesh and normalization). Rows: truth with
    that task's sensors, then one method each."""
    roi = CYL_ROI_COMPACT
    gv = Gallery("cylinder", roi=roi, sensor_ms=1.0)
    gsf = Gallery("cylinder_surface", roi=roi, sensor_ms=0.7)
    for g in (gv, gsf):
        g.num_size = 6.5
    chans = [(r"$u_y$", 1), (r"$p$", 2)]
    for _, j in chans:
        assert np.allclose(gv.scale[j][:2], gsf.scale[j][:2]), "volume/surface truth differ"
    n_r = 1 + len(CYL_MERGED_ROWS)
    lab_w, gap, top, cb_h, bot, ws = 0.72, 0.10, 0.40, 0.07, 0.22, 0.05
    W = COL_W
    pw = (W - lab_w - 0.04 - gap) / (4 + 3 * ws)
    ph = pw * (roi[3] - roi[2]) / (roi[1] - roi[0])
    hs = 0.07
    H = top + n_r * ph + hs * ph * (n_r - 1) + cb_h + 0.05 + bot
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(n_r + 1, 5, width_ratios=[1, 1, gap / pw, 1, 1],
                          height_ratios=[1] * n_r + [cb_h / ph],
                          left=lab_w / W, right=1 - 0.04 / W,
                          top=1 - top / H, bottom=bot / H, wspace=ws, hspace=hs)
    rows = []
    for c0, (lab, j) in zip((0, 3), chans):
        im = None
        for dc, (g, task) in enumerate(((gv, "volume"), (gsf, "surface"))):
            col = c0 + dc
            observed = dict((jj, o) for _, jj, o in g.chans)[j]
            ax_truth = fig.add_subplot(gs[0, col])
            im = g.truth_panel(ax_truth, j, observed)
            ax_truth.set_title(task, fontsize=F_SUB, color=INK_2, pad=2)
            for r, (rlab, vol, srf) in enumerate(CYL_MERGED_ROWS, start=1):
                fname, key = vol if task == "volume" else srf
                name = rlab.split("\n")[0]
                ax = fig.add_subplot(gs[r, col])
                g.method_panel(ax, (name, fname, key), lab, j, observed)
                if col == 0:
                    b = ax.get_position()
                    fig.text(0.04 / W, b.y0 + b.height / 2, rlab, fontsize=F_LABEL - 0.5,
                             color=INK, va="center", ha="left", linespacing=1.1)
            if col == 0:
                b = ax_truth.get_position()
                fig.text(0.04 / W, b.y0 + b.height / 2, "truth +\nsensors", fontsize=F_LABEL - 0.5,
                         color=INK, va="center", ha="left", linespacing=1.1)
        # channel header centred over its two columns
        a0, a1 = gs[0, c0].get_position(fig), gs[0, c0 + 1].get_position(fig)
        fig.text((a0.x0 + a1.x1) / 2, a0.y1 + 0.20 / H, lab, fontsize=F_TITLE + 1, color=INK,
                 ha="center", va="bottom")
        cax = fig.add_subplot(gs[n_r, c0:c0 + 2])
        b = cax.get_position()
        cax.set_position([b.x0 + 0.1 * b.width, b.y0 - 0.03 / H, 0.8 * b.width, b.height])
        gv.colorbar(fig, cax, im, j, orientation="horizontal")
    for task, g in (("volume", gv), ("surface", gsf)):
        rows += [dict(sensing=task, **t) for t in g.table]
    write_figure(fig, COMPACT_STEM["cylinder_merged"], rows,
                 ["sensing", "channel", "method", "observed", "rel_l2"], "cylinder_merged", W, H)


def make_kolmogorov_compact(_ds="kolmogorov"):
    make_grid("kolmogorov", n_r=3, n_c=3, W=COL_W, lab_w=0.06, row_gap=0.24, hspace=0.26,
              cols=COMPACT_COLUMNS["kolmogorov"], stem=COMPACT_STEM["kolmogorov"])


def make_jhu_compact(_ds="jhu"):
    make_blocks("jhu", per_block=len(COMPACT_COLUMNS["jhu"]), cols=COMPACT_COLUMNS["jhu"],
                stem=COMPACT_STEM["jhu"], sensor_ms=1.6, truth_title="truth")


LAYOUT = {"kolmogorov": make_grid2x5, "cylinder": make_vertical,
          "cylinder_surface": make_vertical, "jhu": make_blocks}
# compact variants for the shortened manuscript; not in the default run. Tight
# padding keeps a one-column file at the column width, so fonts print at size.
COMPACT_RC = {"savefig.pad_inches": 0.02}
LAYOUT_COMPACT = {
    "cylinder_merged": lambda _ds: plt.rc_context(COMPACT_RC)(make_cylinder_merged)(),
    "kolmogorov_compact": lambda ds: plt.rc_context(COMPACT_RC)(make_kolmogorov_compact)(ds),
    "jhu_compact": lambda ds: plt.rc_context(COMPACT_RC)(make_jhu_compact)(ds)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(LAYOUT),
                    help=f"any of {list(LAYOUT) + list(LAYOUT_COMPACT)}")
    a = ap.parse_args()
    use_style()
    for ds in a.datasets:
        try:
            {**LAYOUT, **LAYOUT_COMPACT}[ds](ds)
        except SystemExit as e:
            print(e)
