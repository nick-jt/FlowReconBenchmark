"""Headline comparison figures for the PoF benchmark paper.

Outputs (Paper/pof2026/figures/): pareto_cost_accuracy.pdf, capability_heatmap.pdf
Data sources: the main-text JHTDB tables (Pareto) and
HEADLINE_COMPARISON_2026-09-04.md (capability verdicts). Values are inlined
here with a comment naming the source row; regenerate by editing in place.

Design follows the dataviz skill: color assigned per method FAMILY (fixed
order, validated reference palette), classical floors in neutral gray, thin
marks, direct labels, recessive grid, no dual axes, three-state heatmap with
glyph secondary encoding (color is never the only carrier).
"""
from paths import PAPER_DIR

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = str(PAPER_DIR / "figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3e0"

FAM_COLORS = {
    "deterministic": "#2a78d6",  # slot 1 blue
    "latent generative": "#eb6834",  # slot 2 orange
    "ambient generative": "#1baf7a",  # slot 3 aqua
    "classical": "#6e6d69",  # neutral: reference floors
}
FAM_MARKERS = {
    "deterministic": "s",
    "latent generative": "^",
    "ambient generative": "D",
    "classical": "o",
}

# (label, family, CRPS, infer s/field, infer peak GB or None, note)
# From the main-text JHTDB tables (accuracy: outputs/paper/tab_jhu_numbers.json, TEST half;
# inference time: the 3D cost table); CoNFiLD is the single-snapshot arm.
PARETO = [
    ("Latent FM", "latent generative", 0.237, 0.019, 0.6, ""),
    ("FNO3D-FM", "ambient generative", 0.272, 0.097, 6.7, ""),
    ("SiT-point", "ambient generative", 0.323, 149.0, None, ""),
    ("DMF-Gen", "ambient generative", 0.333, 4.08, 1.39, ""),
    ("CoNFiLD", "latent generative", 0.378, 36.0, None, ""),
    ("FNO3D", "deterministic", 0.392, 0.022, 2.2, ""),
    ("IDW k=8", "classical", 0.398, 0.68, None, "CPU"),
    ("KD-tree", "classical", 0.408, 0.16, None, "CPU"),
    ("Gen4Turb", "ambient generative", 0.409, 1.65, 1.4, ""),
    ("Senseiver", "deterministic", 0.487, 0.9, 1.19, ""),
    ("Set-DeepONet", "deterministic", 0.571, 0.23, 3.0, ""),
    ("LSE", "classical", 0.636, 0.23, None, "CPU"),
    ("Gappy POD r80", "classical", 0.676, 0.14, None, "CPU"),
]

LABEL_OFFSETS = {  # (dx, dy) in offset points, tuned against collisions
    "Latent FM": (6, 5),
    "FNO3D-FM": (6, 5),
    "SiT-point": (-7, -13),
    "DMF-Gen": (-7, 5),
    "CoNFiLD": (-7, 6),
    "FNO3D": (6, -10),
    "IDW k=8": (-7, 2),
    "KD-tree": (-4, 8),
    "Gen4Turb": (7, 5),
    "Senseiver": (6, 4),
    "Set-DeepONet": (6, 4),
    "LSE": (6, -10),
    "Gappy POD r80": (6, 4),
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def pareto_figure():
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURFACE)
    style_axes(ax)
    for label, fam, crps, sec, gb, note in PARETO:
        size = 30 if gb is None else 24 + 26 * gb  # area encodes peak GB where measured
        ax.scatter(
            sec, crps, s=size, marker=FAM_MARKERS[fam], color=FAM_COLORS[fam],
            edgecolors=SURFACE, linewidths=1.2, zorder=3,
        )
        dx, dy = LABEL_OFFSETS[label]
        ax.annotate(
            label, (sec, crps), textcoords="offset points", xytext=(dx, dy),
            fontsize=8.5, color=INK, ha="left" if dx > 0 else "right", zorder=4,
        )
    # The Pareto frontier of this plane is the single latent-FM point — it is
    # both fastest and lowest-CRPS at 125^3. Annotate rather than draw a
    # degenerate staircase; the tradeoffs live on the capability/scaling axes.
    ax.annotate(
        "Pareto-optimal in this plane\n(loses grid-free / scaling axes)",
        (0.019, 0.237), textcoords="offset points", xytext=(2, -24),
        fontsize=8, color=INK2, style="italic",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Inference wall-time per full field (s, log)  —  JHU 125$^3$, 1.95M points", fontsize=9.5, color=INK)
    ax.set_ylabel("CRPS (lower is better)", fontsize=9.5, color=INK)
    ax.set_ylim(0.14, 0.70)
    handles = [
        Line2D([], [], marker=FAM_MARKERS[f], color=FAM_COLORS[f], linestyle="",
               markersize=6, label=f)
        for f in FAM_COLORS
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right",
              labelcolor=INK2, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pareto_cost_accuracy.pdf"));fig.savefig(os.path.join(OUT, "pareto_cost_accuracy.png"), dpi=150)
    plt.close(fig)


# Capability matrix: 2 = yes, 1 = partial, 0 = no. Interface PROPERTIES (the
# unstructured-mesh column is a capability of the interface, exercised here
# only by the cylinder's body-fitted mesh). Column keys map to
# HEADLINE_COMPARISON_2026-09-04.md sections 3-6.
# The former "Operator-robust" column (DMF-Gen alone marked yes) was dropped on
# 2026-09-14: the measured Kolmogorov operator table shows no binary split --
# under sigma=0.3 noise DMF-Gen loses 8% and Geo-FNO 10%, under 25% slab
# occlusion 1.35x vs 1.33x -- so a yes/no cell was not supported by the data.
# Operator behavior is reported as the graded table (tab:operator) instead.
CAP_COLS = [
    "Grid-free\ntraining", "Arbitrary-point\nquery", "Resolution\ntransfer",
    "Unstructured\nmesh (interface)", "Generative\nUQ", "Post-hoc\nrecalibrable",
    "Dissipation-\npreserving", "Memory flat\nvs resolution",
    "Sub-second\ninference",
]
CAP_ROWS = [
    ("DMF-Gen", "ambient generative", [2, 2, 2, 2, 2, 2, 2, 2, 0]),
    ("SiT-point", "ambient generative", [2, 1, 2, 2, 2, 2, 0, 2, 0]),
    ("Gen4Turb", "ambient generative", [0, 0, 0, 0, 2, 2, 0, 0, 1]),
    ("S3GM", "ambient generative", [0, 0, 0, 0, 2, 2, 0, 0, 0]),
    ("Latent FM", "latent generative", [0, 0, 0, 0, 2, 2, 0, 0, 2]),
    ("CoNFiLD", "latent generative", [2, 2, 1, 1, 2, 2, 0, 1, 0]),
    ("Senseiver", "deterministic", [2, 2, 2, 2, 0, 0, 0, 2, 2]),
    ("FNO3D", "deterministic", [0, 0, 1, 0, 1, 2, 0, 0, 2]),
    ("DeepONet(++)", "deterministic", [2, 2, 2, 1, 0, 0, 0, 2, 2]),
    ("IDW / KD-tree", "classical", [2, 2, 2, 2, 0, 0, 0, 2, 2]),
    ("Gappy POD", "classical", [2, 2, 1, 2, 0, 0, 0, 2, 2]),
]
CAP_FILL = {2: "#2a78d6", 1: "#c2d9f3", 0: "#efeeec"}
CAP_GLYPH = {2: "●", 1: "◐", 0: "○"}  # ● ◐ ○ secondary encoding
CAP_GLYPH_INK = {2: SURFACE, 1: "#2a5786", 0: "#a5a49f"}


def capability_figure():
    n_r, n_c = len(CAP_ROWS), len(CAP_COLS)
    fig, ax = plt.subplots(figsize=(7.9, 0.42 * n_r + 1.75), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (name, fam, vals) in enumerate(CAP_ROWS):
        y = n_r - 1 - i
        for j, v in enumerate(vals):
            ax.add_patch(plt.Rectangle((j + 0.03, y + 0.03), 0.94, 0.94,
                                       facecolor=CAP_FILL[v], edgecolor=SURFACE, linewidth=1.5))
            ax.text(j + 0.5, y + 0.47, CAP_GLYPH[v], ha="center", va="center",
                    fontsize=9, color=CAP_GLYPH_INK[v])
        ax.text(-0.15, y + 0.5, name, ha="right", va="center", fontsize=8, color=INK)
        ax.plot([-2.55], [y + 0.5], marker=FAM_MARKERS[fam], color=FAM_COLORS[fam],
                markersize=4.5, clip_on=False)
    for j, col in enumerate(CAP_COLS):
        ax.text(j + 0.30, n_r + 0.12, col.replace("\n", " "), ha="left", va="bottom",
                fontsize=7, color=INK2, rotation=32, rotation_mode="anchor")
    ax.set_xlim(-2.8, n_c)
    ax.set_ylim(-0.7, n_r + 2.4)
    ax.axis("off")
    ax.text(0.0, -0.45, "● yes    ◐ partial / adapted    ○ no",
            fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "capability_heatmap.pdf"));fig.savefig(os.path.join(OUT, "capability_heatmap.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pareto_figure()
    capability_figure()
    print("wrote", os.path.join(OUT, "pareto_cost_accuracy.pdf"))
    print("wrote", os.path.join(OUT, "capability_heatmap.pdf"))
