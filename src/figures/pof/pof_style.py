"""Shared style for the PoF 2026 benchmark figures.

Categorical palette: first three slots of the validated dataviz reference
palette (light mode, fixed order -- the triple passes all-pairs CVD and
normal-vision floors); the fourth family (voxel diffusion, the de-emphasized
reference baseline) is neutral gray, with marker shape as secondary encoding.
Sequential/diverging maps are perceptually uniform (no jet).
"""
from paths import REPO_ROOT
import os
from pathlib import Path

import matplotlib

FULL_W = 7.0   # inches, PoF full-width figure
COL_W = 3.4    # inches, single column

# family colors (consistent across every figure)
C_OURS = "#2a78d6"   # slot 1 blue   : DMF-Gen (ours, point generative)
C_CONV = "#eb6834"   # slot 2 orange : latent conv AE / latent-FM
C_SENS = "#1baf7a"   # slot 3 aqua   : Senseiver (point deterministic)
C_VOX = "#898781"    # neutral gray  : Gen4Turb voxel diffusion (de-emphasis)
C_TRUTH = "#0b0b0b"
INK_2 = "#52514e"    # secondary ink
MUTED = "#898781"

# main checkout with the read-only source data
MAIN = Path(os.environ.get(
    "POF_MAIN",
    f"{REPO_ROOT}"))

# worktree output directory (…/0_demo_TurbulentCombustion/Paper/pof2026/figures)
OUT = REPO_ROOT / "outputs" / "paper" / "figures"


def use_style():
    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": "#e1e0d9",       # solid hairline grid, one shade off white
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.85",
        "savefig.bbox": "tight",
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, stem, pdf_dpi=200, png_dpi=150):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", dpi=pdf_dpi)
    fig.savefig(OUT / f"{stem}.png", dpi=png_dpi)
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        print(f"wrote {p}  ({p.stat().st_size/1e6:.2f} MB)")
