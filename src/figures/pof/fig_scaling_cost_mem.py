"""PoF figure: main-text memory-only cost panel (1x2).

The two memory panels carry the OOM-wall claim; timing panels live in the
appendix version (fig_scaling_cost.py). --compact writes the one-column version
(scaling_cost_mem_compact).
"""
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures.pof.pof_style import use_style, save, FULL_W, COL_W, C_OURS, C_SENS, INK_2
from figures.pof.scaling_common import (load_series, draw, res_ticks, annotate_h100,
                                            crossover, FAM_SHORT)

use_style()
s = load_series()

def figure_full():
    """The paper's full-width two-panel figure (scaling_cost_mem)."""
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.3), sharex=True,
                           constrained_layout=True)

    a = ax[0]
    draw(a, s, "infer", "mem")
    a.set_ylabel("peak GPU memory (MB)")
    a.set_xlabel("output points $N$")
    a.set_title("(a)", loc="left", fontsize=10, fontweight="bold")
    a.annotate("flat 1.4 GB, $64^3$–$1024^3$", xy=(s[("ours", "infer")]["n"][-1],
               s[("ours", "infer")]["mem"][-1]), xytext=(0.47, 0.10),
               textcoords="axes fraction", fontsize=7.5, color=C_OURS,
               arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.7))
    nc = crossover(s[("ours", "infer")], s[("conv", "infer")])
    a.plot(nc, 1385, "o", ms=5, mfc="none", mec=INK_2, mew=0.8)
    a.annotate(f"ConvAE cheaper below $\\approx${round(nc**(1/3)/10)*10}$^3$",
               xy=(nc, 1385), xytext=(0.04, 0.60), textcoords="axes fraction",
               fontsize=7.5, color=INK_2,
               arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
    a.legend(loc="upper left", fontsize=7.5, frameon=True)

    a = ax[1]
    draw(a, s, "train", "mem")
    annotate_h100(a)
    a.set_xlabel("field points $N$")
    a.set_title("(b)", loc="left", fontsize=10, fontweight="bold")
    a.text(0.97, 0.30, "DMF-Gen: flat 39.9 GB (39k queries/step)",
           transform=a.transAxes, ha="right", fontsize=7.5, color=C_OURS)
    a.text(0.97, 0.22, "Senseiver: flat 19.4 GB (batch 20)",
           transform=a.transAxes, ha="right", fontsize=7.5, color=C_SENS)
    nct = crossover(s[("ours", "train")], s[("conv", "train")])
    a.annotate(f"ConvAE cheaper below $\\approx${round(nct**(1/3)/10)*10}$^3$",
               xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7.5, color=INK_2)

    for a in ax.flat:
        res_ticks(a)

    save(fig, "scaling_cost_mem")


def figure_compact():
    """One-column version for the shortened manuscript (scaling_cost_mem_compact):
    the same two memory panels sharing both axes, short labels, legend below."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.8, "savefig.pad_inches": 0.02})
    fig, ax = plt.subplots(1, 2, figsize=(COL_W, 2.55), sharex=True, sharey=True,
                           constrained_layout=True)
    nc = crossover(s[("ours", "infer")], s[("conv", "infer")])
    nct = crossover(s[("ours", "train")], s[("conv", "train")])
    for a, mode, title in ((ax[0], "infer", "(a) sampling"), (ax[1], "train", "(b) training")):
        draw(a, s, mode, "mem", labels=FAM_SHORT, ms=2.6, wall_ms=6)
        annotate_h100(a, x=2.6e5, fontsize=6.5, text="80 GB")
        res_ticks(a, res=(64, 256, 1024))
        a.set_xlabel("output points $N$", labelpad=1)
        a.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax[0].set_ylabel("peak GPU memory (MB)")
    ax[0].plot(nc, 1385, "o", ms=4, mfc="none", mec=INK_2, mew=0.8)
    ax[0].annotate(f"AE smaller\nbelow $\\approx${round(nc**(1/3)/10)*10}$^3$", xy=(nc, 1385),
                   xytext=(0.05, 0.62), textcoords="axes fraction", fontsize=6.5, color=INK_2,
                   arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6))
    # flat-memory labels sit on their own lines where no rising grid curve passes:
    # the right end in (a), the left end in (b); point model above, Senseiver below
    for a, fam, mode, up, x, ha in ((ax[0], "ours", "infer", True, 1024 ** 3, "right"),
                                    (ax[1], "ours", "train", True, 1.08 * 64 ** 3, "left"),
                                    (ax[1], "sens", "train", False, 1.08 * 64 ** 3, "left")):
        y = float(np.median(s[(fam, mode)]["mem"]))
        a.text(x, y * (1.18 if up else 1 / 1.18), f"{y / 1e3:.1f} GB", ha=ha,
               va="bottom" if up else "top", fontsize=6.5, color=C_OURS if fam == "ours" else C_SENS)
    ax[1].text(0.04, 0.05, f"AE smaller below $\\approx${round(nct**(1/3)/10)*10}$^3$",
               transform=ax[1].transAxes, fontsize=6.5, color=INK_2)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False, fontsize=6.8,
               handlelength=1.8, columnspacing=1.0)
    print(f"[crossover] sampling {nc:.3g} points (~{nc**(1/3):.0f}^3), "
          f"training {nct:.3g} points (~{nct**(1/3):.0f}^3)")
    save(fig, "scaling_cost_mem_compact")


figure_compact() if "--compact" in sys.argv else figure_full()
