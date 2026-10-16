"""Shared data loading + panel drawing for the cost-of-resolution figures.

Reads the scaling_*.json sweeps (H100 80 GB, bf16; benchmark_cost.py, job
16628458) for four model families:
  DMF-Gen (ours)  point-cloud generative FM   -- flat memory, linear time
  Senseiver       point deterministic          -- flat memory
  ConvAE/latent-FM latent conv grid            -- cubic memory, OOM walls
  Gen4Turb        voxel diffusion              -- cubic memory, earliest wall
Measured OOM walls: voxel train 320^3, conv train 640^3, conv infer 768^3.
Honest note: the conv decoder is CHEAPER than ours below the ~240^3 (inference)
/ ~470^3 (training) memory crossovers -- the flat curves win only at scale.
"""
import json

import numpy as np

from figures.pof.pof_style import MAIN, C_OURS, C_CONV, C_SENS, C_VOX, INK_2, MUTED

EV = (MAIN / "outputs/runs/JHU/pointcloud_ffm/"
      "iclr_jhu_xcube_spec02_DemoN29_20260822_140100/Evaluation")
H100 = 81559  # MB

FAM = {
    "ours": dict(c=C_OURS, m="o", lab="DMF-Gen (ours; point, generative)"),
    "conv": dict(c=C_CONV, m="s", lab="ConvAE / latent-FM (latent conv grid)"),
    "sens": dict(c=C_SENS, m="^", lab="Senseiver (point, deterministic)"),
    "vox": dict(c=C_VOX, m="D", lab="Gen4Turb (voxel diffusion)"),
}
# short legend labels for the one-column figure
FAM_SHORT = {"ours": "DMF-Gen (point)", "conv": "latent-FM AE (grid)",
             "sens": "Senseiver (point)", "vox": "Gen4Turb (voxel)"}


def _load(name):
    return json.load(open(EV / f"{name}.json"))


def _split(rows, tkey, model=None):
    if model is not None:
        rows = [r for r in rows if r.get("model") == model]
    ok = [r for r in rows if r.get("ok")]
    oom = [r["n_points"] for r in rows
           if not r.get("ok") and r.get("error") == "OOM"]
    return dict(n=[r["n_points"] for r in ok],
                mem=[r["peak_mb"] for r in ok],
                t=[r[tkey] for r in ok], oom=oom)


def load_series():
    s = {
        ("ours", "infer"): _split(_load("scaling_ours_infer"), "seconds"),
        ("ours", "train"): _split(_load("scaling_ours_train"), "sec_per_step"),
        ("conv", "infer"): _split(_load("scaling_convae_infer"), "seconds"),
        ("conv", "train"): _split(_load("scaling_convae_train"), "sec_per_step"),
        ("sens", "infer"): _split(_load("scaling_senseiver"), "seconds",
                                  "senseiver_infer"),
        ("sens", "train"): _split(_load("scaling_senseiver"), "sec_per_step",
                                  "senseiver_train"),
        ("vox", "infer"): _split(_load("scaling_gen4turb"), "seconds",
                                 "gen4turb_infer_step"),
        ("vox", "train"): _split(_load("scaling_gen4turb"), "sec_per_step",
                                 "gen4turb_train"),
    }
    # conv training wall measured separately at 640^3 / 768^3
    s[("conv", "train")]["oom"] += _split(_load("scaling_convae_train_wall"),
                                          "sec_per_step")["oom"]
    # voxel full sample = 32 diffusion steps
    s[("vox", "infer")]["t"] = [32 * t for t in s[("vox", "infer")]["t"]]
    return s


def crossover(sa, sb):
    """First n where family b's memory exceeds family a's flat memory."""
    flat = np.median(sa["mem"])
    n, m = np.array(sb["n"], float), np.array(sb["mem"], float)
    i = np.searchsorted(m, flat)
    f = (np.log(flat) - np.log(m[i - 1])) / (np.log(m[i]) - np.log(m[i - 1]))
    return float(np.exp(np.log(n[i - 1]) + f * (np.log(n[i]) - np.log(n[i - 1]))))


def draw(ax, s, mode, ykey, first_wall_only=True, labels=None, ms=3.2, wall_ms=8):
    """One log-log panel; ykey in {'mem','t'}; labels overrides the legend text."""
    for fam in ("ours", "conv", "sens", "vox"):
        d, f = s[(fam, mode)], FAM[fam]
        ax.plot(d["n"], d[ykey], marker=f["m"], ms=ms, lw=1.2, color=f["c"],
                label=f["lab"] if labels is None else labels[fam], zorder=3)
        if ykey == "mem" and d["oom"]:
            walls = sorted(d["oom"])[:1] if first_wall_only else sorted(d["oom"])
            ax.plot(walls, [H100] * len(walls), marker="x", ms=wall_ms, mew=2.2 * wall_ms / 8,
                    color=f["c"], ls="none", zorder=4, clip_on=False)
    if ykey == "mem":
        ax.axhline(H100, color="k", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)


def res_ticks(ax, res=(64, 128, 256, 512, 1024)):
    ax.set_xticks([r ** 3 for r in res])
    ax.set_xticklabels([f"${r}^3$" for r in res])
    ax.set_xticks([], minor=True)


def annotate_h100(ax, x=3e5, below=False, fontsize=6.2, text="H100 80 GB"):
    y = H100 * (0.62 if below else 1.25)
    ax.text(x, y, text, fontsize=fontsize, color="k",
            va="top" if below else "bottom")
