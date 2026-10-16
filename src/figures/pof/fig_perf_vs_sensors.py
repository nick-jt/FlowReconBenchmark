"""Performance vs number of sensor points, 2D and 3D side by side.

Left: 2D Kolmogorov 256^2 (single observed channel, vorticity).
Right: 3D JHU 125^3 cross-cube (Ux/Uz observed; solid = observed-channel
rel-L2, dashed = aggregate over all four channels).

The panels are deliberately the two "opposite sides" of the density story:
at the SAME sensor count the 2D error collapses toward the interpolation
limit while the 3D aggregate saturates against the unobserved-channel
identifiability wall (200x more sensors move DMF-Gen's aggregate only
0.68 -> 0.54, while IDW's observed-channel error keeps falling 0.31 -> 0.08).

Sources (read-only):
  $MAIN/Save_TrainedModel/JHU/baseline_classical/classical_baselines_sweep.json
  $MAIN/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*/
      Evaluation/calib_sweep_nfe4_n*_K8.json   (DMF-Gen, K=8, 50 snapshots)
  $MAIN/Save_TrainedModel/kolmogorov2d/baseline_classical/
      classical_baselines_sweep_nonperiodic_2d.json   (added when job lands;
      falls back to the single-density main_1pct json with a note)
Learned 2D rows are appended automatically once their eval JSONs exist
(scan pattern below) -- rerun this script after the 2D fleet finishes.

--compact: one-column figure of the Kolmogorov panel only (perf_vs_sensors_compact),
with SiT-point and FNO in the colours of the spectra figures.
"""
from paths import PAPER_DIR, REPO_ROOT

import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN = f"{REPO_ROOT}"
# 2D classical runs write into the worktree's Save_TrainedModel, not $MAIN.
WT = str(REPO_ROOT)
OUT = str(PAPER_DIR / "figures")

# Validated categorical palette (see check_palette.py); fixed method order.
C = {
    # learned fleet -- categorical slots, same family colors as every other figure
    "dmfgen": "#2a78d6",
    "dmfgencq": "#2a78d6",   # the Kolmogorov DMF-Gen row
    "latentfm": "#eb6834",
    "senseiver": "#1baf7a",
    "geofno": "#eda100",
    "sit": "#7a5cc6",
    "mlprbf": "#d6446e",
    "s3gm": "#3f8f8a",
    # classical floors -- neutral, separated by linestyle. Deliberately gray:
    # the panel's message is learned-vs-floor, and reusing the categorical
    # slots for IDW/POD would collide with latent-FM and Geo-FNO.
    "idw": "#55534f",
    "kdtree": "#8a887f",
    "pod": "#b0aea5",
    "lse": "#6f6d66",
    "constant": "#9a9890",
    "grid": "#e1e0d9",
    "ink": "#3a3a37",
    "muted": "#8a887f",
}

N2D = 256 * 256
N3D = 125 ** 3
COMPACT = "--compact" in sys.argv
if COMPACT:      # SiT-point and FNO as in fig_spectra_kolmogorov / fig_spectra_stats
    C.update(sit="#eda100", geofno="#4a3aa7")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def classical_curves(path, obs_fields):
    res = json.load(open(path))["results"]
    out = {}
    for key, v in res.items():
        m = re.match(r"([a-z_]+)_n(\d+)$", key)
        if not m:
            continue
        meth, n = m.group(1), int(m.group(2))
        pf = v["per_field"]
        vals = [pf[f]["rel_l2_mean"] for f in obs_fields if f in pf]
        if vals:
            out.setdefault(meth, {})[n] = float(np.mean(vals))
    return {m: sorted(d.items()) for m, d in out.items()}


def dmfgen_curves():
    pat = os.path.join(
        MAIN, "outputs/runs/JHU/pointcloud_ffm",
        "iclr_jhu_xcube_spec02_DemoN29_*", "Evaluation", "calib_sweep_nfe4_n*_K8.json")
    obs, agg = {}, {}
    for f in glob.glob(pat):
        j = json.load(open(f))
        n = int(re.search(r"_n(\d+)_K8", f).group(1))
        per_snap_obs = [
            np.mean([s["per_field"]["Ux"]["rel_l2_mean"], s["per_field"]["Uz"]["rel_l2_mean"]])
            for s in j["snapshots"]
        ]
        obs[n] = float(np.mean(per_snap_obs))
        agg[n] = float(j["summary"]["rel_l2_mean"])
    return sorted(obs.items()), sorted(agg.items())


# 2D Kolmogorov learned fleet: each row writes one JSON per sensor density.
# Every density point of a model comes from its REPORTED family (fleet_select) and is
# placed by the n_obs recorded inside the payload, never by the filename: the canonical
# 655 row and the density-override rows (kolm_fleet_ovr_n...) qualify once they carry
# the frozen protocol's clean stamp.
from eval.fleet_select import REPORTED_FAMILY as _RF
LEARNED_2D = {m: (_RF[m]["kolmogorov2d"] if isinstance(_RF[m], dict) else _RF[m], "*.json")
              for m in ("dmfgencq", "senseiver", "mlprbf", "geofno", "sit", "latentfm", "s3gm")}
LEARNED_LABEL = {"dmfgencq": "DMF-Gen", "senseiver": "Senseiver",
                 "mlprbf": "MLP-RBF", "geofno": "FNO", "sit": "SiT-point",
                 "latentfm": "latent FM", "s3gm": "S3GM"}


def learned_2d_curves():
    """{method: [(n, rel_L2), ...]} for whatever has landed. Rows still running
    are simply absent, so the figure is re-runnable as jobs finish. A point
    qualifies only on the frozen protocol's clean stamp, TEST split, no clamp,
    the canonical K (8 generative / 1 deterministic) and 50 records; two files
    at one density must agree or the figure refuses to choose."""
    from protocol import PROTOCOL_2D
    det = {"senseiver", "mlprbf", "geofno"}
    out = {}
    for meth, (fam, pat) in LEARNED_2D.items():
        pairs = {}
        for root in (WT, MAIN):
            for f in glob.glob(os.path.join(root, "outputs/runs/kolmogorov2d",
                                            fam, "*", "Evaluation", pat)):
                try:
                    j = json.load(open(f))
                except Exception:
                    continue
                if not isinstance(j, dict) or "summary" not in j:
                    continue
                if j.get("protocol") != PROTOCOL_2D or j.get("split") != "test":
                    continue
                if j.get("clamp_hard") is not False or len(j.get("snapshots") or []) != 50:
                    continue
                if int(j.get("K", -1)) != (1 if meth in det else 8):
                    continue
                n = j.get("n_obs")
                if isinstance(n, list):
                    if len(n) != 1:
                        continue
                    n = int(n[0])
                elif n is None:
                    continue
                v = float(j["summary"]["aggregate"]["rel_l2_mean"])
                if n in pairs and abs(pairs[n] - v) > 1e-9:
                    raise SystemExit(f"[perf_vs_sensors] {meth} n={n}: two canonical files disagree "
                                     f"({pairs[n]:.4f} vs {v:.4f}, {os.path.basename(f)})")
                pairs[n] = v
        if pairs:
            out[meth] = sorted(pairs.items())
    return out


def plot_series(ax, pairs, color, label, ls="-", marker="o"):
    if not pairs:
        return
    ns, ys = zip(*pairs)
    ax.plot(ns, ys, linestyle=ls, color=color, lw=1.4, marker=marker, ms=3.5,
            mfc="white", mew=1.1, mec=color, label=label, zorder=3)


def style(ax, n_total, label):
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.grid(True, color=C["grid"], lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("sensor points per observed field")
    top = ax.secondary_xaxis("top", functions=(lambda n: 100 * n / n_total,
                                               lambda p: p * n_total / 100))
    top.set_xlabel("% of grid points", fontsize=8, color=C["muted"])
    top.tick_params(labelsize=8, colors=C["muted"])
    top.spines["top"].set_visible(False)


if COMPACT:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.8})
    fig, axA = plt.subplots(figsize=(3.4, 3.3))
    axB = None
else:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)
FS = 6.8 if COMPACT else 7.5      # annotation font

# ---- Panel A: 2D Kolmogorov -------------------------------------------------
sweep2d = os.path.join(WT, "outputs/runs/kolmogorov2d/baseline_classical",
                       "classical_baselines_sweep_nonperiodic_2d.json")
single2d = os.path.join(WT, "outputs/runs/kolmogorov2d/baseline_classical",
                        "classical_baselines_main_1pct_nonperiodic_2d.json")
if os.path.exists(sweep2d):
    c2 = classical_curves(sweep2d, ["vorticity"])
else:
    c2 = classical_curves(single2d, ["vorticity"])
    axA.annotate("full sweep job queued;\nsingle-density points shown",
                 xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7.5,
                 color=C["muted"])
# classical floors first, so the learned fleet draws on top of them
plot_series(axA, c2.get("idw", []), C["idw"], "IDW $k{=}8$", ls="--", marker="^")
plot_series(axA, c2.get("kdtree", []), C["kdtree"], "nearest sensor",
            ls="-.", marker="v")
plot_series(axA, c2.get("gappy_pod", []), C["pod"],
            "gappy POD" if COMPACT else "gappy POD (rank per density)", ls=":", marker="s")
lse2d = os.path.join(WT, "outputs/runs/kolmogorov2d/baseline_classical", "classical_baselines_sweep_lse_2d.json")
if os.path.exists(lse2d):
    plot_series(axA, classical_curves(lse2d, ["vorticity"]).get("lse", []), C["lse"], "LSE",
                ls=(0, (3, 1, 1, 1)), marker="D")
if "constant" in c2:
    plot_series(axA, c2["constant"], C["constant"], "train mean", ls=":", marker="")
else:
    axA.axhline(1.0, color=C["constant"], ls=":", lw=1.0)
    # centred: the top-right corner now carries the off-scale Geo-FNO value
    axA.annotate("train mean", xy=(0.80, 0.885) if COMPACT else (0.50, 0.965),
                 xycoords="axes fraction", ha="center", fontsize=FS, color=C["constant"])

# learned fleet: all seven rows, plotted in a fixed order so the legend is stable
L2 = learned_2d_curves()
for meth in ("dmfgencq", "sit", "geofno", "latentfm", "mlprbf", "senseiver", "s3gm"):
    plot_series(axA, L2.get(meth, []), C[meth], LEARNED_LABEL[meth])
# points above the axis are marked with their value rather than rescaling the
# panel for one collapse (the budget-matched Geo-FNO reaches 1.42 at 10%)
for meth, pairs in L2.items():
    for n_, v_ in pairs:
        if v_ > 1.05:
            if not COMPACT:  # the compact panel extends the axis and shows the points
                axA.annotate(f"{v_:.2f}", xy=(n_, 1.045), xytext=(n_, 0.955), ha="center",
                             fontsize=FS, color=C[meth],
                             arrowprops=dict(arrowstyle="->", color=C[meth], lw=0.8))
# A row with one or two densities is a point, not a curve, and saying nothing
# about it would let a single marker read as a finished sweep.
partial = [f"{LEARNED_LABEL[m]} ({len(L2.get(m, []))}/5)"
           for m in LEARNED_2D if len(L2.get(m, [])) < 5]
if COMPACT:
    partial = [f"{LEARNED_LABEL[m]}: {L2[m][0][0]} sensors only"
               for m in LEARNED_2D if len(L2.get(m, [])) == 1]
if partial:
    axA.annotate(("" if COMPACT else "sweep incomplete: ") + ", ".join(partial),
                 xy=(0.03, 0.04), xycoords="axes fraction", fontsize=FS, color=C["muted"])

# the crossover is the panel's finding: below it the learned fleet beats
# interpolation, above it plain IDW is the best method on the plot
idw = dict(c2.get("idw", []))
best_learned = {n: min(v for m, pairs in L2.items() for k, v in pairs if k == n)
                for n in idw if any(k == n for pairs in L2.values() for k, _ in pairs)}
cross = [n for n in sorted(best_learned) if idw[n] < best_learned[n]]
if cross:
    n0 = cross[0]
    axA.annotate(f"at {n0} sensors plain IDW\nbeats every learned row",
                 xy=(n0, idw[n0]), xytext=(0.42, 0.70), textcoords="axes fraction",
                 fontsize=7.5, color=C["ink"], ha="left",
                 arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.6,
                                 shrinkB=2))

axA.set_xlim(45, 9000)  # the eventual sweep range {65..6554}, so single points sit in context
style(axA, N2D, "" if COMPACT else "(a)")
axA.set_ylabel("relative $L_2$ error")
if COMPACT:
    axA.set_ylim(0, 1.15)       # shows MLP-RBF (1.09) and nearest sensor (1.10) at 65 sensors

if COMPACT:
    # one panel: no 2D-3D matched-count guide; legend below the axes
    h, l = axA.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, ncol=3, handlelength=1.9,
               columnspacing=0.9, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.savefig(os.path.join(OUT, "perf_vs_sensors_compact.pdf"))
    fig.savefig(os.path.join(OUT, "perf_vs_sensors_compact.png"), dpi=220)
    print("wrote", os.path.join(OUT, "perf_vs_sensors_compact.pdf"))
    sys.exit(0)

# ---- Panel B: 3D JHU --------------------------------------------------------
_jhu_cls = os.path.join(MAIN, "outputs/runs/JHU/baseline_classical", "classical_baselines_sweep.json")
if os.path.exists(_jhu_cls):
    c3 = classical_curves(_jhu_cls, ["Ux", "Uz"])
    d_obs, d_agg = dmfgen_curves()
else:                       # JHU sweep products are not on this host
    c3, d_obs, d_agg = {}, [], []
    axB.annotate("JHU sweep JSONs not on this host", xy=(0.05, 0.9), xycoords="axes fraction",
                 fontsize=7.5, color=C["muted"])
plot_series(axB, d_obs, C["dmfgen"], "DMF-Gen (observed)")
plot_series(axB, d_agg, C["dmfgen"], "DMF-Gen (all channels)", ls="--", marker="s")
plot_series(axB, c3.get("idw", []), C["idw"], "IDW $k{=}8$", ls="--", marker="^")
plot_series(axB, c3.get("kdtree", []), C["kdtree"], "nearest sensor",
            ls="-.", marker="v")
plot_series(axB, c3.get("gappy_pod", []), C["pod"], "gappy POD $r{=}80$",
            ls=":", marker="s")
style(axB, N3D, "(b)")

# identifiability-wall annotation on the dashed aggregate curve
if d_agg:
    n_hi, y_hi = d_agg[-1]
    axB.annotate("unobserved-channel wall:\n$200\\times$ more sensors,\naggregate only "
                 f"{d_agg[0][1]:.2f}$\\rightarrow${y_hi:.2f}",
                 xy=(n_hi, y_hi), xytext=(0.44, 0.80), textcoords="axes fraction",
                 fontsize=7.5, color=C["ink"],
                 arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.6))

# matched-count guide across both panels
for ax, gap, frac in ((axA, (N2D / 1953) ** 0.5, 100 * 1953 / N2D),
                      (axB, (N3D / 1953) ** (1 / 3), 100 * 1953 / N3D)):
    ax.axvline(1953, color=C["muted"], lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.annotate(f"N=1953 ({frac:.2g}%)\nmean gap {gap:.1f} cells",
                xy=(1953, 0.02), xytext=(3, 2), textcoords="offset points",
                fontsize=7.5, color=C["muted"], va="bottom")

# one shared legend below both panels (colors are consistent across panels)
handles, labels = axB.get_legend_handles_labels()
for h, l in zip(*axA.get_legend_handles_labels()):
    if l not in labels:
        handles.append(h)
        labels.append(l)
fig.legend(handles, labels, loc="lower center", frameon=False, ncol=5,
           handlelength=1.9, columnspacing=1.1, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(w_pad=1.2, rect=(0, 0.13, 1, 1))
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "perf_vs_sensors.pdf"))
fig.savefig(os.path.join(OUT, "perf_vs_sensors.png"), dpi=220)
print("wrote", os.path.join(OUT, "perf_vs_sensors.pdf"))
