"""PoF figure: 2D Kolmogorov energy spectra + band-ratio bars.

WINDOWING NOTE (read this before comparing with fig_spectra_stats.py).
Kolmogorov flow lives on the doubly periodic box [0, 2pi)^2 and the saved
256x256 field IS the full period, so the DFT basis is exact: there is no edge
discontinuity, no spectral leakage, and NO WINDOW IS APPLIED HERE.  This is a
deliberate difference from the 3D spectra in this paper (spectra_stats.pdf),
where the JHU/FireBench fields are non-periodic sub-cutouts of a larger DNS and
therefore MUST be Hann-windowed (src/core/spectral_utils.py) to suppress a broadband
leakage floor.  Windowing a genuinely periodic field would only inject an
avoidable low-k bias, so it is omitted -- the two estimators differ because the
boundary conditions differ, not because the protocol is inconsistent.

Spectrum definition.  The saved field is the scalar vorticity omega.  For 2D
incompressible flow the enstrophy spectrum is Z(k) = sum_shell |omega_hat|^2 / 2
and the KINETIC ENERGY spectrum follows from omega_hat = i k x u_hat, i.e.
    E(k) = sum_{|k'| in shell} |omega_hat(k')|^2 / (2 |k'|^2).
That is what is plotted and what the band ratios integrate.  omega_hat is
normalised by N^2 so Parseval gives sum_k 2 k^2 E(k) = <omega^2>.  Shells are
integer bins k = 1..128 (Nyquist) by nearest-integer rounding of |k'|; the k=0
mode carries no energy (zero-mean vorticity) and is dropped.

Sampling.  SINGLE posterior samples for every generative method (pred_sample) --
an ensemble mean is a conditional expectation and annihilates the small scales
by construction, so a mean spectrum would answer a different question.
Senseiver is deterministic (pred_mean == pred_sample) and the classical methods
are deterministic.

Colour: house palette (pof_style) extended along the validated dataviz
categorical order -- slot 4 yellow / 5 magenta / 6 green for the three families
pof_style does not already name.  Series are drawn in that slot order, which is
the ordering the adjacent-pair CVD gate was validated on (worst adjacent CVD
dE 9.1, worst normal-vision dE 19.6, both clear of the 8 / 15 floors; see
check_palette.py).  Yellow, magenta and aqua sit under 3:1 on white, so the
contrast-relief rule applies: line style is a second, meaningful encoding
(solid = generative, dashed = learned deterministic, dotted = classical) and
panel (b) prints every method's name and numeric value -- the table view.

CPU only, no SLURM.  Run:  python -m figures.pof.fig_spectra_kolmogorov
[--compact]  (one-column version, spectra_kolmogorov_compact)
"""
from paths import REPO_ROOT
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figures.pof.pof_style import use_style, save, FULL_W, COL_W, C_OURS, C_CONV, C_SENS, C_TRUTH, INK_2, MUTED

# --- inputs ---------------------------------------------------------------
FD = (REPO_ROOT / "outputs/pof"
      / "field_dumps")
SIDE = 256                     # 256 x 256 vorticity on [0, 2pi)^2
KMAX = SIDE // 2               # 128 = Nyquist
LARGE = (1, 8)                 # large-scale / forcing band
SMALL = (32, KMAX)             # small-scale band

# extra categorical slots (dataviz reference palette, fixed order)
C_SIT = "#eda100"   # slot 4 yellow
C_IDW = "#e87ba4"   # slot 5 magenta
C_POD = "#008300"   # slot 6 green
C_GEO = "#4a3aa7"   # slot 7 violet
C_S3G = "#e34948"   # slot 8 red
# MLP-RBF would be a 9th series, past the 8 validated categorical slots, so it
# is NOT drawn as a line in panel (a); it appears in panel (b) only, in the
# de-emphasis gray, where its name and value are printed beside the bar.
C_MLP = "#898781"
BARS_ONLY = {"MLP-RBF"}

# label, short label (panel b ticks), npz stem, key, colour, linestyle.
# Order == validated palette slot order (see docstring) -- do not reshuffle.
METHODS = [
    ("DMF-Gen",         "DMF-Gen",   "dmfgen_cq_range", "pred_sample", C_OURS, "-"),
    ("latent-FM",       "latent-FM", "latent_fm", "pred_sample",        C_CONV, "-"),
    ("Senseiver",       "Senseiver", "senseiver", "pred_mean",          C_SENS, "--"),
    ("SiT-point",       "SiT-point", "sit",       "pred_sample",        C_SIT,  "-"),
    ("IDW $k$=8",       "IDW",       "classical", "pred_idw",           C_IDW,  ":"),
    ("gappy POD $r$160", "gappy POD", "classical", "pred_gappy_pod_r160", C_POD, ":"),
    # appended in validated slot order (7, 8); added once their dumps existed
    ("FNO",             "FNO",       "geofno",    "pred_mean",          C_GEO,  "--"),
    ("S3GM",            "S3GM",      "s3gm",      "pred_sample",        C_S3G,  "-"),
    ("MLP-RBF",         "MLP-RBF",   "mlprbf",    "pred_mean",          C_MLP,  "--"),
]


def shell_energy_spectrum(field_1d):
    """E(k) = sum_shell |omega_hat|^2 / (2 k^2), k = 1..KMAX, NO window.

    The domain is periodic, so the plain FFT is the exact spectral estimator
    (see module docstring); do not add a window here.
    """
    g = np.asarray(field_1d, dtype=np.float64).reshape(SIDE, SIDE)
    oh2 = np.abs(np.fft.fft2(g) / (SIDE * SIDE)) ** 2
    kk = np.fft.fftfreq(SIDE) * SIDE            # integer wavenumbers
    KX, KY = np.meshgrid(kk, kk, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(kmag > 0, oh2 / (2.0 * kmag ** 2), 0.0)
    shell = np.rint(kmag).astype(int)
    s = np.bincount(shell.ravel(), weights=e.ravel(), minlength=KMAX + 2)
    return s[1:KMAX + 1]


def band(spec, lo, hi):
    return float(spec[lo - 1:hi].sum())


# --- load + verify every panel shares one truth ---------------------------
dumps, truths = {}, {}
for stem in sorted({m[2] for m in METHODS}):
    d = np.load(FD / f"kolm_{stem}.npz", allow_pickle=True)
    dumps[stem] = d
    truths[stem] = d["truth"][:, 0].astype(np.float64)

ref_stem = "dmfgen_cq_range"
ref = truths[ref_stem]
bad = [s for s, t in truths.items()
       if t.shape != ref.shape or not np.allclose(t, ref, rtol=0, atol=0)]
if bad:                       # drop any panel whose truth disagrees
    print(f"[warn] truth mismatch, dropping dumps: {bad}")
    METHODS = [m for m in METHODS if m[2] not in bad]
print(f"[ok] truth identical across {sorted(truths)} "
      f"(max |diff| = {max(float(np.abs(t - ref).max()) for t in truths.values()):.3g})")

meta = json.loads(str(dumps[ref_stem]["meta"]))
print(f"[info] frame val {meta['snapshot_index']} (absolute {meta['absolute_frame']}), "
      f"{meta['n_sensors']} sensors, field {dumps[ref_stem]['names'][0]}")

# NFE is NOT matched across the generative samplers in these dumps (each ran at
# its own tuned budget).  A coarser ODE budget biases a sample smooth, so this
# is a real confound for the small-scale band and is stated on the figure.
NFE = {}
for _lab, _sh, stem, _key, _c, _ls in METHODS:
    m = json.loads(str(dumps[stem]["meta"]))
    if "nfe" in m and m.get("ode_solver", "none") != "none":
        NFE[stem] = int(m["nfe"])
print(f"[info] generative NFE per dump: {NFE}")
nfe_note = ", ".join(f"{lab.split(' ')[0]} NFE={NFE[stem]}"
                     for lab, _sh, stem, _k, _c, _ls in METHODS if stem in NFE)

# --- spectra --------------------------------------------------------------
E_truth = shell_energy_spectrum(ref)
spec, ratios = {}, {}
for lab, _sh, stem, key, _c, _ls in METHODS:
    spec[lab] = shell_energy_spectrum(dumps[stem][key][:, 0].astype(np.float64))
    ratios[lab] = (band(spec[lab], *LARGE) / band(E_truth, *LARGE),
                   band(spec[lab], *SMALL) / band(E_truth, *SMALL))
    print(f"[band] {lab:16s} key={key:20s} "
          f"E(k {LARGE[0]}-{LARGE[1]})/truth = {ratios[lab][0]:6.3f}   "
          f"E(k {SMALL[0]}-{SMALL[1]})/truth = {ratios[lab][1]:7.4f}")

k = np.arange(1, KMAX + 1)


def figure_full():
    """The paper's two-panel full-width figure (spectra_kolmogorov)."""
    use_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5})
    fig = plt.figure(figsize=(FULL_W, 3.3), constrained_layout=True)
    ga, gb = fig.subplots(1, 2, width_ratios=[1.22, 1.0])

    # (a) spectra ---------------------------------------------------------------
    ga.axvspan(*LARGE, color="#8a8880", alpha=0.07, lw=0)
    ga.axvspan(*SMALL, color="#8a8880", alpha=0.07, lw=0)
    ga.loglog(k, E_truth, color=C_TRUTH, lw=1.7, label="DNS truth", zorder=5)
    for lab, _sh, stem, key, col, ls in METHODS:
        if _sh in BARS_ONLY:
            continue
        # dotted/dashed strokes are thickened: magenta, yellow and aqua sit below
        # 3:1 on white, so the non-solid strokes need the extra weight to read.
        kw = {"dashes": (1.1, 1.3)} if ls == ":" else {"ls": ls}
        ga.loglog(k, spec[lab], color=col, label=lab,
                  lw=1.15 if ls == "-" else 1.35, **kw)

    # k^-3 is the classical 2D enstrophy-cascade slope; at this Re the DNS is
    # steeper than -3 (no extended enstrophy inertial range), so the guide is a
    # reference slope, not a fit.
    kg = np.array([7.0, 40.0])
    ga.loglog(kg, 3.0 * E_truth[6] * (kg / 7.0) ** -3.0, lw=0.7, color=MUTED,
              zorder=1)
    ga.text(23, 5.5 * E_truth[6] * (23 / 7.0) ** -3.0,
            r"$k^{-3}$ (truth is steeper)", fontsize=7.5, color=MUTED)
    ga.set_xlim(1, KMAX)
    ga.set_ylim(1e-13, 3e-1)
    ga.set_xlabel("wavenumber $k$")
    ga.set_ylabel(r"energy spectrum  $E(k)$")
    ga.set_title("(a)", loc="left", fontsize=10, fontweight="bold")
    ga.text(np.sqrt(LARGE[0] * LARGE[1]), 1.4e-1, "large scale", fontsize=7.5,
            color=INK_2, ha="center", va="top")
    ga.text(np.sqrt(SMALL[0] * SMALL[1]), 1.4e-1, "small scale", fontsize=7.5,
            color=INK_2, ha="center", va="top")
    ga.legend(loc="lower left", frameon=False, fontsize=7.5, labelspacing=0.28,
              borderpad=0.1, handlelength=1.9)
    ga.grid(True, which="major", lw=0.4, alpha=0.6)
    ga.set_axisbelow(True)

    # (b) band ratios -----------------------------------------------------------
    # Bars are anchored at the reference ratio 1 (truth), not at 0: on a log ratio
    # axis 1 is the natural origin, and both deficits (down) and spurious excess
    # (up) are errors.  Fill = large-scale band, hatch = small-scale band.
    x = np.arange(len(METHODS))
    w = 0.36
    for i, (lab, _sh, stem, key, col, ls) in enumerate(METHODS):
        rl, rs = ratios[lab]
        for off, r, hatch in ((-w / 2 - 0.015, rl, None),
                              (+w / 2 + 0.015, rs, "////")):
            lo, hi = min(1.0, r), max(1.0, r)
            gb.bar(x[i] + off, hi - lo, w, bottom=lo,
                   color=col if hatch is None else "none",
                   edgecolor=col, lw=0.7, hatch=hatch, zorder=3)
            va, dy = ("bottom", 1.25) if r >= 1.0 else ("top", 1 / 1.25)
            gb.text(x[i] + off, r * dy, f"{r:.2f}" if r >= 0.1 else f"{r:.3f}",
                    ha="center", va=va, fontsize=6.8, color=INK_2, zorder=4)

    gb.axhline(1.0, color=C_TRUTH, lw=0.8, zorder=2)
    gb.text(len(METHODS) - 0.42, 1.1, "truth", fontsize=7.5, color=C_TRUTH,
            ha="right", va="bottom")
    gb.set_yscale("log")
    gb.set_ylim(6e-3, 4.5)
    gb.set_xlim(-0.62, len(METHODS) - 0.38)
    gb.set_xticks(x)
    gb.set_xticklabels([m[1] for m in METHODS], fontsize=7.5, rotation=26,
                       ha="right", rotation_mode="anchor")
    gb.set_ylabel(r"band energy  $E/E_{\rm truth}$")
    gb.set_title("(b)", loc="left", fontsize=10, fontweight="bold")
    gb.grid(True, axis="y", lw=0.4, alpha=0.6)
    gb.set_axisbelow(True)
    gb.legend(handles=[Patch(facecolor=MUTED, edgecolor=MUTED,
                             label=f"large scale  $k$ {LARGE[0]}–{LARGE[1]}"),
                       Patch(facecolor="none", edgecolor=MUTED, hatch="////",
                             label=f"small scale  $k$ {SMALL[0]}–{SMALL[1]}")],
              loc="lower left", frameon=False, fontsize=7.5, labelspacing=0.3,
              handlelength=1.5, borderpad=0.1)

    save(fig, "spectra_kolmogorov")


def figure_compact():
    """One-column version for the shortened manuscript (spectra_kolmogorov_compact):
    (a) spectra over (b) small-scale band energy."""
    use_style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.8, "savefig.pad_inches": 0.02})
    fig = plt.figure(figsize=(COL_W, 3.7), constrained_layout=True)
    ga, gb = fig.subplots(2, 1, height_ratios=[1.45, 1.0])

    ga.axvspan(*LARGE, color="#8a8880", alpha=0.07, lw=0)
    ga.axvspan(*SMALL, color="#8a8880", alpha=0.07, lw=0)
    ga.loglog(k, E_truth, color=C_TRUTH, lw=1.6, label="DNS truth", zorder=5)
    for lab, sh, stem, key, col, ls in METHODS:
        if sh in BARS_ONLY:
            continue
        kw = {"dashes": (1.1, 1.3)} if ls == ":" else {"ls": ls}
        ga.loglog(k, spec[lab], color=col, label=sh.replace("latent-FM", "latent FM"),
                  lw=1.1 if ls == "-" else 1.3, **kw)
    kg = np.array([7.0, 40.0])
    ga.loglog(kg, 3.0 * E_truth[6] * (kg / 7.0) ** -3.0, lw=0.7, color=MUTED, zorder=1)
    ga.text(17, 5.0 * E_truth[6] * (17 / 7.0) ** -3.0, r"$k^{-3}$", fontsize=7, color=MUTED)
    ga.set_xlim(1, KMAX)
    ga.set_ylim(1e-13, 3e-1)
    ga.set_xlabel("wavenumber $k$", labelpad=1)
    ga.set_ylabel(r"$E(k)$")
    ga.set_title("(a)", loc="left", fontsize=8.5, fontweight="bold")
    ga.text(np.sqrt(LARGE[0] * LARGE[1]), 1.4e-1, "large", fontsize=6.8, color=INK_2,
            ha="center", va="top")
    ga.text(np.sqrt(SMALL[0] * SMALL[1]), 1.4e-1, "small", fontsize=6.8, color=INK_2,
            ha="center", va="top")
    ga.legend(loc="lower left", frameon=False, ncol=2, fontsize=6.8, labelspacing=0.25,
              columnspacing=0.9, borderpad=0.1, handlelength=1.8)
    ga.grid(True, which="major", lw=0.4, alpha=0.6)
    ga.set_axisbelow(True)

    # (b) small-scale band only, bars anchored at the truth ratio 1 on a log axis
    x = np.arange(len(METHODS))
    for i, (lab, sh, stem, key, col, ls) in enumerate(METHODS):
        r = ratios[lab][1]
        lo, hi = min(1.0, r), max(1.0, r)
        gb.bar(x[i], hi - lo, 0.62, bottom=lo, color=col, edgecolor=col, lw=0.6, zorder=3)
        va, dy = ("bottom", 1.22) if r >= 1.0 else ("top", 1 / 1.22)
        gb.text(x[i], r * dy, f"{r:.2f}" if r >= 0.1 else f"{r:.3f}", ha="center", va=va,
                fontsize=6.5, color=INK_2, zorder=4)
    gb.axhline(1.0, color=C_TRUTH, lw=0.8, zorder=2)
    gb.set_yscale("log")
    gb.set_ylim(2e-3, 8)
    gb.set_xlim(-0.6, len(METHODS) - 0.4)
    gb.set_xticks(x)
    gb.set_xticklabels([m[1].replace("latent-FM", "latent FM") for m in METHODS], fontsize=7,
                       rotation=30, ha="right",
                       rotation_mode="anchor")
    gb.set_ylabel(f"$E/E_{{\\rm truth}}$, $k$ {SMALL[0]}–{SMALL[1]}")
    gb.set_title("(b)", loc="left", fontsize=8.5, fontweight="bold")
    gb.grid(True, axis="y", lw=0.4, alpha=0.6)
    gb.set_axisbelow(True)
    save(fig, "spectra_kolmogorov_compact")


figure_compact() if "--compact" in sys.argv else figure_full()
