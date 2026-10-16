"""PoF figure: windowed energy spectra, band energy and band correlation of single
samples on the held-out 3D cutout (Fig. 8).

The JHU cutouts are non-periodic, so every spectrum uses the Hann-windowed shell
estimator of methods.dmfgen.spectral_utils; an unwindowed FFT has a leakage floor that
invalidates dissipation-band ratios. Single samples only (ensemble means destroy
small scales by construction).

One held-out test-half frame (absolute frame 152, the gallery frame), one sensor
draw (seed 102, 1% of U_x and U_z), one sample per generative method, with the
settings of the 3D table: DMF-Gen (DemoN15, last.pt) at its tuned NFE 2, SiT-point
(last, per-step re-chunk sampler) at N = 32, latent FM (Stage2 DemoN24) and FNO3D
(DemoN90 xcube) at NFE 4, Senseiver (budget.pt), IDW. All fields come from
outputs/pof/field_dumps/frame152/jhu_*.npz (physical units) and are checked below
to share the same truth.

(a) total-velocity spectra; (b) band energy relative to DNS and (c) band-limited
correlation with DNS, both pooled over the observed channels U_x and U_z.
--compact: one-column 2 x 2 version (spectra_stats_compact); no JSON is written.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paths import REPO_ROOT
from figures.pof.pof_style import use_style, save, OUT, FULL_W, COL_W, C_OURS, C_CONV, C_SENS, C_TRUTH, INK_2, MUTED
from methods.dmfgen.spectral_utils import shell_spectrum, reliable_kmax, band_ratio, band_corr

N = 125
FRAME = 152
INERTIAL = (8, 31)
DISSIP = (32, 50)
OBS = (0, 2)                        # U_x, U_z observed; U_y (1) never observed
FD = REPO_ROOT / "outputs/pof/field_dumps/frame152"
CACHE = OUT / "spectra_windowed_cache.npz"
VERSION = "frame152_v3"   # v3: field mean removed before the Hann window

C_SIT = "#eda100"                   # slot 4 yellow, as SiT-point in fig_spectra_kolmogorov.py
C_FNO = "#4a3aa7"                   # slot 7 violet, as FNO there
C_IDW = "#e87ba4"                   # slot 5 magenta, as IDW there
# label, colour, line style. Order is the validated drawing order (all adjacent
# pairs clear the CVD floor); line style is the secondary encoding:
# solid = generative, dashed = learned deterministic, dotted = classical.
METHODS = [("DMF-Gen", C_OURS, "-"), ("latent FM", C_CONV, "-"), ("Senseiver", C_SENS, "--"),
           ("SiT-point", C_SIT, "-"), ("FNO3D-FM", C_FNO, "-"), ("IDW", C_IDW, ":")]


def load_fields():
    """{label: (N^3, 3) velocity of one sample}, plus the shared truth."""
    d = {k: np.load(FD / f"jhu_{k}.npz") for k in ("sit", "dmfgen", "latent_fm", "fno3d",
                                                   "senseiver", "classical")}
    assert json.loads(str(d["sit"]["meta"]))["absolute_frame"] == FRAME, f"reference frame != {FRAME}"
    truth = d["sit"]["truth"][:, :3]
    for k, z in d.items():
        assert np.allclose(z["truth"][:, :3], truth), f"truth mismatch: {k}"
    return truth, {"DMF-Gen": d["dmfgen"]["pred_sample"][:, :3],
                   "latent FM": d["latent_fm"]["pred_sample"][:, :3],
                   "Senseiver": d["senseiver"]["pred_mean"][:, :3],
                   "SiT-point": d["sit"]["pred_sample"][:, :3],
                   "FNO3D-FM": d["fno3d"]["pred_sample"][:, :3],
                   "IDW": d["classical"]["pred_idw"][:, :3]}


def compute():
    truth, fields = load_fields()
    spec3 = lambda a: np.stack([shell_spectrum(a[:, c].reshape(N, N, N), window=True) for c in range(3)])
    cubes = lambda a: [a[:, c].reshape(N, N, N) for c in OBS]
    out = {"truth": spec3(truth)}
    for lab, f in fields.items():
        out[lab] = spec3(f)
        corr = band_corr(cubes(f), cubes(truth), {"inertial": INERTIAL, "dissip": DISSIP})
        out[f"corr|{lab}"] = np.array([corr["inertial"], corr["dissip"]])
    return out


z = np.load(CACHE) if CACHE.exists() else None
if z is None or str(z.get("version", "")) != VERSION:
    res = compute()
    np.savez_compressed(CACHE, version=VERSION, **{k.replace("|", "__"): v for k, v in res.items()})
    z = np.load(CACHE)
spec = {lab: z[lab] for lab in ["truth"] + [m[0] for m in METHODS]}
corr = {lab: dict(zip(("inertial", "dissip"), z[f"corr__{lab}"])) for lab, _, _ in METHODS}

tot = {k: v.sum(0) for k, v in spec.items()}
obs = {k: v[list(OBS)].sum(0) for k, v in spec.items()}
kmax = reliable_kmax(tot["truth"])
k = np.arange(1, kmax + 1)
ratio = {lab: {"inertial": band_ratio(obs[lab], obs["truth"], *INERTIAL),
               "dissip": band_ratio(obs[lab], obs["truth"], *DISSIP)} for lab, _, _ in METHODS}

def figure_full():
    """The paper's three-panel full-width figure and spectra_band_ratios.json."""
    use_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5})
    fig = plt.figure(figsize=(FULL_W, 3.1), constrained_layout=True)
    ga, gb, gc = fig.subplots(1, 3, width_ratios=[1.45, 1, 1])

    # --- (a) total-velocity energy spectrum -------------------------------------
    ga.axvspan(*INERTIAL, color="#8a8880", alpha=0.07, lw=0)
    ga.axvspan(*DISSIP, color="#8a8880", alpha=0.13, lw=0)
    ga.loglog(k, tot["truth"][:kmax], color=C_TRUTH, lw=1.6, label="DNS truth", zorder=5)
    for lab, col, ls in METHODS:
        kw = {"dashes": (1.1, 1.3)} if ls == ":" else {"ls": ls}
        ga.loglog(k, tot[lab][:kmax], color=col, lw=1.15 if ls == "-" else 1.35, label=lab, **kw)
    kk = np.array([8.0, 31.0])
    ga.loglog(kk, 2.5 * tot["truth"][7] * (kk / 8.0) ** (-5 / 3), lw=0.7, color=MUTED)
    ga.text(16, 1.3 * 2.5 * tot["truth"][7] * 2.0 ** (-5 / 3), r"$k^{-5/3}$", fontsize=7.5, color=MUTED,
            va="bottom")
    ga.text(np.sqrt(8 * 31), 2.2e12, "inertial\n$k$ 8–31", fontsize=7.5, color=INK_2, ha="center", va="top")
    ga.text(np.sqrt(32 * 50), 2.2e12, "dissipation\n$k$ 32–50", fontsize=7.5, color=INK_2, ha="center", va="top")
    ga.set_xlim(1, kmax)
    ga.set_ylim(1e3, 4e12)
    ga.set_xlabel("wavenumber $k$")
    ga.set_ylabel("$E(k)$")
    ga.set_title("(a)", loc="left", fontsize=10, fontweight="bold")
    ga.legend(loc="lower left", frameon=False, fontsize=7.5, labelspacing=0.28, handlelength=1.9)
    ga.grid(True, which="major", lw=0.4, alpha=0.6)

    # --- (b) band energy and (c) band correlation, observed channels -----------------
    y = np.arange(len(METHODS))[::-1]
    MARK = {"inertial": ("o", "inertial $k$ 8–31"), "dissip": ("^", "dissipation $k$ 32–50")}
    for ax, vals, label in [(gb, ratio, "(b)"), (gc, corr, "(c)")]:
        for yi, (lab, col, _) in zip(y, METHODS):
            for band, (mk, _) in MARK.items():
                v = vals[lab][band]
                ax.plot(v, yi, mk, ms=6.5, mfc=col if band == "inertial" else "white", mec=col, mew=1.3, zorder=3)
                up = band == "inertial"           # inertial value above its marker, dissipation below
                ax.text(v, yi + (0.2 if up else -0.2), f"{v:.2f}", fontsize=6.8, color=INK_2, ha="center",
                        va="bottom" if up else "top")
        ax.set_yticks(y)
        ax.set_yticklabels([m[0] for m in METHODS])
        ax.set_ylim(-0.7, len(METHODS) + 0.45)     # headroom above the top row for the marker key
        ax.grid(True, axis="x", lw=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
    gb.set_xscale("log")
    gb.set_xlim(3e-3, 25)
    gb.axvline(1.0, color=MUTED, lw=0.8, zorder=1)
    gb.set_xlabel(r"band energy  $E/E_{\rm DNS}$")
    gc.set_xlim(-0.1, 1.0)
    gc.axvline(0.0, color=MUTED, lw=0.8, zorder=1)
    gc.set_xlabel("band correlation with DNS")
    gc.set_yticklabels([])
    gb.legend(handles=[Line2D([], [], marker=mk, ls="", mfc=INK_2 if b == "inertial" else "white", mec=INK_2, mew=1.2, ms=6, label=t)
                       for b, (mk, t) in MARK.items()], loc="upper left", frameon=False, fontsize=7, handletextpad=0.3)
    save(fig, "spectra_stats")

    per_channel = {lab: {b: [band_ratio(spec[lab][c], spec["truth"][c], *rng) for c in range(3)]
                         for b, rng in (("inertial", INERTIAL), ("dissip", DISSIP))} for lab, _, _ in METHODS}
    summary = {"frame": FRAME, "kmax_reliable": kmax, "bands": {"inertial": list(INERTIAL), "dissip": list(DISSIP)},
               "channels": ["Ux", "Uy", "Uz"], "observed": [True, False, True],
               "per_channel_energy_ratio": per_channel,
               "observed_pooled_energy_ratio": ratio,
               "observed_pooled_band_correlation": corr,
               "total_energy_ratio": {lab: {b: band_ratio(tot[lab], tot["truth"], *rng)
                                            for b, rng in (("inertial", INERTIAL), ("dissip", DISSIP))}
                                      for lab, _, _ in METHODS}}
    json.dump(summary, open(OUT / "spectra_band_ratios.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


def figure_compact():
    """One-column 2 x 2 version for the shortened manuscript (spectra_stats_compact):
    (a) spectra across the top, (b) band energy and (c) band correlation below.
    Reads the same cache and writes no JSON."""
    use_style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.8, "savefig.pad_inches": 0.02})
    fig = plt.figure(figsize=(COL_W, 3.75), constrained_layout=True)
    gsp = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ga = fig.add_subplot(gsp[0, :])
    gb = fig.add_subplot(gsp[1, 0])
    gc = fig.add_subplot(gsp[1, 1], sharey=gb)

    ga.axvspan(*INERTIAL, color="#8a8880", alpha=0.07, lw=0)
    ga.axvspan(*DISSIP, color="#8a8880", alpha=0.13, lw=0)
    ga.loglog(k, tot["truth"][:kmax], color=C_TRUTH, lw=1.5, label="DNS truth", zorder=5)
    for lab, col, ls in METHODS:
        kw = {"dashes": (1.1, 1.3)} if ls == ":" else {"ls": ls}
        ga.loglog(k, tot[lab][:kmax], color=col, lw=1.1 if ls == "-" else 1.3, label=lab, **kw)
    kk = np.array([8.0, 31.0])
    ga.loglog(kk, 2.5 * tot["truth"][7] * (kk / 8.0) ** (-5 / 3), lw=0.7, color=MUTED)
    ga.text(16, 1.3 * 2.5 * tot["truth"][7] * 2.0 ** (-5 / 3), r"$k^{-5/3}$", fontsize=7,
            color=MUTED, va="bottom")
    ga.text(np.sqrt(8 * 31), 2.2e12, "inertial", fontsize=6.8, color=INK_2, ha="center", va="top")
    ga.text(np.sqrt(32 * 50), 2.2e12, "dissip.", fontsize=6.8, color=INK_2, ha="center", va="top")
    ga.set_xlim(1, kmax)
    ga.set_ylim(1e3, 4e12)
    ga.set_xlabel("wavenumber $k$", labelpad=1)
    ga.set_ylabel("$E(k)$")
    ga.set_title("(a)", loc="left", fontsize=8.5, fontweight="bold")
    ga.legend(loc="lower left", frameon=False, ncol=2, fontsize=6.8, labelspacing=0.25,
              columnspacing=0.9, handlelength=1.8)
    ga.grid(True, which="major", lw=0.4, alpha=0.6)

    y = np.arange(len(METHODS))[::-1]
    MARK = {"inertial": ("o", "inertial"), "dissip": ("^", "dissipation")}
    for ax, vals, label in [(gb, ratio, "(b)"), (gc, corr, "(c)")]:
        for yi, (lab, col, _) in zip(y, METHODS):
            for band, (mk, _) in MARK.items():
                v = vals[lab][band]
                ax.plot(v, yi, mk, ms=5, mfc=col if band == "inertial" else "white", mec=col,
                        mew=1.1, zorder=3)
            # one column has no room for every value: (c) keeps the inertial-band
            # correlations (all dissipation-band values are |r| <= 0.03); (b) none
            if ax is gc:
                v = vals[lab]["inertial"]
                ax.text(v + 0.06, yi, f"{v:.2f}", fontsize=6.5, color=INK_2, ha="left",
                        va="center")
        ax.set_ylim(-0.75, len(METHODS) + 0.15)
        ax.grid(True, axis="x", lw=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(label, loc="left", fontsize=8.5, fontweight="bold")
    gb.set_yticks(y)
    gb.set_yticklabels([m[0] for m in METHODS])
    gb.set_xscale("log")
    gb.set_xlim(3e-3, 25)
    gb.axvline(1.0, color=MUTED, lw=0.8, zorder=1)
    gb.set_xlabel(r"$E/E_{\rm DNS}$", labelpad=1)
    gc.set_xlim(-0.12, 1.0)
    gc.axvline(0.0, color=MUTED, lw=0.8, zorder=1)
    gc.set_xlabel("correlation with DNS", labelpad=1)
    gc.tick_params(labelleft=False)
    fig.legend(handles=[Line2D([], [], marker=mk, ls="", mfc=INK_2 if b == "inertial" else "white",
                               mec=INK_2, mew=1.0, ms=5, label=t) for b, (mk, t) in MARK.items()],
               loc="lower center", ncol=2, frameon=False, fontsize=6.8, handletextpad=0.3,
               bbox_to_anchor=(0.5, -0.04))
    save(fig, "spectra_stats_compact")


figure_compact() if "--compact" in sys.argv else figure_full()
