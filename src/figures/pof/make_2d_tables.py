#!/usr/bin/env python3
"""Emit the LaTeX row bodies of tab:kolm and tab:cyl straight from the canonical
eval JSONs, so the paper's 2D numbers are machine-derived and re-derivable.

Writes (Paper/pof2026/):
    tab_kolm_body.tex   rows for tab:kolm   (single-channel: vorticity)
    tab_cyl_body.tex    rows for tab:cyl    (per-channel Ux / Uy / p, p unobserved)
    tab_2d_numbers.json every number used, for the prose \\todo fills

Sources, all under Save_TrainedModel/<ds>/:
    <family>/<run>/Evaluation/{kolm,cyl}_fleet_<model>_K*.json   (n=50, K=8 / K=1)
    baseline_classical/classical_baselines_main_*.json           (all held-out frames)

Conventions kept from the eval driver: deterministic rows carry K=1 and NULL
dispersion (CRPS == MAE exactly), so their Cov.90 cell prints "---"; the best
value in each accuracy column is bolded; cost columns are the run's own
recorded GPU (all canonical 2D rows: H100).
"""
from paths import REPO_ROOT
import glob
import json
import re
from pathlib import Path
from statistics import median

ROOT = REPO_ROOT
STM = ROOT / "outputs/runs"
OUT = ROOT / "outputs" / "paper"

NICE = {"dmfgen": r"\dmfgen{}", "dmfgencq": r"\dmfgen{}", "sit": "SiT", "latentfm": "Latent FM", "senseiver": "Senseiver",
        "mlprbf": "MLP-RBF", "geofno": "Geo-FNO", "s3gm": "S3GM"}
ORDER = ["dmfgen", "dmfgencq", "sit", "latentfm", "s3gm", "senseiver", "mlprbf", "geofno"]


def fleet(ds):
    """model -> canonical payload, chosen by payload identity (fleet_select.py),
    never by filename -- a glob here once pulled occlusion runs into tab:kolm."""
    import sys
    from eval.fleet_select import load_canonical_fleet
    return load_canonical_fleet(ds, STM)


def classical(ds):
    rows = {}
    for f in sorted(glob.glob(str(STM / ds / "baseline_classical" / "classical_baselines_main_*.json"))):
        tag = Path(f).stem
        for k, v in json.load(open(f)).get("results", {}).items():
            if isinstance(v, dict) and "aggregate" in v:
                rows[f"{k}|{tag}"] = v
    return rows


def nice(m, d):
    """Row label with the sampler setting the payload records (chosen on the
    tuning half). SiT reads as the 3D row's name for the point-tokenized run."""
    base = NICE[m]
    if m == "sit" and d.get("_family") == "baseline_sit_point":
        base = "SiT-point"
    if m == "geofno" and "kolmogorov" in str(d.get("data_path", "")).lower():
        base = "FNO"  # the periodic grid FNO; Geo-FNO is the cylinder-mesh variant
    nfe = int(d["nfe"])
    if m in ("dmfgen", "dmfgencq", "latentfm", "sit", "s3gm"):
        return rf"{base} ($N{{=}}{nfe}$)"
    return base


def spread_error(a):
    """Spread-error cell; deterministic rows carry no ensemble."""
    v = a.get("spread_error_ratio")
    return "---" if v is None else f"{v:.2f}"


def by_re(label):
    """Re 80 and Re 250 aggregate cells from outputs/paper/cross_re_cylinder.json (cross_re_cylinder.py),
    whose row labels are the ones this script prints."""
    rows = {r["row"]: r for r in json.loads((OUT / "cross_re_cylinder.json").read_text())["rows"]}
    return f"{rows[label]['re80']:.3f} & {rows[label]['re250']:.3f}"


# Kolmogorov TEST band passes (fleet_2d --band-metrics, eval_kolm_band) at each row's canonical
# setting; the SiT-point pass is the per-step sampler run, per-frame identical to its TEST row.
KOLM_BAND = {"dmfgencq": "kolm_band_dmfgencq_K8_nfe2.json",
             "sit": "kolm_band_sit_K8_nfe32.json",
             "latentfm": "kolm_band_latentfm_K8_nfe4.json", "s3gm": "kolm_band_s3gm_K8_nfe1000.json",
             "senseiver": "kolm_band_senseiver_K1.json", "mlprbf": "kolm_band_mlprbf_K1.json",
             "geofno": "kolm_band_geofno_K1.json"}
KOLM_BAND_CLASSICAL = {"idw": r"IDW ($k{=}8$) (non-per.)", "gappy_pod_r160": r"Gappy POD ($r{=}160$)"}


def small_band(snaps, key=None):
    """Median over the TEST frames of the k = 32-128 band energy of single members
    relative to the truth."""
    return median((r[key] if key else r["band"])["small"]["energy"] for r in snaps)


def kolm_band(learned):
    """Row label -> small-band energy cell, checked against the canonical frames and setting."""
    out = {}
    for m, d in learned:
        band = json.loads((Path(d["run_dir"]) / "Evaluation" / KOLM_BAND[m]).read_text())
        if (band["frame_indices"], band["nfe"], band["weights"]) != (d["frame_indices"], d["nfe"], d["weights"]):
            raise SystemExit(f"[band] {KOLM_BAND[m]} does not match the canonical {m} row")
        out[nice(m, d)] = small_band(band["snapshots"])
    cls = json.loads((STM / "kolmogorov2d/baseline_classical/kolm_band_classical.json").read_text())
    for k, label in KOLM_BAND_CLASSICAL.items():
        out[label] = small_band(cls["snapshots"], k)
    return out


def b(x, best, n=3):
    """format, bolding the column optimum"""
    s = f"{x:.{n}f}"
    return rf"\textbf{{{s}}}" if abs(x - best) < 1e-12 else s


def kolm_table():
    ds = "kolmogorov2d"
    F, C = fleet(ds), classical(ds)
    learned = [(m, F[m]) for m in ORDER if m in F]
    l2 = [d["summary"]["aggregate"]["rel_l2_mean"] for _, d in learned]
    cr = [d["summary"]["aggregate"]["crps"] for _, d in learned]
    cls = []
    for key, v in C.items():
        name, tag = key.split("|")
        if name.startswith("constant_test"):
            continue
        per = "periodic" in tag and "nonperiodic" not in tag
        label = {"kdtree": "Nearest neighbor", "idw": r"IDW ($k{=}8$)",
                 "gappy_pod": rf"Gappy POD ($r{{=}}{int(v.get('rank', 0))}$)", "lse": "LSE",
                 "constant_train_mean": "Training mean"}.get(name.split("_n")[0], name)
        if name.startswith(("kdtree", "idw")):
            label += " (per.)" if per else " (non-per.)"
        elif per:
            continue                      # POD / mean are identical in both files
        cls.append((label, v["aggregate"]))
    all_l2 = l2 + [a["rel_l2_mean"] for _, a in cls]
    all_cr = cr + [a["crps"] for _, a in cls]
    bl2, bcr = min(all_l2), min(all_cr)
    # single-sample rel-L2 (what one posterior draw scores; the gallery number).
    # Deterministic and classical rows have one prediction, so sample == mean.
    sm = [d["summary"]["aggregate"].get("rel_l2_single",
                                        d["summary"]["aggregate"]["rel_l2_mean"])
          for _, d in learned]
    bsm = min(sm + [a["rel_l2_mean"] for _, a in cls])

    band = kolm_band(learned)
    ecell = lambda label: "---" if label not in band else f"{band[label]:.{3 if band[label] < 0.1 else 2}f}"
    lines = []
    for m, d in learned:
        a = d["summary"]["aggregate"]
        cov = a.get("coverage_90")
        smp = a.get("rel_l2_single", a["rel_l2_mean"])
        lines.append(f"{nice(m, d)} & {b(a['rel_l2_mean'], bl2)} & {b(smp, bsm)} & {b(a['crps'], bcr)} & "
                     f"{spread_error(a)} & {'---' if cov is None else f'{cov:.2f}'} & {ecell(nice(m, d))} & "
                     f"{d['inference_seconds_per_field_mean']:.2f} & "
                     f"{d['inference_peak_gpu_gb']:.2f} \\\\")
    lines.append(r"\hline")
    for label, a in sorted(cls, key=lambda r: r[1]["rel_l2_mean"]):
        lines.append(f"{label} & {b(a['rel_l2_mean'], bl2)} & {b(a['rel_l2_mean'], bsm)} & {b(a['crps'], bcr)} & --- & --- & "
                     f"{ecell(label)} & --- & --- \\\\")
    (OUT / "tab_kolm_body.tex").write_text(
        "% GENERATED by src/figs_pof/make_2d_tables.py -- do not hand-edit\n" + "\n".join(lines) + "\n")
    return {m: d["summary"]["aggregate"] for m, d in learned}, {l: a for l, a in cls}


def cyl_table():
    ds = "cylinder2d"
    F, C = fleet(ds), classical(ds)
    learned = [(m, F[m]) for m in ORDER if m in F]
    fields = learned[0][1]["field_names"]
    l2 = [d["summary"]["aggregate"]["rel_l2_mean"] for _, d in learned]
    cls = []
    for key, v in C.items():
        name, tag = key.split("|")
        if name.startswith("constant_test"):
            continue
        base = name.split("_n")[0]
        label = {"kdtree": "Nearest neighbor", "idw": r"IDW ($k{=}8$)", "lse": "LSE",
                 "constant_train_mean": "Training mean"}.get(base)
        if base == "gappy_pod":
            label = rf"Gappy POD ($r{{=}}{int(v.get('rank', 0))}$)"   # rank chosen on a training holdout
        if label is None or any(label == c[0] for c in cls):
            continue
        cls.append((label, v))
    bl2 = min(l2 + [v["aggregate"]["rel_l2_mean"] for _, v in cls])
    bp = min([d["summary"]["per_field"]["p"]["rel_l2_mean"] for _, d in learned]
             + [v["per_field"]["p"]["rel_l2_mean"] for _, v in cls])

    lines = []
    for m, d in learned:
        a, pf = d["summary"]["aggregate"], d["summary"]["per_field"]
        cov = a.get("coverage_90")
        lines.append(f"{nice(m, d)} & {b(a['rel_l2_mean'], bl2)} & {by_re(nice(m, d))} & "
                     + " & ".join(f"{pf[f]['rel_l2_mean']:.3f}" if f != "p"
                                  else b(pf[f]["rel_l2_mean"], bp) for f in fields)
                     + f" & {a['crps']:.3f} & {spread_error(a)} & {'---' if cov is None else f'{cov:.2f}'} & "
                       f"{d['inference_seconds_per_field_mean']:.2f} \\\\")
    lines.append(r"\hline")
    for label, v in sorted(cls, key=lambda r: r[1]["aggregate"]["rel_l2_mean"]):
        a, pf = v["aggregate"], v["per_field"]
        lines.append(f"{label} & {b(a['rel_l2_mean'], bl2)} & {by_re(label)} & "
                     + " & ".join(f"{pf[f]['rel_l2_mean']:.3f}" if f != "p"
                                  else b(pf[f]["rel_l2_mean"], bp) for f in fields)
                     + f" & {a['crps']:.3f} & --- & --- & --- \\\\")
    (OUT / "tab_cyl_body.tex").write_text(
        "% GENERATED by src/figs_pof/make_2d_tables.py -- do not hand-edit\n" + "\n".join(lines) + "\n")
    return ({m: (d["summary"]["aggregate"], d["summary"]["per_field"]) for m, d in learned},
            {l: v for l, v in cls})


if __name__ == "__main__":
    k_l, k_c = kolm_table()
    c_l, c_c = cyl_table()
    dump = {
        "kolmogorov2d": {"learned": {m: {"rel_l2": a["rel_l2_mean"], "crps": a["crps"],
                                         "cov90": a.get("coverage_90"),
                                         "spread_error": a.get("spread_error_ratio")}
                                     for m, a in k_l.items()},
                         "classical": {l: {"rel_l2": a["rel_l2_mean"], "crps": a["crps"]}
                                       for l, a in k_c.items()}},
        "cylinder2d": {"learned": {m: {"rel_l2": a["rel_l2_mean"], "crps": a["crps"],
                                       "cov90": a.get("coverage_90"),
                                       "spread_error": a.get("spread_error_ratio"),
                                       "per_field": {f: pf[f]["rel_l2_mean"] for f in pf}}
                                   for m, (a, pf) in c_l.items()},
                       "classical": {l: {"rel_l2": v["aggregate"]["rel_l2_mean"],
                                         "crps": v["aggregate"]["crps"],
                                         "per_field": {f: v["per_field"][f]["rel_l2_mean"]
                                                       for f in v["per_field"]}}
                                     for l, v in c_c.items()}},
    }
    (OUT / "tab_2d_numbers.json").write_text(json.dumps(dump, indent=1))
    for f in ("tab_kolm_body.tex", "tab_cyl_body.tex", "tab_2d_numbers.json"):
        print("wrote", OUT / f)
    print("\n--- tab_kolm_body.tex ---\n" + (OUT / "tab_kolm_body.tex").read_text())
    print("--- tab_cyl_body.tex ---\n" + (OUT / "tab_cyl_body.tex").read_text())
