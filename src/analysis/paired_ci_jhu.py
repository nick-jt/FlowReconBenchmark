#!/usr/bin/env python3
"""3D (JHU) table numbers and paired per-snapshot CIs on the TEST half.

Every 3D row is scored on the 50-snapshot canonical draw of the held-out cube
(sensors seeded by seed * 777 + held-out index, checked by the snap-29
fingerprint gate) and split afterwards: odd held-out indices form the TUNE half,
even ones the TEST half (protocol). The candidate settings of a row
(checkpoint, NFE) are compared on TUNE by ensemble-mean relative L2 and the
chosen one is reported on TEST. Rows are paired on the 25 shared TEST snapshots:
percentile bootstrap (20k) + Wilcoxon, resolved only if both agree.

Sources are named explicitly; a setting may merge several files (e.g. a TUNE-only
run and a TEST rerun). A row whose chosen setting lacks the 25 TEST snapshots is
reported as missing, never filled from another setting.

Usage:  python -m analysis.paired_ci_jhu
Writes outputs/paper/tab_jhu_numbers.json and outputs/paper/paired_ci_jhu.json.
"""
from paths import REPO_ROOT
import argparse
import glob
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

J = REPO_ROOT / "outputs/runs/JHU"
RS = REPO_ROOT / "outputs/rescore_jhu"
OLD = J
CF = J / "baseline_confild"
N15 = J / "pointcloud_ffm/iclr_jhu_xcube_aug_DemoN15_20260818_083446/Evaluation"
SIT = OLD / "baseline_sit/matched/Baseline_sit_Stage1_DemoN41_20260828_181938"
CLS = RS / "classical/classical_baselines_oldraw_main_n19531.json"
CLS_BAND = RS / "classical/classical_baselines_oldraw_idw_band.json"
CH = ["Ux", "Uy", "Uz", "p"]
B = 20000
RNG = np.random.default_rng(0)

# row -> {setting: [sources]}; a source is a JSON path, a glob of per-frame files,
# or (path, method) for the classical anchors. *_demean files: the TUNE-selected setting rescored
# after spectral_utils removed each field's mean before the Hann window (same members).
ROWS = {
    "DMF-Gen": {
        "best nfe4": [N15 / "crps_nfe4_K8_all50.json"],
        "best nfe2": [RS / "dmfgen_N15_best_nfe2_K8_val50odd.json"],
        "best nfe8": [RS / "dmfgen_N15_best_nfe8_K8_val50odd.json"],
        **{f"last nfe{n}": [RS / f"dmfgen_N15_last_nfe{n}_K8_val50odd.json",
                            RS / f"dmfgen_N15_last_nfe{n}_K8_val50even_band{'_demean' if n == 2 else ''}.json"]
           for n in (2, 4, 8)},
    },
    # Latent FM and S3GM: last.pt only, because their best.pt was chosen on all 50 held-out frames
    "Latent FM": {"last": [RS / "latentfm_N24_last_K8_nfe4_val50.json",
                           RS / "latentfm_N24_last_K8_nfe4_val50_band_demean.json"]},
    "SiT-point": {c: [RS / f"sit_N41_{c}_rechunk_tune_K2/crps_snap*.json",
                      RS / f"sit_N41_{c}_rechunk_test_K8{'_demean' if c == 'last' else ''}/crps_snap*.json"]
                  for c in ("best", "last")},
    "FNO3D-FM": {f"{c} nfe{n}": [RS / f"fno3d_N90_{c}_nfe{n}_K8_val50.json",
                              RS / f"fno3d_N90_{c}_nfe{n}_K8_val50even_band"
                                   f"{'_demean' if (c, n) == ('best', 4) else ''}.json"]
              for c in ("best", "last") for n in (4, 16)},
    "FNO3D": {"best": [RS / "fno3d_det_N97_best_val50.json", RS / "fno3d_det_N97_best_val50_band_demean.json"],
              "last": [RS / "fno3d_det_N97_last_val50.json"]},
    # window-32 prior, sensors on the target row only; the published DPS scale (1.0) was set for 32
    # sensed rows, so the scale is re-chosen on TUNE
    "CoNFiLD (single snapshot)": {
        "best scale1.0": [CF / "improve/window1_P/Evaluation/crps_window1_snap*.json"],
        "best scale0.1": [RS / "confild_window1_best_scale0.1/Evaluation/crps_w1_snap*.json"]},
    "S3GM": {"last": [RS / "s3gm_norm_last_test_shard*.json"]},
    "Senseiver": {"budget": [RS / "senseiver_N43_budget_val50.json",
                             RS / "senseiver_N43_budget_val50_band_demean.json"]},
    "Set-DeepONet": {"budget": [RS / "deeponet_N60_budget_val50.json"]},   # wall-clock stop, no validation
    "Set-DeepONet++": {"best": [RS / "deeponetpp_p640_best_val50.json"]},
    "IDW": {"k8": [(CLS, "idw_n19531"), (CLS_BAND, "idw_n19531")]},
    "Nearest neighbor": {"k1": [(CLS, "kdtree_n19531")]},
    "Gappy POD": {"r80": [(CLS, "gappy_pod_n19531")]},
    "LSE": {"ridge": [(RS / "classical/classical_baselines_oldraw_lse_n19531.json", "lse_n19531")]},
    # scored on the central 120^3 crop (88% of the points), so never paired with the full-cube rows
    "Gen4Turb (120^3 crop)": {c: [RS / f"gen4turb_octa_{c}_crop120_val50.json"]
                              for c in ("model_3590", "model_3890", "model_4190")},
    # SM rows
    "Training mean": {"train": [(CLS, "constant_train_mean")]},
    "S3GM (retuned guidance)": {c: [OLD / "baseline_s3gm/matched/Baseline_s3gm_Stage1_DemoN94_20260828_224632/"
                                          f"eval_s3gm3d_jhu_tuned_{c}.json"] for c in ("best", "last")},
    "SiT-point (chunked sampler)": {c: [SIT / f"Evaluation_seeded_{c}/crps_snap*.json"] for c in ("best", "last")},
    "CoNFiLD (32-snapshot windows)": {
        c: [CF / f"unified_published_prior/Baseline_confild_Stage2_DemoN23_20260829_075611/"
                 f"Evaluation/crps_canonical_{c}_snap*.json"] for c in ("best", "last")},
    **{f"CoNFiLD ({name}, 32-snapshot windows)": {
        c: [CF / f"{arm}/Evaluation/crps_canonical_{c}_snap*.json"] for c in ("best", "last")}
       for name, arm in (("capacity-matched", "unified_cap1024/Baseline_confild_Stage2_DemoN23_20260829_075628"),
                         ("384-d latent", "unified_faithful384/Baseline_confild_Stage2_DemoN23_20260829_075618"))},
    # last.pt: both runs saved best.pt on all 50 held-out frames
    "DMF-Gen spectral (150 snapshots)": {"last nfe4": [RS / "dmfgen_spec_N29_last_nfe4_K8_test_band.json"]},
    "DMF-Gen spectral (1200 snapshots, 20 cubes)": {"last nfe4": [RS / "dmfgen_spec_N36_last_nfe4_K8_test_band_sharedstats.json"]},
}
PAIRED = ["DMF-Gen", "Latent FM", "SiT-point", "FNO3D-FM", "FNO3D",
          "CoNFiLD (single snapshot)", "S3GM", "Senseiver", "Set-DeepONet", "Set-DeepONet++", "IDW", "LSE"]


def records(src):
    """{held-out index: per-snapshot record} from one source."""
    if isinstance(src, tuple):
        path, method = src
        if not path.exists():
            return {}
        recs = json.load(open(path))["results"][method]["per_snapshot"]
        # classical records are flat and deterministic
        return {int(r["snapshot"]): {**r, "aggregate": {"rel_l2_mean": r["rel_l2_mean"],
                                                         "crps": r["crps"], "spread": 0.0}}
                for r in recs}
    out = {}
    for p in sorted(glob.glob(str(src))):
        d = json.load(open(p))
        recs = [d] if "per_field" in d and "snapshot" in d else (d.get("snapshots") or d["per_snapshot"])
        for r in recs:
            s = r["snapshot"] if "snapshot" in r else re.search(r"snap(\d+)\.json$", p).group(1)
            out[int(s)] = r
    return out


def summary(recs):
    agg = lambda k: float(np.mean([r["aggregate"][k] for r in recs]))
    out = {f: float(np.mean([r["per_field"][f]["rel_l2_mean"] for r in recs])) for f in CH}
    out.update(agg=agg("rel_l2_mean"), crps=agg("crps"), n=len(recs))
    if agg("spread") > 0:                       # deterministic rows have no ensemble
        out.update(spread_error=agg("spread_error_ratio"), cov90=agg("coverage_90"))
    if all("band" in r for r in recs):          # band energy / correlation: median over snapshots
        out["band"] = {b: {m: float(np.median([r["band"][b][m] for r in recs])) for m in v}
                       for b, v in recs[0]["band"].items()}
    return out


def assemble():
    table = {}
    for row, settings in ROWS.items():
        recs = {}
        for s, srcs in settings.items():
            r = {}
            for src in srcs:
                r.update(records(src))
            if r:
                recs[s] = r
        tune = {s: [r[i] for i in sorted(r) if i % 2] for s, r in recs.items()}
        tune_err = {s: summary(v)["agg"] for s, v in tune.items() if len(v) == 25}
        if tune_err:
            best = min(tune_err, key=tune_err.get)
        elif len(recs) == 1:                    # one fixed setting: nothing to select on TUNE
            best = next(iter(recs))
        else:
            table[row] = {"missing": f"no setting with the 25 TUNE snapshots ({sorted(recs)})"}
            continue
        test = [recs[best][i] for i in sorted(recs[best]) if i % 2 == 0]
        entry = {"setting": best, "tune_agg": tune_err}
        if len(test) == 25:
            entry["test"] = summary(test)
            entry["test_series"] = {i: recs[best][i]["aggregate"]["rel_l2_mean"]
                                    for i in sorted(recs[best]) if i % 2 == 0}
        else:
            entry["missing"] = f"setting {best!r} has {len(test)} of 25 TEST snapshots"
        table[row] = entry
    return table


def boot_ci(diff, alpha=0.05):
    idx = RNG.integers(0, diff.size, size=(B, diff.size))
    m = diff[idx].mean(axis=1)
    lo, hi = np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired(table):
    rows = [r for r in PAIRED if "test" in table.get(r, {})]
    order = sorted(rows, key=lambda r: table[r]["test"]["agg"])
    res = {"n_snapshots": 25, "ranking": order, "pairs": {}}
    for x, y in combinations(order, 2):
        sx, sy = table[x]["test_series"], table[y]["test_series"]
        d = np.array([sx[i] - sy[i] for i in sorted(sx)])
        lo, hi = boot_ci(d)
        pv = float(wilcoxon(d).pvalue)
        res["pairs"][f"{x} - {y}"] = {"mean_diff": float(d.mean()), "ci95": [lo, hi], "wilcoxon_p": pv,
                                      "resolved": bool(((lo > 0) or (hi < 0)) and pv < 0.05)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    table = assemble()
    for row, e in table.items():
        if "test" in e:
            t = e["test"]
            print(f"{row:<44} {e['setting']:<10} " + " ".join(f"{t[f]:.3f}" for f in CH)
                  + f" agg {t['agg']:.3f} crps {t['crps']:.3f}"
                  + (f" spr/err {t['spread_error']:.2f} cov90 {t['cov90']:.2f}" if "cov90" in t else ""))
        else:
            print(f"{row:<44} MISSING: {e['missing']}")
    res = paired(table)
    for k, v in res["pairs"].items():
        print(f"  {k:<50} {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
              f"p={v['wilcoxon_p']:.1e} {'resolved' if v['resolved'] else 'TIE'}")
    for e in table.values():
        e.pop("test_series", None)
    (REPO_ROOT / "outputs/paper/tab_jhu_numbers.json").write_text(json.dumps(table, indent=1))
    (REPO_ROOT / "outputs/paper/paired_ci_jhu.json").write_text(json.dumps(res, indent=1))
    print("wrote outputs/paper/tab_jhu_numbers.json, outputs/paper/paired_ci_jhu.json")


if __name__ == "__main__":
    main()
