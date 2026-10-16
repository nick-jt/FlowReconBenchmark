#!/usr/bin/env python3
"""Paired per-snapshot confidence intervals for the 2D fleets.

Every method in a fleet is scored on the SAME 50 held-out frames with the SAME
canonical seeded sensor draws, so method-to-method comparisons are paired and
the per-snapshot difference is the right unit of analysis. This computes, for
each ordered pair, the mean paired difference in relative L2 and in CRPS with a
BCa-free percentile bootstrap CI (20k resamples of the 50 differences) plus the
two-sided Wilcoxon signed-rank p-value, and reports which comparisons are
statistically resolved at the benchmark's n.

Also answers the question a leaderboard needs: is the top method's lead larger
than the paired noise, or is the top group a tie?

Usage:  python paired_ci_2d.py [kolmogorov2d|cylinder2d|both] [--out FILE.json]
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

ROOT = REPO_ROOT   # src/analysis/ -> src -> demo
STM = ROOT / "outputs/runs"
NICE = {"dmfgen": "DMF-Gen", "sit": "SiT", "latent_fm": "latent-FM",
        "senseiver": "Senseiver", "mlp_rbf": "MLP-RBF", "geofno": "Geo-FNO", "s3gm": "S3GM"}
B = 20000
RNG = np.random.default_rng(0)


def fleet_snapshots(ds):
    """model -> {snapshot: (rel_l2, crps)} for the CANONICAL row only, chosen by
    payload identity (fleet_select.py) rather than filename sort order."""
    from eval.fleet_select import load_canonical_fleet
    out = {}
    for m, d in load_canonical_fleet(ds, STM).items():
        out[m] = {int(s["snapshot"]): (float(s["aggregate"]["rel_l2_mean"]),
                                        float(s["aggregate"]["crps"]))
                  for s in d["snapshots"]}
    return out


CLS_LABEL = {                # classical result key (before "_n<N>") -> display label
    "kdtree": "Nearest neighbor",
    "idw": "IDW (k=8)",
    "gappy_pod": "Gappy POD",    # rank suffix " (r=<rank>)" from the payload
    "lse": "LSE",
    "constant_train_mean": "Training mean",
}
# Reconstructed 2026-09-10: this constant was lost when fleet_snapshots() was
# spliced out by span -- the file was untracked, so git could not restore it.
# Only labels are affected; the numbers come from the JSONs.


def classical_snapshots(ds):
    """label -> {snapshot: (rel_l2, crps)}. The classical runs score the same
    50 frames under the same seeded sensor draws as the fleet and key each
    record by the split index (protocol.snapshot_key), so the classical-vs-
    learned comparison is paired on identical frames."""
    out = {}
    for f in sorted(glob.glob(str(STM / ds / "baseline_classical" / "classical_baselines_main_*.json"))):
        tag = Path(f).stem
        if "periodic" in tag and "nonperiodic" not in tag:
            continue                     # keep the non-periodic convention for Kolmogorov
        d = json.load(open(f))
        if d.get("protocol", {}).get("snapshot_key") != "split index":
            raise SystemExit(f"[paired_ci_2d] {f}: per-snapshot records are not keyed by split "
                             "index; rerun the classical anchors")
        for k, v in d.get("results", {}).items():
            base = k.split("_n")[0]
            if base not in CLS_LABEL or "per_snapshot" not in v:
                continue
            lbl = CLS_LABEL[base]
            if base == "gappy_pod":
                lbl += f" (r={int(v['rank'])})"
            out.setdefault(lbl, {s["snapshot"]: (s["rel_l2_mean"], s["crps"])
                                 for s in v["per_snapshot"]})
    return out


def boot_ci(diff, alpha=0.05):
    idx = RNG.integers(0, diff.size, size=(B, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def analyse(ds):
    F = fleet_snapshots(ds)
    F.update(classical_snapshots(ds))     # paired on the same 50 frames
    if len(F) < 2:
        print(f"[{ds}] fewer than two models with per-snapshot data"); return {}
    common = sorted(set.intersection(*(set(v) for v in F.values())))
    print(f"\n########## {ds}: {len(F)} methods on {len(common)} shared frames ##########")
    order = sorted(F, key=lambda m: np.mean([F[m][s][0] for s in common]))
    print("ranking by mean relative L2:")
    for m in order:
        r = np.array([F[m][s][0] for s in common]); c = np.array([F[m][s][1] for s in common])
        print(f"  {NICE.get(m, m):<11} relL2 {r.mean():.4f} +/- {r.std(ddof=1)/np.sqrt(r.size):.4f} (sem)"
              f"   CRPS {c.mean():.4f}")
    res = {"dataset": ds, "n_frames": len(common), "ranking": order, "pairs": {}}
    print("\npaired differences (row - col), 95% percentile-bootstrap CI on the mean:")
    for a, bm in combinations(order, 2):
        for k, lbl in ((0, "rel_l2"), (1, "crps")):
            d = np.array([F[a][s][k] - F[bm][s][k] for s in common])
            lo, hi = boot_ci(d)
            try:
                pv = float(wilcoxon(d).pvalue)
            except ValueError:
                pv = float("nan")
            # CONSERVATIVE: a difference counts as resolved only if the
            # bootstrap CI excludes zero AND the signed-rank test agrees.
            # (DMF-Gen vs SiT on Kolmogorov is exactly why: the CI barely
            # excludes 0 while Wilcoxon gives p=0.16, i.e. a few frames drive
            # the mean and the ranks are balanced -- that is a tie.)
            resolved = ((lo > 0) or (hi < 0)) and (pv == pv) and (pv < 0.05)
            res["pairs"][f"{a}-{bm}|{lbl}"] = {"mean_diff": float(d.mean()), "ci95": [lo, hi],
                                               "wilcoxon_p": pv, "resolved": bool(resolved)}
            if lbl == "rel_l2":
                mark = "resolved" if resolved else "TIE (CI spans 0)"
                print(f"  {NICE.get(a,a):<10} - {NICE.get(bm,bm):<10} "
                      f"{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={pv:.1e}  {mark}")
    # leader tie-group: everyone whose gap to the leader is unresolved
    lead = order[0]
    tie = [lead] + [m for m in order[1:]
                    if not res["pairs"].get(f"{lead}-{m}|rel_l2", {}).get("resolved", True)]
    res["leader"] = lead
    res["tie_group_rel_l2"] = tie
    print(f"\n  leader: {NICE.get(lead, lead)};  statistically tied with it: "
          f"{[NICE.get(m, m) for m in tie[1:]] or 'none (lead is resolved)'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="both")
    ap.add_argument("--out", default=str(ROOT / "outputs/paper/paired_ci_2d.json"))
    a = ap.parse_args()
    dss = ["kolmogorov2d", "cylinder2d"] if a.which == "both" else [a.which]
    allres = {ds: analyse(ds) for ds in dss}
    Path(a.out).write_text(json.dumps(allres, indent=1))
    print(f"\nwrote {a.out}")
