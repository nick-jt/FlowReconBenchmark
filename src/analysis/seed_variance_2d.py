#!/usr/bin/env python3
"""Training-seed variance for the 2D fleets (P1-6).

The benchmark's rankings are read off single training runs. This asks the
question a reviewer will: is the gap between two methods larger than the spread
you get by retraining the same method with a different seed?

Reads the canonical fleet JSON (seed 42) and the replicate fleets (seeds 7 and
1337) written by src/submissions/fleet_2d/submit_2d_seed_replicates_delta.sh, which differ from the
canonical configs ONLY in `seed`, demo number and save root. For every method
it reports mean +/- sample std of aggregate rel-L2 across the available seeds,
then compares the seed spread against the adjacent method-to-method gap in the
canonical ranking, and flags any pair whose gap is smaller than the pooled seed
spread -- those pairs are not resolved by the benchmark at one run per method,
whatever the point estimates say.

Usage:  python seed_variance_2d.py [kolmogorov2d|cylinder2d|both] [--json OUT]
"""
from paths import REPO_ROOT

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = REPO_ROOT   # src/analysis/ -> src -> demo
STM = ROOT / "outputs/runs"
SEEDS = {42: "", 7: "_seed7", 1337: "_seed1337"}
ORDER = ["dmfgen", "dmfgen_cq", "sit", "latent_fm", "s3gm", "senseiver", "mlp_rbf", "geofno"]
NICE = {"dmfgen": "DMF-Gen", "dmfgen_cq": "DMF-Gen", "sit": "SiT", "latent_fm": "latent-FM", "s3gm": "S3GM",
        "senseiver": "Senseiver", "mlp_rbf": "MLP-RBF", "geofno": "Geo-FNO"}


_FLEETS = {}


def fleet_value(dataset, suffix, model):
    """aggregate rel-L2 for one (dataset, seed, model), or None.

    Chosen by payload identity via fleet_select.py (clean protocol stamp,
    canonical n_obs / K / frame count), not by filename. The previous
    glob-and-K-order rule was correct only because no operator or override runs
    shared these directories when it was written."""
    from eval.fleet_select import load_canonical_fleet
    key = f"{dataset}{suffix}"
    if key not in _FLEETS:
        try:
            _FLEETS[key] = load_canonical_fleet(key, STM)
        except KeyError:
            _FLEETS[key] = {}
    d = _FLEETS[key].get(model.replace("_", ""))
    if d is None:
        return None, None
    return float(d["summary"]["aggregate"]["rel_l2_mean"]), d.get("run_dir")


def main(dataset):
    print(f"\n########## {dataset}: training-seed variance")
    rows, spreads = {}, []
    for m in ORDER:
        vals = {}
        for seed, suf in SEEDS.items():
            v, p = fleet_value(dataset, suf, m)
            if v is not None:
                vals[seed] = v
        if not vals:
            continue
        arr = np.array(list(vals.values()))
        sd = float(arr.std(ddof=1)) if arr.size > 1 else None
        rows[m] = {"by_seed": vals, "mean": float(arr.mean()), "std": sd, "n_seeds": arr.size}
        if sd is not None:
            spreads.append(sd)
        seedstr = "  ".join(f"s{k}={v:.4f}" for k, v in sorted(vals.items()))
        print(f"  {NICE[m]:<11} n={arr.size}  mean={arr.mean():.4f}  "
              f"std={'n/a' if sd is None else f'{sd:.4f}'}   {seedstr}")

    done = [m for m in rows if rows[m]["n_seeds"] > 1]
    if not done:
        print("  (no method has more than one seed evaluated yet)")
        return {"dataset": dataset, "methods": rows, "status": "insufficient"}

    pooled = float(np.sqrt(np.mean(np.square([rows[m]["std"] for m in done]))))
    print(f"\n  pooled seed std over {len(done)} method(s): {pooled:.4f}")
    order = sorted(rows, key=lambda m: rows[m]["by_seed"].get(42, rows[m]["mean"]))
    print("  canonical ranking, adjacent gaps vs that spread:")
    unresolved = []
    for a, b in zip(order, order[1:]):
        ga = rows[a]["by_seed"].get(42, rows[a]["mean"])
        gb = rows[b]["by_seed"].get(42, rows[b]["mean"])
        gap = gb - ga
        bad = gap < pooled
        if bad:
            unresolved.append([a, b, gap])
        print(f"    {NICE[a]:<11} -> {NICE[b]:<11} gap={gap:.4f}"
              f"{'   <-- SMALLER THAN SEED SPREAD' if bad else ''}")
    return {"dataset": dataset, "methods": rows, "pooled_seed_std": pooled,
            "ranking": order, "gaps_below_seed_spread": unresolved}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="both")
    ap.add_argument("--json", default=str(ROOT / "outputs/paper/seed_variance_2d.json"))
    a = ap.parse_args()
    dss = ["kolmogorov2d", "cylinder2d"] if a.which == "both" else [a.which]
    out = {ds: main(ds) for ds in dss}
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.json}")
