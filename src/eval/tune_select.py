"""Summarise the TUNE-half selection sweeps of one 2D dataset (launcher case tune_2d).

    python -m eval.tune_select kolmogorov2d [--by crps]

Reads every fleet_2d JSON stamped `split: tune` under outputs/runs/<dataset> and prints,
per model and run family, the settings (NFE / steps, raw or EMA weights, checkpoint, run)
ordered by ensemble-mean relative L2 (or by CRPS with --by crps), with CRPS, spread-error
ratio and 90% coverage alongside. The first line of each block is the choice under that
criterion; the frozen protocol carries the rel-L2 choice to the TEST scoring. Nothing here
is a reported number.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from paths import RUNS_DIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default="kolmogorov2d")
    p.add_argument("--by", choices=["rel_l2", "crps"], default="rel_l2",
                   help="Selection criterion (lower is better).")
    args = p.parse_args()
    ds = args.dataset
    rows = []
    for path in sorted(glob.glob(str(RUNS_DIR / ds / "*" / "*" / "Evaluation" / "*_tune*.json"))):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("split") != "tune" or "summary" not in d:
            continue
        a = d["summary"]["aggregate"]
        rows.append({"model": str(d["model"]), "family": Path(path).parents[2].name,
                     "run": Path(path).parents[1].name, "file": Path(path).name,
                     "nfe": int(d.get("nfe", 0)), "weights": str(d.get("weights", "")),
                     "ckpt": str(d.get("ckpt", "")), "epoch": int(d.get("ckpt_epoch", -1)),
                     "rel_l2": float(a["rel_l2_mean"]), "crps": float(a["crps"]),
                     "spread_error": a.get("spread_error_ratio"), "cov90": a.get("coverage_90")})
    if not rows:
        raise SystemExit(f"[tune_select] no split=tune JSONs under {RUNS_DIR / ds}")
    label = {"rel_l2": "rel-L2 of the ensemble mean", "crps": "CRPS"}[args.by]
    print(f"TUNE-half sweeps, {ds} (ordered by {label}; first line = choice)")

    def fmt(v):
        return "   --" if v is None else f"{v:.2f}"

    for model, family in sorted({(r["model"], r["family"]) for r in rows}):
        print(f"\n{model}  ({family})")
        for r in sorted((r for r in rows if (r["model"], r["family"]) == (model, family)), key=lambda r: r[args.by]):
            print(f"  relL2={r['rel_l2']:.4f} crps={r['crps']:.4f} spr/err={fmt(r['spread_error'])} "
                  f"cov90={fmt(r['cov90'])}  nfe={r['nfe']:<4d} weights={r['weights']:<12s} "
                  f"ckpt={r['ckpt']:<5s} epoch={r['epoch']:<5d} {r['run']}  [{r['file']}]")


if __name__ == "__main__":
    main()
