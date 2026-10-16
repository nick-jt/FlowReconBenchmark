"""Route-1 post-hoc spread recalibration: one spread multiplier per generative row,
fitted on its TUNE-half payload and applied frozen to its TEST-half payload.

A multiplier alpha scales every member's deviation from the ensemble mean, so it
scales each frame's spread-error ratio by alpha and leaves the ensemble mean (and
its rel-L2) unchanged. The fit is

    alpha = target / mean over TUNE frames of the spread-error ratio,
    target = sqrt(K / (K + 1))          (0.943 at K = 8),

the expected ratio of a calibrated K-member ensemble whose spread uses ddof=1 and
whose error is that of the K-member mean. The reported number is the TEST mean of
the per-frame ratio times alpha. A row scored with stratify_blocks = B > 1 (the
cylinder: Re 80, then Re 250) also gets one multiplier per sub-block, and the TEST
ratio per sub-block under the single pooled multiplier.

Coverage after rescaling needs the per-point ensembles (dump_calib_points.py +
conformal_recalib.py); it is not derivable from these payloads.

    python -m calibration.recalibrate_spread \\
        --pair <run>/Evaluation/kolm_tune_last_dmfgen_K8_nfe2.json <run>/Evaluation/kolm_fleet_dmfgen_K8_nfe2.json \\
        [--pair TUNE TEST ...] --out outputs/paper/recalib/spread_multiplier_tune_test_2d.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Fields that must agree between the TUNE and the TEST payload of one row.
SAME = ("model", "run_dir", "ckpt", "nfe", "weights", "K", "n_obs", "cond_fields",
        "cond_source", "stratify_blocks", "n_frames")


def ratios(d: dict) -> np.ndarray:
    """Per-frame aggregate spread-error ratio, in frame-list order."""
    return np.array([s["aggregate"]["spread_error_ratio"] for s in d["snapshots"]], dtype=np.float64)


def fit_pair(tune_path: str, test_path: str) -> dict:
    tune, test = json.load(open(tune_path)), json.load(open(test_path))
    if tune.get("split") != "tune" or test.get("split") != "test":
        raise SystemExit(f"[recalib] need a split=tune then a split=test payload: {tune_path}, {test_path}")
    diff = {k: (tune.get(k), test.get(k)) for k in SAME if tune.get(k) != test.get(k)}
    if diff:
        raise SystemExit(f"[recalib] {Path(test_path).name}: TUNE and TEST settings differ on {diff}")
    K = int(test["K"])
    target = float(np.sqrt(K / (K + 1)))
    r_tu, r_te = ratios(tune), ratios(test)
    alpha = target / r_tu.mean()
    row = {"dataset": Path(test_path).parents[3].name, "model": test["model"],
           "run": Path(test["run_dir"]).name, "ckpt": test["ckpt"], "nfe": test["nfe"],
           "weights": test.get("weights"), "K": K, "target": target,
           "tune_file": str(tune_path), "test_file": str(test_path),
           "n_tune": int(r_tu.size), "n_test": int(r_te.size), "alpha": float(alpha),
           "spread_error_tune": float(r_tu.mean()), "spread_error_test_before": float(r_te.mean()),
           "spread_error_test_after": float(alpha * r_te.mean())}
    B = int(test.get("stratify_blocks") or 1)
    if B > 1:
        row["blocks"] = []
        for b, (tu, te) in enumerate(zip(np.split(r_tu, B), np.split(r_te, B))):
            a_b = target / tu.mean()
            row["blocks"].append({"block": b, "alpha": float(a_b), "spread_error_tune": float(tu.mean()),
                                  "spread_error_test_before": float(te.mean()),
                                  "spread_error_test_after": float(a_b * te.mean()),
                                  "spread_error_test_after_pooled_alpha": float(alpha * te.mean())})
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("TUNE", "TEST"),
                   help="TUNE-half and TEST-half fleet_2d payloads of one row (same settings).")
    p.add_argument("--out", default=None, help="output JSON path")
    args = p.parse_args()

    rows = [fit_pair(tu, te) for tu, te in args.pair]
    print(f"{'dataset':<14}{'model':<11}{'alpha':>7}{'s/e tune':>10}{'s/e test':>10}{'-> after':>10}  target")
    for r in rows:
        print(f"{r['dataset']:<14}{r['model']:<11}{r['alpha']:>7.3f}{r['spread_error_tune']:>10.3f}"
              f"{r['spread_error_test_before']:>10.3f}{r['spread_error_test_after']:>10.3f}  {r['target']:.4f}")
        for b in r.get("blocks", []):
            print(f"{'':<14}{'block ' + str(b['block']):<11}{b['alpha']:>7.3f}{b['spread_error_tune']:>10.3f}"
                  f"{b['spread_error_test_before']:>10.3f}{b['spread_error_test_after']:>10.3f}"
                  f"  pooled alpha -> {b['spread_error_test_after_pooled_alpha']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"method": "alpha = sqrt(K/(K+1)) / TUNE mean of the per-frame "
                                              "spread-error ratio; applied frozen to TEST", "rows": rows}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
