"""Kolmogorov S3GM rows rescored with the upstream sampler (fleet_2d --s3gm-sampler upstream).

TUNE: one block per alpha_case from the per-snapshot records (a cancelled arm still counts
its frames). A member is divergent if it is not finite or its max |vorticity| exceeds
DIVERGE x the frame's true max (stable members stay below 1.4x; divergent ones reach 1e6-1e8).
TEST: every S3GM payload of the upstream rescore next to the pc_dps payload it replaces
(Evaluation/superseded_pcdps/), for the main row, band pass, seeds, densities and operators.

    python -I src/analysis/diagnostics/s3gm_rescore_2d.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "outputs/runs"
OUT = REPO / "outputs/rescore_2d/s3gm_upstream/summary.json"
DIVERGE = 2.0
KEYS = ("rel_l2_mean", "rel_l2_single", "crps", "spread_error_ratio", "coverage_90")


def evaluation(ds):
    return next((RUNS / ds / "baseline_s3gm").glob("Baseline_s3gm_Stage1_DemoN*/Evaluation"))


def tune_rows():
    rows = {}
    for p in sorted(evaluation("kolmogorov2d").glob("crps_tune_best_nfe1000_ema_if_kept_s3gmupstream_a*_n655_snap*.json")):
        d = json.loads(p.read_text())
        rows.setdefault(d["s3gm_sampler"], []).append(d)
    out = {}
    for name, recs in sorted(rows.items()):
        ratio = np.array([np.max(r["member_max_abs"]) / r["truth_max_abs"] for r in recs])
        div = [r["snapshot"] for r in recs
               if not np.all(np.isfinite(r["member_max_abs"]))
               or np.max(r["member_max_abs"]) > DIVERGE * r["truth_max_abs"]]
        n_div = sum(int(np.sum(~np.isfinite(r["member_max_abs"])
                               | (np.array(r["member_max_abs"]) > DIVERGE * r["truth_max_abs"]))) for r in recs)
        ok = [r for r in recs if r["snapshot"] not in div]
        out[name] = {"n_frames": len(recs), "divergent_frames": div, "divergent_members": n_div,
                     "max_member_over_truth_max_stable_frames": float(max(
                         np.max(r["member_max_abs"]) / r["truth_max_abs"] for r in ok)) if ok else None,
                     **{k: float(np.mean([r["aggregate"][k] for r in recs])) for k in KEYS}}
        if div:
            out[name]["note"] = "means include the divergent frames"
    return out


def summary(d):
    a = d["summary"]["aggregate"]
    row = {k: a[k] for k in KEYS}
    row["seconds_per_member"] = d.get("inference_seconds_per_field_mean")
    snaps = [s for s in d["snapshots"] if "member_max_abs" in s]
    if snaps:
        bad = {s["snapshot"]: int(np.sum(~np.isfinite(s["member_max_abs"])
                                         | (np.array(s["member_max_abs"]) > DIVERGE * s["truth_max_abs"])))
               for s in snaps}
        row["members_over_threshold_by_frame"] = {k: v for k, v in bad.items() if v}
        row["frames_rel_l2_mean_above_1"] = [s["snapshot"] for s in snaps if s["aggregate"]["rel_l2_mean"] > 1]
        row["rel_l2_mean_median_over_frames"] = float(np.median([s["aggregate"]["rel_l2_mean"] for s in snaps]))
    if "band_summary" in d:
        e = [s["band"]["small"]["energy"] for s in d["snapshots"]]
        row["small_energy_median"] = float(np.median(e))
        row["small_energy_mean"] = float(np.mean(e))
    return row


def test_rows():
    out = {}
    for ds in ("kolmogorov2d", "kolmogorov2d_seed7", "kolmogorov2d_seed1337"):
        ev = evaluation(ds)
        for p in sorted(ev.glob("kolm_*_s3gm_K8_nfe1000.json")):
            new = json.loads(p.read_text())
            if new.get("split") != "test" or "s3gm_sampler" not in new:
                continue
            old_p = ev / "superseded_pcdps" / p.name
            out[f"{ds}/{p.stem}"] = {
                "protocol": new["protocol"], "n_obs": new["n_obs"], "s3gm_sampler": new["s3gm_sampler"],
                "new": summary(new),
                "old_pc_dps": summary(json.loads(old_p.read_text())) if old_p.exists() else None}
    return out


def main():
    out = {"tune": tune_rows(), "test": test_rows(), "divergence_rule": f"member max |x| > {DIVERGE} x truth max, or non-finite"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("TUNE (50 frames per complete arm)")
    for k, r in out["tune"].items():
        print(f"  {k:16s} frames={r['n_frames']:2d} div_frames={r['divergent_frames']} div_members={r['divergent_members']} "
              f"max/truth(stable)={r['max_member_over_truth_max_stable_frames']} "
              + " ".join(f"{c}={r[c]:.4g}" for c in KEYS))
    print("TEST (new | old pc_dps)")
    for k, r in out["test"].items():
        o = r["old_pc_dps"] or {}
        cells = " ".join(f"{c}={r['new'][c]:.4f}|{o.get(c, float('nan')):.4f}" for c in KEYS + ("small_energy_median",)
                         if c in r["new"])
        print(f"  {k:60s} {cells}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
