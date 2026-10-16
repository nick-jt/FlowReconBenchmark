"""Kolmogorov S3GM row on the TUNE half under the fleet_2d --s3gm-sampler variants.

Collects the per-snapshot records that s3gm_sampler_tune.sbatch writes to
outputs/rescore_2d/s3gm_sampler_tune/ (every 5th canonical TUNE frame, canonical
sensors and noise seeds, K=8, 655 sensors) and the reported sampler's existing TUNE
records for the same frames. Per arm: ensemble-mean and single-member relative L2,
CRPS, spread/error and the k=32-128 single-member kinetic-energy ratio (median and
mean over frames), all means over the scored frames.

    python -I src/analysis/diagnostics/s3gm_sampler_tune.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/rescore_2d/s3gm_sampler_tune"
EVAL = REPO / ("outputs/runs/kolmogorov2d/baseline_s3gm/"
               "Baseline_s3gm_Stage1_DemoN63_20261003_144213/Evaluation")
KEYS = ("rel_l2_mean", "rel_l2_single", "crps", "spread_error_ratio", "coverage_90")


def row(recs, key=None):
    """Mean metrics over snapshot records; key selects a nested block (before_denoise)."""
    blocks = [r[key] if key else r for r in recs]
    out = {k: float(np.mean([b["aggregate"][k] for b in blocks])) for k in KEYS}
    if all("band" in b for b in blocks):
        e = [b["band"]["small"]["energy"] for b in blocks]
        out["small_energy_median"] = float(np.median(e))
        out["small_energy_mean"] = float(np.mean(e))
    out["n_frames"] = len(blocks)
    out["rel_l2_mean_per_frame"] = [b["aggregate"]["rel_l2_mean"] for b in blocks]
    return out


def main():
    arms, snaps = {}, None
    for p in sorted(OUT.glob("kolm_tune_*_s3gm_K8_nfe*.json")):
        d = json.loads(p.read_text())
        recs = d["snapshots"]
        snaps = snaps or [r["snapshot"] for r in recs]
        if [r["snapshot"] for r in recs] != snaps:
            raise SystemExit(f"{p.name}: frames differ from the other arms")
        smp = d["s3gm_sampling"]
        name = (f"upstream_a{smp['alpha_case']:g}_N{d['nfe']}" if d["ode_solver"] == "s3gm_upstream"
                else f"{d['ode_solver']}_N{d['nfe']}")
        arms[name] = {**row(recs), "seconds_per_member": d["inference_seconds_per_field_mean"]}
        if "before_denoise" in recs[0]:
            arms[f"pc_dps_N{d['nfe']} (same chains, before denoise)"] = row(recs, "before_denoise")
    if snaps is None:
        raise SystemExit("no arm JSONs yet")
    base = arms.get("pc_dps_N1000 (same chains, before denoise)")
    if base:   # paired: same frames, sensors and seeds
        for r in arms.values():
            r["frames_better_than_pc_dps_N1000"] = int(sum(
                a < b for a, b in zip(r["rel_l2_mean_per_frame"], base["rel_l2_mean_per_frame"])))
    # The reported sampler's existing TUNE records on the same frames (no band metrics).
    old = [json.loads((EVAL / f"crps_tune_nfe1000_ema_if_kept_n655_snap{s}.json").read_text()) for s in snaps]
    ref = {k: float(np.mean([r["aggregate"][k] for r in old])) for k in KEYS}
    full = json.loads((EVAL / "kolm_tune_best_s3gm_K8_nfe1000.json").read_text())["summary"]["aggregate"]
    out = {"frames": snaps, "split": "tune", "K": 8, "n_obs": 655,
           "reported_sampler_existing_tune_records_same_frames": ref,
           "reported_sampler_existing_tune_50_frames": {k: full[k] for k in KEYS},
           "arms": arms}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    cols = KEYS + ("small_energy_median", "small_energy_mean")
    print(f"{'arm':48s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
    for name, r in [("existing TUNE records (pc_dps N1000)", ref)] + list(arms.items()):
        print(f"{name:48s} " + " ".join(f"{r[c]:12.4f}" if c in r else f"{'-':>12s}" for c in cols))


if __name__ == "__main__":
    sys.exit(main())
