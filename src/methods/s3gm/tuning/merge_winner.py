"""Merge the two tuning jobs' tables, select the winning normalized setting,
and write winner.json for run_final_eval.py.

Gate (divergence-only, per the task's selection metric: per-channel relL2
primary + no divergence + obs_rmse sanity):
  finite metrics, obs_rmse_z < 1.0 (sample must track its own observations;
  1.41 = statistically independent), agg relL2 < 1.0 (better than the
  unconditional reference 1.11). max|x| is reported as a caveat -- the padded
  video includes a 3-pixel band with no observations; the cropped-volume
  max|x| from job 2 is the physical number.
"""
from __future__ import annotations
from paths import RUNS_DIR

import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = Path(f"{RUNS_DIR}/JHU/baseline_s3gm/matched/Baseline_s3gm_Stage1_DemoN94_20260828_224632")
RUN_OUT = RUN / "Evaluation" / "norm_guidance_tuning"


def _gate(rec):
    m = rec["mean"]
    return (all(np.isfinite(v) for v in (m["rel_l2_agg"], m["obs_rmse_final"]))
            and m["obs_rmse_final"] < 1.0 and m["rel_l2_agg"] < 1.0)


def main():
    results = []
    for name in ("results/tune_results.json", "results2/tune_results.json"):
        p = HERE / name
        if p.exists():
            with open(p) as fh:
                d = json.load(fh)
            for r in d["results"]:
                r["source"] = name
                results.append(r)
    print(f"[merge] {len(results)} settings loaded")
    for r in sorted(results, key=lambda r: (r["guidance_mode"],
                                            r["zeta_obs"], r["zeta_consis"],
                                            r["sigma_scale"])):
        m = r["mean"]
        print(f"  {r['name']:32s} mode={r['guidance_mode']:4s} "
              f"zo={r['zeta_obs']:8g} zc={r['zeta_consis']:6g} "
              f"s(t)={r['sigma_scale']:5s} UxUz={m['rel_l2_UxUz']:.4f} "
              f"agg={m['rel_l2_agg']:.4f} crps={m['crps']:.4f} "
              f"obs_rmse={m['obs_rmse_final']:.3e} max|x|={m['max_abs']:.1f}"
              + (f" max|x|crop={m['max_abs_cropped']:.1f}"
                 if "max_abs_cropped" in m else "")
              + ("  [GATED-OUT]" if not _gate(r) else ""))

    norm = [r for r in results if r["guidance_mode"] == "norm" and _gate(r)]
    winner = min(norm, key=lambda r: r["mean"]["rel_l2_UxUz"])
    raw = [r for r in results if r["guidance_mode"] == "raw"]

    payload = {
        "label": "S3GM (normalized guidance, val-tuned)",
        "guidance_mode": winner["guidance_mode"],
        "zeta_obs": winner["zeta_obs"],
        "zeta_consis": winner["zeta_consis"],
        "sigma_scale": winner["sigma_scale"],
        "tune_mean": winner["mean"],
        "tune_protocol": {"seed": 0, "snapshots": winner["snapshots"],
                          "K": winner["K"], "n_steps": winner["n_steps"],
                          "split": "TUNE (cube-3 odd)"},
        "selection": "min mean relL2(Ux,Uz); gate: obs_rmse_z<1.0, agg relL2<1.0, finite",
        "raw_comparison_arm": ({"zeta_obs": raw[0]["zeta_obs"],
                                "zeta_consis": raw[0]["zeta_consis"],
                                "mean": raw[0]["mean"]} if raw else None),
        "n_settings_searched": len(results),
    }
    for d in (HERE / "results", RUN_OUT):
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "winner.json", "w") as fh:
            json.dump(payload, fh, indent=2)
    print(f"[merge] WINNER: mode={winner['guidance_mode']} "
          f"zo={winner['zeta_obs']:g} zc={winner['zeta_consis']:g} "
          f"s(t)={winner['sigma_scale']} "
          f"UxUz={winner['mean']['rel_l2_UxUz']:.4f} "
          f"agg={winner['mean']['rel_l2_agg']:.4f}")

    # merged table copy for the run dir
    with open(RUN_OUT / "tune_results_merged.json", "w") as fh:
        json.dump({"results": results, "winner": payload}, fh, indent=2)

    # collect job-2 diagnostics into the run dir too
    for f in (HERE / "results2").glob("traces_*.json"):
        shutil.copy2(f, RUN_OUT / f.name)
    for f in (HERE / "results2").glob("diag_*.png"):
        shutil.copy2(f, RUN_OUT / f.name)


if __name__ == "__main__":
    main()
