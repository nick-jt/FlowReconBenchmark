"""Task 5: canonical 50-snapshot eval (K=8) with the val-tuned normalized
guidance, through the unmodified eval_s3gm3d machinery.

The winning setting is read from winner.json (written by tune_norm_guidance.py
on the TUNE split). s3gm_norm_guidance.install() monkey-patches
s3gm3d.s3gm_reconstruct_ensemble, so eval_s3gm3d's driver -- seeding, sensor
draws, fingerprint gate, metrics, instrumentation -- runs byte-for-byte; only
the guidance update inside the sampler changes. The alpha/beta that
eval_s3gm3d computes are ignored by the patched sampler and the output JSON is
re-labelled afterwards.

Reporting: TEST split (even snapshot indices) primary; full-50 secondary
("tuned on odd split"); TUNE (odd) shown for the selection audit. The
jhu_tuned alpha=0.5 row from finalize job 17038277 remains the pre-registered
row and is not touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from methods.s3gm.tuning.common import RUN

import methods.s3gm.norm_guidance as NG

OUT_JSON = RUN / "eval_s3gm3d_normguided_valtuned_best.json"
RUN_OUT = RUN / "Evaluation" / "norm_guidance_tuning"


def _split_agg(per_snapshot, pred):
    rows = [r for r in per_snapshot if pred(int(r["snapshot"]))]
    if not rows:
        return None
    keys = list(rows[0]["aggregate"].keys())
    agg = {k: float(np.mean([r["aggregate"][k] for r in rows])) for k in keys}
    fields = list(rows[0]["per_field"].keys())
    per_field = {f: {k: float(np.mean([r["per_field"][f][k] for r in rows]))
                     for k in rows[0]["per_field"][f]} for f in fields}
    return {"n_snapshots": len(rows), "aggregate": agg, "per_field": per_field}


def main():
    with open(HERE / "results" / "winner.json") as fh:
        winner = json.load(fh)
    print("[final] winner setting:", json.dumps(winner, indent=2), flush=True)

    traces = []
    NG.install(zeta_obs=float(winner["zeta_obs"]),
               zeta_consis=float(winner["zeta_consis"]),
               sigma_scale=winner["sigma_scale"],
               guidance_mode=winner["guidance_mode"],
               trace_store=traces, trace_limit=8)

    import methods.s3gm.eval_jhu as E
    sys.argv = [
        "eval_s3gm3d.py",
        "--run-dir", str(RUN),
        "--ckpt", "best",
        "--arm", "jhu_tuned",          # arm's alpha/beta are ignored by the patch
        "--seed", "0", "--op-seed", "1000",
        "--n-snapshots", "50", "--K", "8",
        "--cond-fields", "0", "2",
        "--n-obs", "19531", "19531",
        "--out", str(OUT_JSON),
    ]
    print("[final] NOTE: the [arm] banner below is overridden -- guidance is "
          "the installed normalized sampler, not alpha/beta.", flush=True)
    E.main()

    # ---- re-label + split aggregates -------------------------------------
    with open(OUT_JSON) as fh:
        d = json.load(fh)
    d["arm"] = "normalized_valtuned"
    d["arm_label"] = "S3GM (normalized guidance, val-tuned)"
    d["arm_note"] = ("DPS-style normalized+step-sized guidance "
                     "(zeta*g/||g|| per term, per posterior member); "
                     "zetas tuned on the TUNE split (cube-3 odd indices). "
                     "The jhu_tuned alpha=0.5 row from finalize 17038277 is "
                     "the pre-registered comparison row.")
    d["is_upstream_faithful"] = False
    d["guidance"] = {
        "guidance_mode": winner["guidance_mode"],
        "zeta_obs": winner["zeta_obs"],
        "zeta_consis": winner["zeta_consis"],
        "sigma_scale": winner["sigma_scale"],
        "tuned_on": "TUNE split (cube-3 odd indices), 6 snaps, K=2, N=200",
        "upstream_alpha_case": 0.5, "upstream_beta": 0.4,
    }
    ps = d.get("per_snapshot", [])
    d["splits"] = {
        "test_even_primary": _split_agg(ps, lambda s: s % 2 == 0),
        "full50_secondary_tuned_on_odd": {"n_snapshots": len(ps),
                                          "aggregate": d["aggregate"],
                                          "per_field": d["per_field"]},
        "tune_odd_selection_audit": _split_agg(ps, lambda s: s % 2 == 1),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(d, fh, indent=2)
    print("[final] re-labelled + split aggregates written:", OUT_JSON, flush=True)
    te = d["splits"]["test_even_primary"]
    print(f"[final] TEST(even) agg relL2={te['aggregate']['rel_l2_mean']:.4f} "
          f"Ux={te['per_field'].get('Ux', {}).get('rel_l2_mean', float('nan')):.4f} "
          f"Uz={te['per_field'].get('Uz', {}).get('rel_l2_mean', float('nan')):.4f} "
          f"crps={te['aggregate']['crps']:.4f}", flush=True)

    # ---- diagnostics from the first evaluated snapshot -------------------
    RUN_OUT.mkdir(parents=True, exist_ok=True)
    if traces:
        with open(RUN_OUT / "final_eval_traces.json", "w") as fh:
            json.dump({"setting": d["guidance"], "traces": traces}, fh)
        NG.save_diag_figure(traces, RUN_OUT / "diag_final_eval.png",
                            title="final eval (K=8, N=200): "
                                  f"zo={winner['zeta_obs']:g} "
                                  f"zc={winner['zeta_consis']:g} "
                                  f"s(t)={winner['sigma_scale']}")
        print("[final] diagnostics written to", RUN_OUT, flush=True)


if __name__ == "__main__":
    main()
