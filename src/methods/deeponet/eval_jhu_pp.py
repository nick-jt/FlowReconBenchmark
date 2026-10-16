"""DeepONet++ evaluation = the CANONICAL eval_deeponet_iclr.py, re-used by
import with exactly one substitution: the model builder.

eval_deeponet_iclr.py drives the model exclusively through
encode / trunk_forward / combine / n_fields / p / bottleneck_scalars /
n_params, all of which DeepONetPP exposes identically, and it reads the
architecture from cfg["deeponet_params"]["architecture"], which the
DeepONet++ config keeps under the same key.  So the entire scoring path --
fingerprint gate, compute-node guard, ensemble_metrics, nearest-sensor floor,
sensor sweep, acceptance gate -- is the canonical code, byte-identical.

Two additions, both outside the scoring path:
  * DPNPP_SNAPSHOT_IDS (env): overrides the rng.choice snapshot selection so
    the TUNE (odd cube-3) and TEST (even) split numbers of
    PLAN_IMPROVE_2026-08-30 can be produced.  Accepts the named tokens
    TUNE_ODD ({1,3,...,49}) and TEST_EVEN ({0,2,...,48}), or an explicit
    list separated by ',' or ':'.  Named tokens are the ONLY form the
    self-chained sbatch hand-off may use: sbatch --export splits its value
    on commas, so a comma-separated list survives only when the wrapper is
    invoked directly (bug found 2026-08-30, jobs 17117990/91).  Per-snapshot
    sensor draws keep the canonical torch.manual_seed(seed*777+snap), so the
    masks are the canonical ones regardless of the subset.  The snap-29
    canonical fingerprint gate fires whenever 29 is in the subset.
  * the output JSON is restamped model="deeponetpp" (+ subset provenance).
"""
from __future__ import annotations
from paths import SRC_DIR

import json
import os
import sys
from pathlib import Path


import numpy as np

import methods.deeponet.eval_jhu as E
from methods.deeponet.modelpp import build_deeponetpp





_NAMED_SUBSETS = {
    "TUNE_ODD": list(range(1, 50, 2)),    # cube-3 odd val indices, 25 snaps
    "TEST_EVEN": list(range(0, 49, 2)),   # cube-3 even val indices, 25 snaps
}


def _parse_subset(spec: str):
    if spec.upper() in _NAMED_SUBSETS:
        return _NAMED_SUBSETS[spec.upper()]
    return [int(x) for x in spec.replace(":", ",").split(",") if x.strip() != ""]


def main():
    ids_env = os.environ.get("DPNPP_SNAPSHOT_IDS", "").strip()
    subset = None
    if ids_env:
        subset = _parse_subset(ids_env)
        print(f"[subset] snapshot override active: {subset}", flush=True)

    args = E.parse_args()
    E.main(build_fn=build_deeponetpp, snapshot_ids=subset)

    out = (Path(args.out) if args.out
           else Path(args.run_dir).resolve() / "Evaluation" / "iclr_protocol_eval.json")
    payload = json.loads(out.read_text())
    payload["model"] = "deeponetpp"
    payload["arch"] = "DeepONet++ (structured branch: multires binned pooling "
    payload["arch"] += "1+4^3+8^3 + Fourier moments; RFF-MLP trunk)"
    if subset is not None:
        payload["snapshot_subset_override"] = subset
        payload["snapshot_subset_label"] = os.environ.get("DPNPP_SUBSET_LABEL", "")
    out.write_text(json.dumps(payload, indent=2))
    print(f"[eval] restamped model=deeponetpp in {out}", flush=True)


if __name__ == "__main__":
    main()
