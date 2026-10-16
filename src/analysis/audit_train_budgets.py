#!/usr/bin/env python3
"""Audit configured vs as-run optimizer steps for the 2D fleets.

Written after two Kolmogorov rows turned out to have trained on ~half the
"~50k-step matched budget" the paper claims. The point of the script is that
every number carries the SOURCE it came from, because the sources are not
equally trustworthy:

  checkpoint   : the epoch stored inside last.pt, times steps/epoch.
                 AUTHORITATIVE -- it survives preemption and resume.
  instrumented : cost_train.json -> total_optimizer_steps. Looks authoritative
                 and is NOT: on a resumed run it counts only the final chunk.
  inferred     : rows in loss_history.csv x steps/epoch. Same trap -- the CSV
                 restarts at the resume epoch.

That trap produced a false finding once (two rows looked like they had trained
on half their budget; their last.pt showed the full count and their CSVs simply
began at epoch 1391 and 81). The script therefore reports the checkpoint number
as the answer and prints the other two only to flag disagreement.

Usage:  python audit_train_budgets.py [--json out.json]
"""
from paths import REPO_ROOT
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = REPO_ROOT   # src/analysis/ -> src -> demo
STM = ROOT / "outputs/runs"
# dataset -> (train frames, {row label: family dir}); the family of every row is the
# one fleet_select declares as reported (variants are never audited in its place).
from eval.fleet_select import REPORTED_FAMILY

def _fams(ds):
    out = {}
    for label in ("dmfgen", "senseiver", "mlp_rbf", "geofno", "sit", "s3gm", "latent_fm"):
        fam = REPORTED_FAMILY[label.replace("_", "")]
        fam = fam[ds] if isinstance(fam, dict) else fam
        if fam is not None:                 # None: no reported row on this dataset
            out[label] = fam
    return out

FLEETS = {"kolmogorov2d": (2560, _fams("kolmogorov2d")),
          "cylinder2d": (1200, _fams("cylinder2d"))}


def cfg_of(d: Path):
    """(configured epochs, batch size) from run_config.yaml or args.json."""
    rc, aj = d / "run_config.yaml", d / "args.json"
    if rc.exists():
        c = yaml.safe_load(open(rc))
        bm = c.get("baseline_model")
        st = c.get("training_stage", 1)
        pp = c.get(f"{bm}_params", {})
        pp = pp.get(f"stage{st}", pp)
        tr = pp.get("training", {}) if isinstance(pp, dict) else {}
        return tr.get("epochs"), tr.get("batch_size")
    if aj.exists():
        c = json.load(open(aj))
        return c.get("epochs"), c.get("batch_size")
    return None, None


def rows_of(d: Path):
    for pat in ("loss_history.csv", "Loss_*/losses.csv"):
        for f in glob.glob(str(d / pat)):
            try:
                return len(list(csv.reader(open(f)))) - 1
            except Exception:
                pass
    return None


def audit(dataset, deep=True):
    ntrain, fams = FLEETS[dataset]
    out = []
    for label, fam in fams.items():
        cands = [Path(p) for p in sorted(glob.glob(str(STM / dataset / fam / "*")))
                 if (Path(p) / "best.pt").exists()]
        if not cands:
            out.append({"row": label, "status": "no run"}); continue
        d = cands[-1]
        # latent FM: the protocol budget covers both stages (autoencoder + velocity
        # net), so the stage-1 run of the same family is added to the stage-2 row
        s1_steps = 0
        if label == "latent_fm":
            s1 = [c for c in cands if "Stage1" in c.name]
            if s1:
                ep1, bs1 = cfg_of(s1[-1])
                spe1 = (ntrain + bs1 - 1) // bs1 if bs1 else 0
                s1_steps = (ep1 or 0) * spe1
        ep, bs = cfg_of(d)
        spe = (ntrain + bs - 1) // bs if bs else None
        target = ep * spe if ep and spe else None
        ck_ep = None
        if deep:
            try:
                import torch
                c = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
                ck_ep = c.get("epoch") if isinstance(c, dict) else None
            except Exception:
                ck_ep = None
        ct_steps = None
        ct = d / "cost_train.json"
        if ct.exists():
            try:
                ct_steps = json.load(open(ct)).get("total_optimizer_steps")
            except Exception:
                pass
        rows = rows_of(d)
        inferred = rows * spe if rows and spe else None
        if ck_ep and spe:
            steps, source = ck_ep * spe, "checkpoint"
        elif ct_steps is not None:
            steps, source = ct_steps, "instrumented(unverified)"
        else:
            steps, source = inferred, "inferred(unverified)"
        # a resumed run shows up as cost_train/rows well below the checkpoint
        resumed = bool(steps and ((ct_steps and ct_steps < 0.95 * steps)
                                  or (inferred and inferred < 0.95 * steps)))
        if s1_steps:
            target = (target or 0) + s1_steps
            steps = (steps or 0) + s1_steps
            source += "+stage1"
        out.append({"row": label, "run": d.name, "cfg_epochs": ep, "batch": bs,
                    "steps_per_epoch": spe, "target_steps": target,
                    "as_run_steps": steps, "source": source,
                    "checkpoint_epoch": ck_ep, "cost_train_steps": ct_steps,
                    "loss_rows": rows, "resumed_run": resumed,
                    "fraction": round(steps / target, 3) if steps and target else None})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="skip reading last.pt (much faster, but then the step "
                         "counts are the UNVERIFIED cost_train/CSV values that "
                         "under-report resumed runs)")
    a = ap.parse_args()
    if a.fast:
        print("WARNING: --fast reads cost_train.json / loss_history.csv only. On a\n"
              "         RESUMED run both cover just the final chunk, so a row can look\n"
              "         like it trained half its budget when it did not. Verdict flags\n"
              "         are suppressed in this mode; drop --fast for the real answer.")
    allr = {}
    for ds in FLEETS:
        rows = audit(ds, deep=not a.fast)
        allr[ds] = rows
        # The fleet reference is the MEDIAN as-run step count: "did this row run
        # its own config" (frac) is a different question from "did every row get
        # the same budget" (vs_fleet), and only the second catches a config whose
        # epoch count was matched while its batch size was not.
        got = [r["as_run_steps"] for r in rows if r.get("as_run_steps")]
        ref = float(np.median(got)) if got else None
        print(f"\n##### {ds}   fleet reference = {ref:.0f} steps (median)" if ref else f"\n##### {ds}")
        print(f"{'row':<11} {'target':>8} {'as-run':>8} {'frac':>6} {'vs fleet':>9}  source")
        for r in rows:
            if r.get("status"):
                print(f"{r['row']:<11} {r['status']}"); continue
            f, steps = r["fraction"], r.get("as_run_steps")
            vs = steps / ref if steps and ref else None
            r["vs_fleet_reference"] = None if vs is None else round(vs, 3)
            verified = r["source"] == "checkpoint"
            flag = ""
            if verified:
                if f is not None and f < 0.9:
                    flag += "  <-- SHORT vs its own config"
                if vs is not None and vs > 1.5:
                    flag += f"  <-- {vs:.1f}x the fleet budget"
                if vs is not None and vs < 0.67:
                    flag += f"  <-- {vs:.2f}x the fleet budget"
            if r.get("resumed_run"):
                flag += "  [resumed]"
            print(f"{r['row']:<11} {str(r['target_steps']):>8} {str(steps):>8} "
                  f"{'' if f is None else f'{f:.2f}':>6} {'' if vs is None else f'{vs:.2f}':>9}"
                  f"  {r['source']}{flag}")
    if a.json:
        Path(a.json).write_text(json.dumps(allr, indent=1))
        print(f"\nwrote {a.json}")
