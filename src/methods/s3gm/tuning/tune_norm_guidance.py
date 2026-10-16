"""Tasks 3+4: tune zeta_obs / zeta_consis for the normalized S3GM guidance on
the TUNE split, with per-sigma diagnostics per setting.

Protocol (PLAN_IMPROVE_2026-08-30.md fleet hygiene):
  * TUNE split = cube-3 ODD val indices; we use {1, 9, 17, 29, 41, 49}
    (6 snaps, includes the canonical-fingerprint snapshot 29).
  * Canonical sensor draw: helpers.build_sparse_condition, seed 0,
    torch.manual_seed(0*777 + snap), 19,531 sensors on channels {0 (Ux), 2 (Uz)}.
  * K=2 posterior draws, N=200 denoising steps, best.pt (ep 1390), EMA weights.

Search (coordinate, not full grid -- the isolation sweep showed beta is a weak
knob at this protocol):
  A: zeta_obs in {10, 30, 100, 300, 1000, 3000}, zeta_consis=0, s(t)=1
  R: refine zeta_obs around the stage-A best (x/1.78, x1.78)
  B: best zeta_obs x zeta_consis in {0.03, 0.1, 0.3} * zeta_obs
  C: ablations -- winner with sigma_scale="pigdm" (PiGDM-motivated late decay),
     and the UN-normalized alpha=1.0 / beta=0.004 stability-edge arm
     (isolation_final.json: agg 0.652 at N=200/K=1) through the same traced
     code path (guidance_mode="raw").

Selection: mean over snaps of (relL2_Ux + relL2_Uz)/2 of the ensemble mean
(the two conditioned channels are primary), subject to sanity gates
max|x| < 10 and final obs_rmse_z < sqrt(2) with all values finite.

Outputs (results dir + a NEW run-dir subdir Evaluation/norm_guidance_tuning/):
  tune_results.json    -- full table, updated incrementally
  winner.json          -- the selected normalized setting (read by final eval)
  traces_<name>.json   -- per-sigma traces (snapshot 1, both draws)
  diag_<name>.png      -- 3-panel per-sigma diagnostic figure
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from methods.s3gm.tuning.common import RUN, load_run

import methods.s3gm.s3gm3d as S
import methods.s3gm.norm_guidance as NG
from methods.dmfgen.helpers import build_sparse_condition
from methods.dmfgen.ensemble_eval import check_canonical_fingerprint, ensemble_metrics, require_compute_node
import methods.s3gm.sde as s3gm_sde

SEED = 0
SNAPS = [1, 9, 17, 29, 41, 49]
K = 2
N_STEPS = 200
COND_FIELDS = [0, 2]
N_OBS = [19531, 19531]

RESULTS = HERE / "results"
RUN_OUT = RUN / "Evaluation" / "norm_guidance_tuning"


def _sane(rec):
    vals = [rec["mean"]["rel_l2_agg"], rec["mean"]["obs_rmse_final"], rec["mean"]["max_abs"]]
    return (all(np.isfinite(v) for v in vals)
            and rec["mean"]["max_abs"] < 10.0
            and rec["mean"]["obs_rmse_final"] < math.sqrt(2.0))


class Tuner:
    def __init__(self):
        require_compute_node()
        self.cfg, self.adapter, self.bundle, self.val_set, self.ck = load_run("best")
        cmp_ = self.bundle.components
        self.cmp = cmp_
        self.device = self.bundle.device
        self.results = []
        RESULTS.mkdir(parents=True, exist_ok=True)
        RUN_OUT.mkdir(parents=True, exist_ok=True)

        # Pre-build sensor videos once per snapshot (draw is deterministic).
        self.obs = {}
        for snap in SNAPS:
            item = self.val_set[int(snap)]
            coords = item["coords"].unsqueeze(0).to(self.device)
            fields = item["fields"].unsqueeze(0).to(self.device)
            torch.manual_seed(SEED * 777 + int(snap))
            oc, ov, om, oi, ofid = build_sparse_condition(
                coords_full=coords, fields_full=fields,
                cond_fields=COND_FIELDS, n_obs_min=N_OBS, n_obs_max=N_OBS)
            sel = oi[om > 0]
            print(f"[seedcheck] snap={snap} sensors={int(sel.numel())} "
                  f"idx_sum={int(sel.sum().item())}", flush=True)
            check_canonical_fingerprint(int(snap), int(sel.numel()),
                                        int(sel.sum().item()), SEED,
                                        COND_FIELDS, N_OBS)
            yv, mv = S.obs_to_video(ov, om, ofid, oi, self.val_set.num_fields,
                                    self.val_set.num_points,
                                    cmp_["Nz"], cmp_["Ny"], cmp_["Nx"],
                                    cmp_["H_pad"], cmp_["W_pad"])
            self.obs[snap] = (yv, mv, fields[0].cpu().numpy())

    def run_setting(self, name, guidance_mode, zeta_obs, zeta_consis, sigma_scale):
        cmp_ = self.cmp
        per_snap = []
        traces_first = []
        t0 = time.perf_counter()
        with self.adapter.evaluation_weights(self.bundle):
            self.bundle.model.eval()
            for snap in SNAPS:
                yv, mv, true_np = self.obs[snap]
                sde_s = s3gm_sde.VESDE(config=None, sigma_min=cmp_["sigma_min"],
                                 sigma_max=cmp_["sigma_max"], N=N_STEPS)
                torch.manual_seed(SEED * 131 + int(snap))
                store = [] if snap == SNAPS[0] else None
                with torch.enable_grad():
                    vol = NG.s3gm_reconstruct_ensemble_norm(
                        self.bundle.model, sde_s, yv, mv, cmp_["Nz"], K=K,
                        chunk=1, t_win=cmp_["t_win"], overlap=cmp_["overlap"],
                        zeta_obs=zeta_obs, zeta_consis=zeta_consis,
                        sigma_scale=sigma_scale, guidance_mode=guidance_mode,
                        device=self.device, progress=False,
                        trace_store=store, trace_limit=K)
                if store:
                    traces_first = store
                with torch.no_grad():
                    nobs = float(mv.sum()) * K
                    obs_rmse = float(torch.sqrt(
                        (((vol - yv) * mv) ** 2).sum() / max(nobs, 1.0)))
                    max_abs = float(vol.abs().max())
                ens = S.video_to_pointcloud(vol, cmp_["Nz"], cmp_["Ny"],
                                            cmp_["Nx"]).float().cpu().numpy()
                m = ensemble_metrics(ens, true_np,
                                     [str(x) for x in self.val_set.field_names])
                rec = {"snapshot": int(snap),
                       "rel_l2": {f: m["per_field"][f]["rel_l2_mean"]
                                  for f in m["per_field"]},
                       "rel_l2_agg": m["aggregate"]["rel_l2_mean"],
                       "crps": m["aggregate"]["crps"],
                       "obs_rmse_final": obs_rmse,
                       "max_abs": max_abs}
                per_snap.append(rec)
                print(f"[tune] {name} snap={snap} "
                      f"Ux={rec['rel_l2'].get('Ux', float('nan')):.4f} "
                      f"Uz={rec['rel_l2'].get('Uz', float('nan')):.4f} "
                      f"agg={rec['rel_l2_agg']:.4f} obs_rmse={obs_rmse:.3e} "
                      f"max_abs={max_abs:.3f}", flush=True)
                del vol, ens
                torch.cuda.empty_cache()

        fields = list(per_snap[0]["rel_l2"].keys())
        mean = {
            "rel_l2": {f: float(np.mean([r["rel_l2"][f] for r in per_snap]))
                       for f in fields},
            "rel_l2_agg": float(np.mean([r["rel_l2_agg"] for r in per_snap])),
            "crps": float(np.mean([r["crps"] for r in per_snap])),
            "obs_rmse_final": float(np.mean([r["obs_rmse_final"] for r in per_snap])),
            "max_abs": float(np.max([r["max_abs"] for r in per_snap])),
        }
        mean["rel_l2_UxUz"] = float(np.mean([mean["rel_l2"].get("Ux", np.nan),
                                             mean["rel_l2"].get("Uz", np.nan)]))
        rec = {"name": name, "guidance_mode": guidance_mode,
               "zeta_obs": zeta_obs, "zeta_consis": zeta_consis,
               "sigma_scale": sigma_scale, "K": K, "n_steps": N_STEPS,
               "snapshots": SNAPS, "per_snapshot": per_snap, "mean": mean,
               "wall_s": time.perf_counter() - t0}
        rec["sane"] = bool(_sane(rec))
        self.results.append(rec)
        print(f"[tune] SETTING {name}: UxUz={mean['rel_l2_UxUz']:.4f} "
              f"agg={mean['rel_l2_agg']:.4f} obs_rmse={mean['obs_rmse_final']:.3e} "
              f"max_abs={mean['max_abs']:.3f} sane={rec['sane']} "
              f"wall={rec['wall_s']:.0f}s", flush=True)

        # diagnostics (task 4)
        if traces_first:
            tr_path = RESULTS / f"traces_{name}.json"
            with open(tr_path, "w") as fh:
                json.dump({"setting": {k: rec[k] for k in
                                       ("name", "guidance_mode", "zeta_obs",
                                        "zeta_consis", "sigma_scale")},
                           "snapshot": SNAPS[0], "traces": traces_first}, fh)
            NG.save_diag_figure(
                traces_first, RESULTS / f"diag_{name}.png",
                title=f"{name}: mode={guidance_mode} zo={zeta_obs:g} "
                      f"zc={zeta_consis:g} s(t)={sigma_scale} | snap {SNAPS[0]}, K={K}")
        self.flush()
        return rec

    def flush(self):
        payload = {"run_dir": str(RUN), "checkpoint": "best.pt",
                   "epoch": int(self.ck.get("epoch", 0)),
                   "protocol": {"seed": SEED, "snapshots": SNAPS, "K": K,
                                "n_steps": N_STEPS, "cond_fields": COND_FIELDS,
                                "n_obs": N_OBS, "split": "TUNE (cube-3 odd)"},
                   "selection_metric": "mean relL2 of (Ux,Uz) ensemble mean, "
                                       "sanity: max|x|<10, obs_rmse_z<sqrt(2)",
                   "results": self.results}
        for d in (RESULTS, RUN_OUT):
            with open(d / "tune_results.json", "w") as fh:
                json.dump(payload, fh, indent=2)

    def best_norm(self, subset=None):
        cands = [r for r in (subset or self.results)
                 if r["sane"] and r["guidance_mode"] == "norm"]
        if not cands:
            return None
        return min(cands, key=lambda r: r["mean"]["rel_l2_UxUz"])


def main():
    T = Tuner()

    # ---- stage A: zeta_obs sweep, no consistency term
    stage_a = []
    for zo in [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]:
        stage_a.append(T.run_setting(f"A_zo{zo:g}", "norm", zo, 0.0, "none"))
    best_a = T.best_norm(stage_a)
    if best_a is None:
        raise SystemExit("[tune] no sane stage-A setting; aborting")
    zo0 = best_a["zeta_obs"]

    # ---- stage R: refine around the stage-A best (grid spacing ~3.16x)
    stage_r = []
    for zo in [zo0 / 1.78, zo0 * 1.78]:
        stage_r.append(T.run_setting(f"R_zo{zo:g}", "norm", zo, 0.0, "none"))
    zo_best = T.best_norm(stage_a + stage_r)["zeta_obs"]

    # ---- stage B: consistency step size at the best zeta_obs
    stage_b = []
    for frac in [0.03, 0.1, 0.3]:
        zc = zo_best * frac
        stage_b.append(T.run_setting(f"B_zo{zo_best:g}_zc{zc:g}",
                                     "norm", zo_best, zc, "none"))
    winner_so_far = T.best_norm()

    # ---- stage C: ablations
    T.run_setting(f"C_pigdm_zo{winner_so_far['zeta_obs']:g}_"
                  f"zc{winner_so_far['zeta_consis']:g}", "norm",
                  winner_so_far["zeta_obs"], winner_so_far["zeta_consis"],
                  "pigdm")
    T.run_setting("C_raw_a1.0_b0.004", "raw", 1.0, 0.004, "none")

    # ---- select winner (best sane NORMALIZED setting; raw arm is comparison)
    winner = T.best_norm()
    raw = [r for r in T.results if r["guidance_mode"] == "raw"]
    payload = {
        "label": "S3GM (normalized guidance, val-tuned)",
        "guidance_mode": winner["guidance_mode"],
        "zeta_obs": winner["zeta_obs"],
        "zeta_consis": winner["zeta_consis"],
        "sigma_scale": winner["sigma_scale"],
        "tune_mean": winner["mean"],
        "tune_protocol": {"seed": SEED, "snapshots": SNAPS, "K": K,
                          "n_steps": N_STEPS, "split": "TUNE (cube-3 odd)"},
        "raw_comparison_arm": ({"zeta_obs": 1.0, "zeta_consis": 0.004,
                                "mean": raw[0]["mean"]} if raw else None),
    }
    for d in (RESULTS, RUN_OUT):
        with open(d / "winner.json", "w") as fh:
            json.dump(payload, fh, indent=2)
    print(f"[tune] WINNER: {json.dumps(payload['tune_mean'])} "
          f"zo={winner['zeta_obs']:g} zc={winner['zeta_consis']:g} "
          f"s(t)={winner['sigma_scale']}", flush=True)

    # copy per-setting diagnostics to the run-dir subdir as well
    for f in RESULTS.glob("traces_*.json"):
        shutil.copy2(f, RUN_OUT / f.name)
    for f in RESULTS.glob("diag_*.png"):
        shutil.copy2(f, RUN_OUT / f.name)
    print("[tune] done", flush=True)


if __name__ == "__main__":
    main()
