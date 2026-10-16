"""Supplementary tuning arms for the normalized S3GM guidance (job 2).

Job 1 (tune_norm_guidance.py) showed the stage-A optimum sits between
zeta_obs=30 and 100 (UxUz 0.170 / 0.153), but its in-process refinement keys
off a max|x|<10 gate measured on the PADDED video -- too strict, and possibly
dominated by the unobserved 3-pixel pad band. This job:

  * runs the missing refinement arms zeta_obs in {56, 75} (zc=0),
  * records max|x| on BOTH the padded video and the cropped 125^3 volume,
  * merges its records with job 1's tune_results.json and picks the best
    normalized arm under the divergence-only gate (finite, obs_rmse_z < 1.0,
    relL2_agg < 1.0), then
  * runs the consistency sweep zc in {0.03, 0.1, 0.3}*zo_best and the
    PiGDM-scaled ablation at the best (zo, zc).

Winner selection across BOTH jobs is done afterwards (merge_winner.py);
this job does NOT write winner.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

import methods.s3gm.tuning.tune_norm_guidance as TN

# Redirect outputs so the two jobs never write the same files.
RESULTS2 = HERE / "results2"
RUN_OUT2 = TN.RUN / "Evaluation" / "norm_guidance_tuning" / "job2"
TN.RESULTS = RESULTS2
TN.RUN_OUT = RUN_OUT2

import methods.s3gm.s3gm3d as S
import methods.s3gm.norm_guidance as NG
import methods.s3gm.sde as s3gm_sde

JOB1_RESULTS = HERE / "results" / "tune_results.json"


class Tuner2(TN.Tuner):
    def run_setting(self, name, guidance_mode, zeta_obs, zeta_consis, sigma_scale):
        # Same as job 1 but also records max|x| on the CROPPED volume.
        cmp_ = self.cmp
        per_snap = []
        traces_first = []
        t0 = time.perf_counter()
        with self.adapter.evaluation_weights(self.bundle):
            self.bundle.model.eval()
            for snap in TN.SNAPS:
                yv, mv, true_np = self.obs[snap]
                sde_s = s3gm_sde.VESDE(config=None, sigma_min=cmp_["sigma_min"],
                                 sigma_max=cmp_["sigma_max"], N=TN.N_STEPS)
                torch.manual_seed(TN.SEED * 131 + int(snap))
                store = [] if snap == TN.SNAPS[0] else None
                with torch.enable_grad():
                    vol = NG.s3gm_reconstruct_ensemble_norm(
                        self.bundle.model, sde_s, yv, mv, cmp_["Nz"], K=TN.K,
                        chunk=1, t_win=cmp_["t_win"], overlap=cmp_["overlap"],
                        zeta_obs=zeta_obs, zeta_consis=zeta_consis,
                        sigma_scale=sigma_scale, guidance_mode=guidance_mode,
                        device=self.device, progress=False,
                        trace_store=store, trace_limit=TN.K)
                if store:
                    traces_first = store
                with torch.no_grad():
                    nobs = float(mv.sum()) * TN.K
                    obs_rmse = float(torch.sqrt(
                        (((vol - yv) * mv) ** 2).sum() / max(nobs, 1.0)))
                    max_abs = float(vol.abs().max())
                ens = S.video_to_pointcloud(vol, cmp_["Nz"], cmp_["Ny"],
                                            cmp_["Nx"]).float().cpu().numpy()
                max_abs_cropped = float(np.abs(ens).max())
                m = TN.ensemble_metrics(ens, true_np,
                                        [str(x) for x in self.val_set.field_names])
                rec = {"snapshot": int(snap),
                       "rel_l2": {f: m["per_field"][f]["rel_l2_mean"]
                                  for f in m["per_field"]},
                       "rel_l2_agg": m["aggregate"]["rel_l2_mean"],
                       "crps": m["aggregate"]["crps"],
                       "obs_rmse_final": obs_rmse,
                       "max_abs": max_abs,
                       "max_abs_cropped": max_abs_cropped}
                per_snap.append(rec)
                print(f"[tune2] {name} snap={snap} "
                      f"Ux={rec['rel_l2'].get('Ux', float('nan')):.4f} "
                      f"Uz={rec['rel_l2'].get('Uz', float('nan')):.4f} "
                      f"agg={rec['rel_l2_agg']:.4f} obs_rmse={obs_rmse:.3e} "
                      f"max_abs={max_abs:.3f} max_abs_cropped={max_abs_cropped:.3f}",
                      flush=True)
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
            "max_abs_cropped": float(np.max([r["max_abs_cropped"] for r in per_snap])),
        }
        mean["rel_l2_UxUz"] = float(np.mean([mean["rel_l2"].get("Ux", np.nan),
                                             mean["rel_l2"].get("Uz", np.nan)]))
        rec = {"name": name, "guidance_mode": guidance_mode,
               "zeta_obs": zeta_obs, "zeta_consis": zeta_consis,
               "sigma_scale": sigma_scale, "K": TN.K, "n_steps": TN.N_STEPS,
               "snapshots": TN.SNAPS, "per_snapshot": per_snap, "mean": mean,
               "wall_s": time.perf_counter() - t0}
        rec["sane"] = bool(_gate(rec))
        self.results.append(rec)
        print(f"[tune2] SETTING {name}: UxUz={mean['rel_l2_UxUz']:.4f} "
              f"agg={mean['rel_l2_agg']:.4f} obs_rmse={mean['obs_rmse_final']:.3e} "
              f"max_abs={mean['max_abs']:.3f} "
              f"max_abs_cropped={mean['max_abs_cropped']:.3f} sane={rec['sane']} "
              f"wall={rec['wall_s']:.0f}s", flush=True)

        if traces_first:
            with open(TN.RESULTS / f"traces_{name}.json", "w") as fh:
                json.dump({"setting": {k: rec[k] for k in
                                       ("name", "guidance_mode", "zeta_obs",
                                        "zeta_consis", "sigma_scale")},
                           "snapshot": TN.SNAPS[0], "traces": traces_first}, fh)
            NG.save_diag_figure(
                traces_first, TN.RESULTS / f"diag_{name}.png",
                title=f"{name}: mode={guidance_mode} zo={zeta_obs:g} "
                      f"zc={zeta_consis:g} s(t)={sigma_scale} | snap {TN.SNAPS[0]}, K={TN.K}")
        self.flush()
        return rec


def _gate(rec):
    """Divergence-only gate: finite, sample not independent of its own
    observations (obs_rmse_z < 1.0), reconstruction better than the
    unconditional reference (agg relL2 < 1.0). max|x| recorded, not gated."""
    m = rec["mean"]
    vals = [m["rel_l2_agg"], m["obs_rmse_final"]]
    return (all(np.isfinite(v) for v in vals)
            and m["obs_rmse_final"] < 1.0
            and m["rel_l2_agg"] < 1.0)


def main():
    RESULTS2.mkdir(parents=True, exist_ok=True)
    RUN_OUT2.mkdir(parents=True, exist_ok=True)
    T = Tuner2()

    for zo in [56.0, 75.0]:
        T.run_setting(f"A2_zo{zo:g}", "norm", zo, 0.0, "none")

    # merge with job 1 stage-A records for the zo choice
    merged = list(T.results)
    if JOB1_RESULTS.exists():
        with open(JOB1_RESULTS) as fh:
            j1 = json.load(fh)["results"]
        merged += [r for r in j1 if r["guidance_mode"] == "norm"
                   and r["sigma_scale"] == "none" and r["zeta_consis"] == 0.0]
    cands = [r for r in merged if _gate(r)]
    best = min(cands, key=lambda r: r["mean"]["rel_l2_UxUz"])
    zo_best = best["zeta_obs"]
    print(f"[tune2] merged stage-A best: zo={zo_best:g} "
          f"UxUz={best['mean']['rel_l2_UxUz']:.4f}", flush=True)

    for frac in [0.03, 0.1, 0.3]:
        zc = zo_best * frac
        T.run_setting(f"B2_zo{zo_best:g}_zc{zc:g}", "norm", zo_best, zc, "none")

    seen = {id(r) for r in T.results}
    pool = T.results + [r for r in merged if id(r) not in seen]
    pick = min([r for r in pool if _gate(r)],
               key=lambda r: r["mean"]["rel_l2_UxUz"])
    T.run_setting(f"C2_pigdm_zo{pick['zeta_obs']:g}_zc{pick['zeta_consis']:g}",
                  "norm", pick["zeta_obs"], pick["zeta_consis"], "pigdm")

    print("[tune2] done (winner selection happens in merge_winner.py)", flush=True)


if __name__ == "__main__":
    main()
