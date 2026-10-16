"""Isolate WHICH term of the S3GM guided sampler diverges.

Context: at epoch 200 the score network reached train 0.0265 / val 0.0350
against a zero-score baseline of 1.0 (~97% of the denoising signal captured),
yet obs_rmse_z stayed pinned at ~5.6e7 with no trend from epoch 40 to 200.
"Undertrained score" is therefore falsified, leaving two candidates:
  (a) upstream's guidance genuinely diverges at this scale/protocol, or
  (b) a bug in our port of the sampler.

Three gradient terms act on x each step, so alpha=0 alone does NOT isolate the
DPS term -- the window-overlap consistency term (beta) is also a gradient:
    loss = alpha * loss_dps + beta * loss_consis
Arms below separate all three.

REFERENCE VALUES (fields are per-field z-scored, sigma_data = 1):
  sample std   ~ 1.0   a sane unconditional draw
  obs_rmse_z   ~ 1.41  sqrt(2); sample statistically INDEPENDENT of the sensors
  obs_rmse_z   < 1.41  guidance is actually helping
  obs_rmse_z  >> 1.41  guidance pushing the sample away from its own data

alpha = 5.0 is not arbitrary: it is upstream's OWN rule
(sample_kol.ipynb) alpha = alpha_case / sqrt(1 - sparsity) = 0.5 / sqrt(0.01)
evaluated at our protocol's 1% sensor coverage.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch

import methods.s3gm.s3gm3d as S
from methods.dmfgen.helpers import build_sparse_condition
import methods.common.config as bconfig
import methods.common.data as bdata
import methods.s3gm.sde as s3gm_sde

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--ckpt", default="last")
ap.add_argument("--n-steps", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto")
ap.add_argument("--out", default=None)
a = ap.parse_args()

run_dir = Path(a.run_dir)
cfg = bconfig.validate_and_normalize_config(bconfig.load_yaml(run_dir / "run_config.yaml"))
p = cfg["shared"]["paths"]
if not Path(p["data_path"]).exists() and p.get("data_path_shared"):
    p["data_path"] = p["data_path_shared"]
# Pick the visible CUDA device with the most free memory. SLURM packs jobs onto
# partially-used nodes, and if GPU isolation is not enforced every job that
# blindly takes cuda:0 collides on the same physical device.
_n = torch.cuda.device_count()
_stats = []
for _i in range(_n):
    _f, _t = torch.cuda.mem_get_info(torch.device(f"cuda:{_i}"))
    _stats.append((_f, _i, _t))
    print(f"[iso] visible cuda:{_i} free={_f/2**30:.2f} GiB of {_t/2**30:.2f} GiB", flush=True)
if a.device == "auto":
    _best = max(_stats)
    dev = torch.device(f"cuda:{_best[1]}")
    print(f"[iso] auto-selected {dev} ({_best[0]/2**30:.2f} GiB free)", flush=True)
else:
    dev = torch.device(a.device)
torch.cuda.set_device(dev)
print(f"[iso] device={dev} ({torch.cuda.get_device_name(dev)})", flush=True)
# SLURM packs jobs onto partially-used nodes, so the allocated GPU can already
# hold another process (observed: a co-tenant holding 69.4 of 79.1 GiB). The
# sampler needs ~13.3 GiB, so wait for the co-tenant rather than OOM-ing.
_need_gib, _waited = 18.0, 0.0
while True:
    _free, _tot = torch.cuda.mem_get_info(dev)
    _f = _free / 2**30
    if _f >= _need_gib or _waited >= 1500.0:
        break
    if _waited == 0.0:
        print(f"[iso] only {_f:.2f} GiB free of {_tot/2**30:.2f}; a co-tenant is on "
              f"this GPU. Waiting up to 25 min for >= {_need_gib} GiB...", flush=True)
    time.sleep(60.0); _waited += 60.0
print(f"[iso] gpu_free={_f:.2f} GiB of {_tot/2**30:.2f} GiB "
      f"(waited {_waited:.0f}s)", flush=True)
if _f < _need_gib:
    print(f"[iso] ABORTING CLEANLY: {_f:.2f} GiB free < {_need_gib} GiB needed. "
          f"This is GPU contention, NOT a result about the sampler.", flush=True)
    raise SystemExit(75)

stats = run_dir / "dataset_stats.pt"
val = bdata.build_dataset(cfg, split="val", stats_path=stats)
tr = bdata.build_dataset(cfg, split="train", stats_path=stats)
ad = S.S3GM3DAdapter()
b = ad.build_for_training(cfg, dev, run_dir, tr, val)
ck = bconfig.safe_torch_load(run_dir / f"{a.ckpt}.pt", map_location="cpu")
ad.load_checkpoint(b, ck)
print(f"[iso] ckpt epoch={ck['epoch']} train_loss={ck['train_loss']:.5f} "
      f"val_loss={ck['val_loss']:.5f} opt_steps={ck.get('total_opt_steps')}", flush=True)

c = b.components
Nz, Ny, Nx = c["Nz"], c["Ny"], c["Nx"]
Hp, Wp = c["H_pad"], c["W_pad"]

# Canonical protocol seeding, first snapshot of the canonical order.
rng = np.random.default_rng(a.seed)
snap = int(rng.choice(len(val), size=len(val), replace=False)[0])
item = val[snap]
coords = item["coords"].unsqueeze(0).to(dev)
truth = item["fields"].unsqueeze(0).to(dev)
torch.manual_seed(a.seed * 777 + snap)
oc, ov, om, oi, ofid = build_sparse_condition(
    coords_full=coords, fields_full=truth, cond_fields=[0, 2],
    n_obs_min=[19531, 19531], n_obs_max=[19531, 19531])
sel = oi[om > 0]
print(f"[seedcheck] snap={snap} sensors={int(sel.numel())} idx_sum={int(sel.sum().item())}", flush=True)
yv, mv = S.obs_to_video(ov, om, ofid, oi, val.num_fields, val.num_points, Nz, Ny, Nx, Hp, Wp)
obs_frac = float(int(om.sum())) / float(val.num_points * 2)
alpha_upstream = 0.5 / math.sqrt(obs_frac)
print(f"[iso] observed_fraction={obs_frac:.5f} -> upstream alpha rule gives "
      f"{alpha_upstream:.3f}", flush=True)

names = [str(n) for n in val.field_names]
t_np = truth[0].cpu().numpy()

# Sweep 1 showed the consistency term ALONE diverges (a=0, b=0.4 -> std 758).
# Holding b=0.4 while sweeping alpha therefore cannot isolate the DPS term, so
# the DPS arms are re-run at b=0 and beta gets its own sweep.
ARMS_V1 = [
    ("unconditional   (a=0,     b=0)  ", 0.0, 0.0),
    ("consistency     (a=0,     b=0.4)", 0.0, 0.4),
    ("dps a=0.005     (b=0.4)         ", 0.005, 0.4),
    ("dps a=0.05      (b=0.4)         ", 0.05, 0.4),
    ("dps a=0.5       (b=0.4)         ", 0.5, 0.4),
    ("dps a=5.0 UPSTR (b=0.4)         ", 5.0, 0.4),
]
ARMS_V2 = [
    ("unconditional   (a=0,     b=0)  ", 0.0, 0.0),
    # DPS alone, consistency OFF -- isolates upstream's alpha rule
    ("dps  a=0.005    (b=0)           ", 0.005, 0.0),
    ("dps  a=0.05     (b=0)           ", 0.05, 0.0),
    ("dps  a=0.5      (b=0)           ", 0.5, 0.0),
    ("dps  a=1.0      (b=0)           ", 1.0, 0.0),
    ("dps  a=5.0 UPSTREAM RULE (b=0)  ", 5.0, 0.0),
    # consistency alone -- isolates the window-overlap term
    ("cons a=0        (b=0.004)       ", 0.0, 0.004),
    ("cons a=0        (b=0.04)        ", 0.0, 0.04),
    ("cons a=0        (b=0.4) UPSTREAM", 0.0, 0.4),
    # both at their upstream values
    ("BOTH a=5.0      (b=0.4) UPSTREAM", 5.0, 0.4),
]
# V3: re-confirmation on the FINAL checkpoint. The stability boundary was
# located on a 15,000-step model; it may move once the score is converged, and
# the candidate pair (alpha=0.5, beta=0.004) has never been run TOGETHER.
ARMS_V3 = [
    ("reference     (a=0,   b=0)     ", 0.0, 0.0),
    ("CANDIDATE     (a=0.5, b=0.004) ", 0.5, 0.004),
    ("             (a=0.5, b=0)      ", 0.5, 0.0),
    ("conservative  (a=0.05,b=0.004) ", 0.05, 0.004),
    ("beta up       (a=0.5, b=0.04)  ", 0.5, 0.04),
    ("above edge    (a=1.0, b=0.004) ", 1.0, 0.004),
    ("UPSTREAM ROW  (a=5.0, b=0.4)   ", 5.0, 0.4),
]
_sel = __import__("os").environ.get("ISO_ARMS", "v3")
ARMS = {"v1": ARMS_V1, "v2": ARMS_V2, "v3": ARMS_V3}[_sel]
print(f"[iso] arm set = {_sel}", flush=True)
rows = []
with ad.evaluation_weights(b):
    b.model.eval()
    for label, al, be in ARMS:
        torch.manual_seed(1234)
        t0 = time.perf_counter()
        with torch.enable_grad():
            vol = S.s3gm_reconstruct(b.model, s3gm_sde.VESDE(config=None,
                        sigma_min=c["sigma_min"], sigma_max=c["sigma_max"], N=a.n_steps),
                        yv, mv, Nz, K=1, t_win=c["t_win"], overlap=c["overlap"],
                        alpha=al, beta=be, snr=float(c["sampling_cfg"]["snr"]),
                        n_corrector_steps=0, device=dev, progress=False)
        wall = time.perf_counter() - t0
        pc = S.video_to_pointcloud(vol, Nz, Ny, Nx)[0].float().cpu().numpy()
        obs_rmse = float(np.sqrt((((vol - yv) * mv) ** 2).sum().item() / max(float(mv.sum()), 1.0)))
        rl2 = {n: float(np.linalg.norm(pc[:, i] - t_np[:, i]) / (np.linalg.norm(t_np[:, i]) + 1e-12))
               for i, n in enumerate(names)}
        r = {"arm": label.strip(), "alpha": al, "beta": be,
             "sample_std": float(pc.std()), "sample_max_abs": float(np.abs(pc).max()),
             "obs_rmse_z": obs_rmse, "rel_l2": rl2,
             "rel_l2_agg": float(np.mean(list(rl2.values()))), "wall_s": wall}
        rows.append(r)
        print(f"[arm] {label} std={r['sample_std']:.4g} max_abs={r['sample_max_abs']:.4g} "
              f"obs_rmse_z={obs_rmse:.4g} relL2_agg={r['rel_l2_agg']:.4g} "
              f"({', '.join(f'{n}={v:.3g}' for n, v in rl2.items())}) {wall:.0f}s", flush=True)
        del vol
        torch.cuda.empty_cache()

out = Path(a.out) if a.out else run_dir / "isolation_sweep.json"
out.write_text(json.dumps({
    "purpose": __doc__, "checkpoint_epoch": int(ck["epoch"]),
    "train_loss": float(ck["train_loss"]), "val_loss": float(ck["val_loss"]),
    "opt_steps": ck.get("total_opt_steps"), "snapshot": snap,
    "n_steps": a.n_steps, "observed_fraction": obs_frac,
    "alpha_upstream_rule": alpha_upstream,
    "reference": {"sane_sample_std": 1.0, "independent_obs_rmse_z": math.sqrt(2.0)},
    "arms": rows}, indent=2))
print(f"[iso] wrote {out}", flush=True)
