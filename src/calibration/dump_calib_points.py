"""Route-2 dump: per-point ensemble data for conformal recalibration.

Generalizes the fleet-figure per-point dump to all 50 canonical snapshots at
2--3 sensor densities: for each snapshot it saves the K-member ensemble, the
truth, and distance-to-nearest-sensor (pooled and per observed channel) on a
fixed random query subset, as one compressed .npz per snapshot.

Seeding, sensor draws, and the canonical-fingerprint gate are inherited from
ensemble_eval.py (same torch.manual_seed(seed*777+snap) sensor path, same
check_canonical_fingerprint call), so the dumped conditioning is bit-identical
to the canonical evals.  Compute-node only (require_compute_node aborts on
login nodes).  ONE cheap eval job per density; see dump_calib_points.sh.

Output npz keys: ens [K,N,C] f16, truth [N,C] f16, dist [N] f32 (nearest valid
sensor, any channel), dist_ch [N,F] f32 (nearest valid sensor per observed
channel), query_idx [N] i64, plus scalars snap/K/n_steps/n_obs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from methods.dmfgen.ensemble_eval import build_sparse_condition, check_canonical_fingerprint, load_run, require_compute_node, sample_ensemble


def _min_dist(query: torch.Tensor, sensors: torch.Tensor, chunk: int = 65536) -> np.ndarray:
    """Min Euclidean distance from each query point to any sensor. [N] on cpu.

    cdist materializes a [chunk, n_sensors] matrix, so the caller's chunk is a
    memory bomb at high sensor density: at 1% of 1.95M points per channel the
    pair matrix is 262144 x 390624 x 4 B = 410 GB and the job dies on the first
    call. The chunk is therefore capped here by a ~2 GB budget on that matrix,
    which makes the function safe at every density instead of only the ones we
    happened to run first.
    """
    n_sensors = max(1, int(sensors.shape[0]))
    budget_elems = 512_000_000            # ~2 GB of float32
    safe = max(1024, budget_elems // n_sensors)
    chunk = int(min(chunk, safe))
    out = torch.empty(query.shape[0], dtype=torch.float32)
    for i in range(0, query.shape[0], chunk):
        d = torch.cdist(query[i:i + chunk], sensors)
        out[i:i + chunk] = d.min(dim=1).values.float().cpu()
        del d
    return out.numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0, 2])
    p.add_argument("--n-obs", type=int, nargs="+", required=True,
                   help="per-channel sensor count, e.g. --n-obs 19531 19531")
    p.add_argument("--query-subset", type=int, default=200_000)
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    require_compute_node()
    model, dataset, cfg = load_run(args.run_dir, args.ckpt, args.device)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    snap_ids = rng.choice(len(dataset), size=min(args.n_snapshots, len(dataset)),
                          replace=False)

    for si, snap in enumerate(snap_ids):
        # Resume: this dump is ~2 min/snapshot at the highest sensor density, so a
        # wall-clock timeout used to throw away every completed snapshot. Skipping
        # files already written makes the job restartable; the sensor draw is
        # seeded per snapshot, so a resumed run reproduces the same conditioning.
        _out = out_dir / f"calib_points_snap{int(snap):03d}.npz"   # must match the write below
        if _out.exists() and _out.stat().st_size > 0:
            print(f"[resume] snap {snap} -> {_out.name} already present", flush=True)
            continue
        item = dataset[int(snap)]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)

        torch.manual_seed(args.seed * 777 + int(snap))
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=args.cond_fields, n_obs_min=args.n_obs,
            n_obs_max=args.n_obs,
        )
        _sel = oi[om > 0]
        check_canonical_fingerprint(int(snap), int(_sel.numel()),
                                    int(_sel.sum().item()), args.seed,
                                    args.cond_fields, args.n_obs)
        obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
               "field_ids": ofid}

        sel = torch.from_numpy(
            rng.choice(coords.shape[1], size=min(args.query_subset, coords.shape[1]),
                       replace=False)).sort().values.to(device)
        coords_q = coords[:, sel]
        truth_q = fields[:, sel][0]

        valid = om[0] > 0
        sensor_xyz = oc[0][valid]
        dist = _min_dist(coords_q[0], sensor_xyz)
        dist_ch = np.stack(
            [_min_dist(coords_q[0], oc[0][valid & (ofid[0] == f)])
             for f in args.cond_fields], axis=1)

        ens = sample_ensemble(model, coords_q, obs, K=args.K,
                              n_steps=args.n_steps, chunk=args.chunk,
                              clamp_hard=False, seed=args.seed * 131 + si)

        out = out_dir / f"calib_points_snap{int(snap):03d}.npz"
        np.savez_compressed(
            out,
            ens=ens.numpy().astype(np.float16),
            truth=truth_q.cpu().numpy().astype(np.float16),
            dist=dist.astype(np.float32),
            dist_ch=dist_ch.astype(np.float32),
            query_idx=sel.cpu().numpy().astype(np.int64),
            snap=int(snap), K=args.K, n_steps=args.n_steps,
            n_obs=np.array(args.n_obs), cond_fields=np.array(args.cond_fields),
        )
        print(f"[dump] snap {int(snap)} -> {out} "
              f"(N={coords_q.shape[1]}, K={args.K})", flush=True)

    meta = {"run_dir": args.run_dir, "ckpt": args.ckpt, "K": args.K,
            "n_steps": args.n_steps, "n_obs": args.n_obs,
            "cond_fields": args.cond_fields, "seed": args.seed,
            "query_subset": args.query_subset,
            "field_names": list(dataset.field_names)}
    (out_dir / "dump_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[dump] wrote {out_dir}/dump_meta.json")


if __name__ == "__main__":
    main()
