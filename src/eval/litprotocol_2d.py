#!/usr/bin/env python3
"""Seen-trajectory and sensor-lattice probes on 2D Kolmogorov flow.

Every probed method is scored with its canonical TEST row's run, checkpoint, NFE,
weights and K, read from the payload that fleet_select.load_canonical_fleet
returns, so the probe measures the reported configuration and its result can be
set against that row's TEST score over the same number of frames.

Arms (655 sensors per frame unless the lattice replaces them):

  seen     50 frames of TRAINING trajectories from the stride-1 raw npy, raw frame
           4k+2: between two stride-4 frames the models trained on, evenly spaced
           over the (train trajectory, k) candidates. Truth is z-scored with the
           run's train stats. Sensors: torch.manual_seed(seed*777 + p) before the
           canonical helpers.build_sparse_condition draw on CUDA, p = candidate
           index (these frames have no split index). IDW is the control for how
           hard these frames are compared with the TEST frames.
  lattice  the 50 TEST frames of the canonical rows (same frames, same ensemble
           noise seeds), with sensors on a fixed centred rows x cols lattice
           instead of the seeded scatter. The nearest-sensor distance (grid
           cells) of both layouts on these frames is recorded as well.
           DMF-Gen-CQ runs this arm only: its upstream run holds only the stored
           frames, not the raw frames of the seen arm.

Ensemble noise follows fleet_2d: base = seed*131 + si, one batched call seeded at
base for DMF-Gen-CQ, torch.manual_seed(base*10000 + k) before member k otherwise.
Deterministic rows are scored as two identical members with dispersion nulled.
IDW (k=8, power 1) runs with non-periodic and with periodic neighbour search, as
the TEST anchors classical_baselines_main_1pct_{nonperiodic,periodic}_2d.json do.

Outputs: <out-dir>/kolm_<arm>_<model>.json per method and kolm_<arm>_summary.json.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from eval.fleet_2d import load_baseline, load_dmfgen_cq, null_dispersion, require_block_split, summarize
from eval.fleet_select import load_canonical_fleet
from paths import DATA_DIR, RUNS_DIR, site_path
from protocol import PROTOCOL_2D, frame_list

MANIFEST = DATA_DIR / "kolmogorov2d" / "kolmogorov2d_manifest.json"
CLASSICAL = RUNS_DIR / "kolmogorov2d" / "baseline_classical"
DET = {"senseiver", "mlp_rbf", "geofno"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Kolmogorov seen-trajectory / sensor-lattice probes")
    p.add_argument("--arms", nargs="+", default=["seen", "lattice"], choices=["seen", "lattice"])
    p.add_argument("--models", nargs="+", default=["sit", "senseiver", "geofno", "latent_fm"],
                   choices=["dmfgen_cq", "sit", "senseiver", "geofno", "latent_fm", "mlp_rbf"],
                   help="Canonical Kolmogorov rows to probe; the IDW control always runs.")
    p.add_argument("--n-obs", type=int, default=655)
    p.add_argument("--n-frames", type=int, default=50)
    p.add_argument("--idw-k", type=int, default=8)
    p.add_argument("--lattice", type=int, nargs=2, default=[25, 26],
                   help="Lattice rows cols; 25x26 = 650 sensors against the 655 scattered.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--out-dir", default=str(RUNS_DIR / "kolmogorov2d" / "litprotocol"))
    p.add_argument("--dry-run", action="store_true", help="Print the plan; no GPU, no model.")
    return p.parse_args()


def seen_frames(manifest: dict, n_frames: int) -> list[dict]:
    """(p, train trajectory, raw frame 4k+2) picks, evenly spaced over the candidates."""
    train, fpt, stride = list(manifest["train_trajs"]), int(manifest["frames_per_traj"]), int(manifest["stride"])
    out = []
    for j in range(n_frames):
        p = (j * len(train) * fpt) // n_frames
        out.append({"p": p, "traj": int(train[p // fpt]), "k": p % fpt,
                    "raw_frame": stride * (p % fpt) + stride // 2})
    return out


def uniform_lattice(ny: int, nx: int, rows: int, cols: int) -> np.ndarray:
    """Row-major flat indices of a centred rows x cols lattice."""
    iy = ((2 * np.arange(rows) + 1) * ny) // (2 * rows)
    ix = ((2 * np.arange(cols) + 1) * nx) // (2 * cols)
    return (iy[:, None] * nx + ix[None, :]).ravel().astype(np.int64)


def scattered(seed_value: int, n_pts: int, n_obs: int, device: str) -> torch.Tensor:
    """Sensor indices of the canonical draw under torch.manual_seed(seed_value). The draw
    depends only on the seed, the point count and the device, so zeros stand in for the
    coordinates and fields."""
    from methods.dmfgen.helpers import build_sparse_condition
    torch.manual_seed(seed_value)
    z = torch.zeros(1, n_pts, 1, device=device)
    return build_sparse_condition(coords_full=z, fields_full=z, cond_fields=[0],
                                  n_obs_min=[n_obs], n_obs_max=[n_obs])[3][0]


def observe(coords: torch.Tensor, truth: torch.Tensor, idx: torch.Tensor) -> dict:
    """build_sparse_condition's output layout for one observed field at indices idx."""
    m = idx.numel()
    return {"coords": coords[:, idx], "values": truth[:, idx, 0:1],
            "mask": torch.ones(1, m, device=idx.device), "indices": idx[None],
            "field_ids": torch.zeros(1, m, dtype=torch.long, device=idx.device)}


def nearest_sensor_cells(idx: np.ndarray, ny: int, nx: int, periodic: bool) -> tuple[float, float]:
    """Mean and max over grid cells of the distance, in cells, to the nearest sensor."""
    cells = np.stack(np.divmod(np.arange(ny * nx), nx), axis=1).astype(np.float64)
    tree = cKDTree(cells[idx], boxsize=(ny, nx) if periodic else None)
    d, _ = tree.query(cells, k=1)
    return float(d.mean()), float(d.max())


def load_sampler(name: str, row: dict, device: str, chunk: int):
    """(TEST dataset, sampler, weights context) of one canonical row. sampler(coords
    [1, N, 3], obs, base, snap, truth [1, N, C]) returns the members [K, N, C]
    (deterministic: two identical); only DMF-Gen-CQ reads snap and truth."""
    run_dir, ckpt, nfe, K = Path(row["_run_dir"]), str(row["ckpt"]), int(row["nfe"]), int(row["K"])
    if name == "dmfgen_cq":
        up, ds, _, batch, members = load_dmfgen_cq(run_dir, ckpt, device, nfe, "test", K,
                                                   row["weights"] == "raw")

        def sample(coords, obs, base, snap, truth):
            return members(up.sample(batch(snap, truth, obs["values"], obs["indices"],
                                           obs["field_ids"], obs["mask"]), base))
        return ds, sample, contextlib.nullcontext()

    import methods.common.config as bconfig
    import methods.common.helpers_baseline as HB
    import methods.sit.adapter as sit_adapter
    adapter, bundle, ds, cfg, device, _ = load_baseline(run_dir, ckpt, name, "test")
    bundle.model.eval()
    comp = bundle.components
    ny, nx = int(cfg["shared"]["data"]["num_y"]), int(cfg["shared"]["data"]["num_x"])
    nf, npts, p2g = ds.num_fields, ds.num_points, comp.get("point_to_grid")
    solver = None if name in DET else str(bconfig.resolve_stage_config(cfg)["sampling"]["ode_solver"])
    if name == "sit" and str(comp["tokenizer"]) != "pointnet":
        raise SystemExit("[guard] the canonical SiT row is SiT-point; this run has a patch tokenizer")
    if name == "geofno" and str(comp["variant"]) == "deform":
        raise SystemExit("[guard] the Kolmogorov FNO row is the grid variant; this run is the mesh variant")

    def member(coords, o):
        oc, ov, om, oi, ofid = (o[k] for k in ("coords", "values", "mask", "indices", "field_ids"))
        if name in ("senseiver", "mlp_rbf"):
            return bundle.model(coords, oc, ov, om, ofid)
        if name == "geofno":
            gv, gm = HB.build_obs_grid_mask(ov, om, ofid, oi, nf, npts, ny, nx, ny, nx, point_to_grid=p2g)
            return HB.grid_to_pointcloud(bundle.model(gv, gm), ny, nx, point_to_grid=p2g)
        if name == "latent_fm":
            gv, gm = HB.build_obs_grid_mask(ov, om, ofid, oi, nf, npts, ny, nx, ny, nx)
            grid = bundle.model.sample({"obs_value_grid": gv, "obs_mask_grid": gm}, n_steps=nfe, ode_solver=solver)
            return HB.grid_to_pointcloud(grid, ny, nx)
        return sit_adapter.sit_conditional_sample_points_chunked(
            net=bundle.model, transport=comp["transport"], coords=coords, obs_coords=oc, obs_values=ov,
            obs_mask=om, obs_field_ids=ofid, n_fields=nf, device=device, n_steps=nfe, sampler_type=solver,
            chunk=int(comp.get("node_subsample") or 0) or int(npts), sigma=float(comp.get("cond_fill_sigma", 0.05)),
            rechunk=True)

    @torch.no_grad()
    def sample(coords, obs, base, snap=None, truth=None):
        if name in DET:
            return np.repeat(member(coords, obs)[0].float().cpu().numpy()[None], 2, axis=0)
        ens = []
        for k in range(K):
            torch.manual_seed(base * 10_000 + k)
            ens.append(member(coords, obs)[0].float().cpu().numpy())
        return np.stack(ens)

    weights = contextlib.nullcontext() if row["weights"] == "raw" else adapter.evaluation_weights(bundle)
    return ds, sample, weights


def main() -> None:
    args = parse_args()
    require_block_split()
    if "dmfgen_cq" in args.models and "seen" in args.arms:
        raise SystemExit("[guard] dmfgen_cq runs the lattice arm only")
    out_dir = Path(args.out_dir)
    manifest = json.load(open(MANIFEST))
    ny, nx = (int(v) for v in manifest["grid"])
    lattice_np = uniform_lattice(ny, nx, *args.lattice)
    canon = load_canonical_fleet("kolmogorov2d", RUNS_DIR)
    rows = {m: canon[m.replace("_", "")] for m in args.models}
    idw_test = {per: json.load(open(CLASSICAL / f"classical_baselines_main_1pct_{per}_2d.json"))
                ["results"][f"idw_n{args.n_obs}"]["aggregate"] for per in ("nonperiodic", "periodic")}
    npy = site_path("kolmogorov_shu_npy")
    seen = seen_frames(manifest, args.n_frames)
    print(f"[plan] arms={args.arms} n_obs={args.n_obs} n_frames={args.n_frames} "
          f"lattice={args.lattice[0]}x{args.lattice[1]}={lattice_np.size} npy={npy} out={out_dir}")
    for m, r in rows.items():
        print(f"[plan] {m}: {Path(r['_run_dir']).name} ckpt={r['ckpt']} nfe={r['nfe']} weights={r['weights']} "
              f"K={r['K']} TEST relL2={r['summary']['aggregate']['rel_l2_mean']:.4f}")
    if args.dry_run:
        print("[dry-run] seen frames: " + " ".join(f"t{f['traj']}r{f['raw_frame']}" for f in seen))
        assert all(f["raw_frame"] % 4 == 2 and f["raw_frame"] < 4 * int(manifest["frames_per_traj"]) for f in seen)
        print(f"[dry-run] npy exists={npy.exists()}; OK")
        return

    from methods.classical.anchors_2d import build_coords_box, kd_predict, read_layout
    from methods.dmfgen.ensemble_eval import ensemble_metrics, require_compute_node
    require_compute_node()
    device = "cuda:0"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(npy, mmap_mode="r")
    lattice = torch.from_numpy(lattice_np).to(device)
    m_mean = float(manifest["train_stats"]["vorticity"]["mean"])
    m_std = float(manifest["train_stats"]["vorticity"]["std"])

    def arm_frames(arm, ds):
        """[(record, truth [N, C] z-scored with the run's train stats, sensor indices)]."""
        mean, std = float(ds.mean.ravel()[0]), float(ds.std.ravel()[0])
        if abs(m_std - std) > 1e-3 * m_std or abs(m_mean - mean) > 1e-3 * m_std:
            raise SystemExit("[guard] the run's train stats disagree with the manifest's")
        if arm == "seen":
            return [(dict(f), ((np.asarray(raw[f["traj"], f["raw_frame"]], np.float32).reshape(-1, 1) - mean)
                               / std).astype(np.float32),
                     scattered(args.seed * 777 + f["p"], ds.num_points, args.n_obs, device)) for f in seen]
        return [({"snapshot": int(s)}, ds[int(s)]["fields"].numpy(), lattice)
                for s in frame_list(len(ds), args.n_frames)]

    def write(arm, name, setting, per, field_names, test_agg):
        summ = summarize(per, field_names)
        agg = summ["aggregate"]
        pay = {"protocol": f"{PROTOCOL_2D}_{'seen_trajectory' if arm == 'seen' else 'lattice'}",
               "arm": arm, "model": name, **setting, "n_obs": [int(per[0]["sensors"])], "seed": args.seed,
               "n_frames": len(per), "lattice": args.lattice if arm == "lattice" else None,
               "npy": str(npy) if arm == "seen" else None, "canonical_test": test_agg,
               "summary": summ, "snapshots": per}
        (out_dir / f"kolm_{arm}_{name}.json").write_text(json.dumps(pay, indent=1))
        print(f"[out] kolm_{arm}_{name}.json relL2={agg['rel_l2_mean']:.4f} crps={agg['crps']:.4f} "
              f"(TEST {test_agg['rel_l2_mean']:.4f} / {test_agg['crps']:.4f})", flush=True)
        return agg

    results = {arm: {} for arm in args.arms}
    ref = {}                      # arm -> frames of the first model, reused by the IDW control
    for name in args.models:
        row = rows[name]
        ds, sample, weights = load_sampler(name, row, device, args.chunk)
        field_names = list(ds.field_names)
        coords = ds[0]["coords"].unsqueeze(0).to(device)
        setting = {"run_dir": row["_run_dir"], "ckpt": row["ckpt"], "nfe": row["nfe"],
                   "weights": row["weights"], "K": row["K"]}
        with weights:
            for arm in args.arms:
                frames = arm_frames(arm, ds)
                ref.setdefault(arm, frames)
                per = []
                for si, (rec, truth, idx) in enumerate(frames):
                    t = torch.from_numpy(truth).unsqueeze(0).to(device)
                    ens = sample(coords, observe(coords, t, idx), args.seed * 131 + si, rec.get("snapshot"), t)
                    m = ensemble_metrics(ens, truth, field_names)
                    per.append({**rec, "si": si, "sensors": int(idx.numel()), "idx_sum": int(idx.sum()),
                                **(null_dispersion(m) if name in DET else m)})
                results[arm][name] = (write(arm, name, setting, per, field_names, row["summary"]["aggregate"]),
                                      row["summary"]["aggregate"])
        del sample, ds
        torch.cuda.empty_cache()

    craw = read_layout(str(site_path("kolmogorov2d_h5")))[0]
    for per_tag, periodic in (("nonperiodic", False), ("periodic", True)):
        box, boxsize = build_coords_box(craw, periodic=periodic)
        for arm in args.arms:
            per = []
            for si, (rec, truth, idx) in enumerate(ref[arm]):
                i = idx.cpu().numpy()
                pred = kd_predict(box, {0: (i, truth[i, 0])}, n_ch=1, mode="idw", k=args.idw_k, boxsize=boxsize)
                m = null_dispersion(ensemble_metrics(np.repeat(pred[None], 2, axis=0), truth, ["vorticity"]))
                per.append({**rec, "si": si, "sensors": int(i.size), "idx_sum": int(i.sum()), **m})
            name = f"idw_{per_tag}"
            results[arm][name] = (write(arm, name, {"idw_k": args.idw_k, "periodic": periodic}, per,
                                        ["vorticity"], idw_test[per_tag]), idw_test[per_tag])

    for arm in args.arms:
        table = {}
        for name, (agg, test) in results[arm].items():
            table[name] = {"test_rel_l2": test["rel_l2_mean"], "arm_rel_l2": agg["rel_l2_mean"],
                           "test_crps": test["crps"], "arm_crps": agg["crps"],
                           "rel_gain": 1.0 - agg["rel_l2_mean"] / test["rel_l2_mean"]}
        for name in table:
            table[name]["rel_gain_net_of_idw"] = table[name]["rel_gain"] - table["idw_nonperiodic"]["rel_gain"]
        summary = {"protocol": PROTOCOL_2D, "arm": arm, "n_frames": args.n_frames, "n_obs": args.n_obs,
                   "gain_note": "rel_gain = 1 - arm rel-L2 / canonical TEST rel-L2; net of the non-periodic "
                                "IDW control's rel_gain", "rows": table}
        if arm == "lattice":
            test_snaps = [r["snapshot"] for r, _, _ in ref[arm]]
            draws = [scattered(args.seed * 777 + s, ny * nx, args.n_obs, device).cpu().numpy() for s in test_snaps]
            canon_sums = [s["idx_sum"] for s in rows[args.models[0]]["snapshots"]]
            summary["scatter_matches_canonical_test_draw"] = [int(d.sum()) for d in draws] == canon_sums
            summary["nearest_sensor_cells"] = {}
            for per_tag, periodic in (("nonperiodic", False), ("periodic", True)):
                sc = np.array([nearest_sensor_cells(d, ny, nx, periodic) for d in draws])
                la = nearest_sensor_cells(lattice_np, ny, nx, periodic)
                summary["nearest_sensor_cells"][per_tag] = {
                    "scattered_mean": float(sc[:, 0].mean()), "scattered_max_frame_mean": float(sc[:, 1].mean()),
                    "scattered_max_overall": float(sc[:, 1].max()), "lattice_mean": la[0], "lattice_max": la[1]}
        (out_dir / f"kolm_{arm}_summary.json").write_text(json.dumps(summary, indent=1))
        print(f"\n[{arm}] {'row':<18}{'TEST':>8}{arm:>9}{'gain':>8}{'net':>8}")
        for name, r in table.items():
            print(f"[{arm}] {name:<18}{r['test_rel_l2']:>8.4f}{r['arm_rel_l2']:>9.4f}"
                  f"{r['rel_gain']:>8.3f}{r['rel_gain_net_of_idw']:>8.3f}")
        if arm == "lattice":
            print(f"[lattice] draw matches canonical TEST: {summary['scatter_matches_canonical_test_draw']}; "
                  f"nearest-sensor cells {json.dumps(summary['nearest_sensor_cells'])}")


if __name__ == "__main__":
    main()
