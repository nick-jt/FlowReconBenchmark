"""Pick each method's CANONICAL 2D fleet row by what the payload says it is.

The read-side twin of artifact_guard.py. Every script that turned a directory of
eval JSONs into paper numbers chose files by filename glob, last-file-wins. On
2026-09-10 that silently ingested operator runs (kolm_fleet_occl0.25_*) into
tab:kolm -- DMF-Gen read 0.660 (occluded) instead of 0.487 -- and only SiT and
S3GM survived, because their clean files happened to sort last. Density-override
files (kolm_fleet_ovr_n65_*) were being ingested the same way.

A row here is canonical only if its PAYLOAD says so: clean protocol stamp of the
current protocol, TEST split, no sensor overwrite, canonical sensor count,
canonical K (8 generative / 1 deterministic), exactly the canonical number of
per-snapshot records, and a run whose recorded config is a file in configs/.
Filenames are not consulted for meaning. If more than one file still qualifies
for a model, or the family declared to hold the reported row has none, that is
an error, not a tie to break by sort order or a fallback to another family.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import yaml

from paths import CONFIGS_DIR, REPO_ROOT
from protocol import PROTOCOL_2D

CANON = {
    "kolmogorov2d": {"prefix": "kolm_fleet", "n_obs": (655,), "n_frames": 50},
    "cylinder2d":   {"prefix": "cyl_fleet", "n_obs": (238,), "n_frames": 50},
}
PROTOCOL = PROTOCOL_2D          # the 2D evaluator's clean stamp
DETERMINISTIC = {"senseiver", "mlp_rbf", "mlprbf", "geofno"}

# The family directory (<ds>/<family>/<run>) that holds each model's reported row.
# Other families of the same model are labelled variants and never stand in for
# it: a missing reported family is an error. Entries may be per-dataset; None
# declares that the model has no reported row on that dataset.
#   dmfgen  the v1 configuration (no spectral penalty; the optimiser and sampling-time
#           settings shared with the JHU row), decided 2026-10-04; the upstream-recipe
#           runs (pointcloud_ffm) are a labelled variant. Kolmogorov reports dmfgencq
#           instead, decided 2026-10-11. The cylinder reports the v1 configuration at
#           batch 128 x 2380 queries (pointcloud_ffm_v1_b128), decided 2026-10-15
#   sit    point-tokenized SiT (the 3D row's configuration), decided 2026-09-28
#   geofno  Kolmogorov is a periodic grid (plain FNO); the cylinder runs the
#           mesh-native Geo-FNO (deform variant), decided 2026-10-02
#   dmfgencq  the upstream GL_rbf_CQ backbone trained by the upstream code
#           (eval.upstream_cq), Kolmogorov only, added 2026-10-11; trained on the
#           shared 65-6554 sensor range (upstream_gl_rbf_cq_range), decided 2026-10-15
# A per-dataset entry is looked up by the full dataset key first, then by its base key,
# so a seed replicate can declare that the reported configuration was not replicated.
REPORTED_FAMILY = {
    "dmfgen": {"kolmogorov2d": None, "cylinder2d": "pointcloud_ffm_v1_b128",
               "cylinder2d_seed7": None, "cylinder2d_seed1337": None},
    "dmfgencq": "upstream_gl_rbf_cq_range",
    "sit": "baseline_sit_point",
    "s3gm": "baseline_s3gm",
    "latentfm": "baseline_latent_fm",
    "senseiver": "baseline_det",
    "mlprbf": "baseline_mlp_rbf",
    "geofno": {"kolmogorov2d": "baseline_geofno", "cylinder2d": "baseline_geofno_mesh"},
}


def family_for(model: str, ds: str) -> str | None:
    """REPORTED_FAMILY entry of `model` on dataset key `ds` (full key, then base key)."""
    fam = REPORTED_FAMILY[model.replace("_", "")]
    if isinstance(fam, dict):
        fam = fam[ds] if ds in fam else fam[base_key(ds)]
    return fam


def reported_family(model: str, root: Path, ds: str) -> str:
    """Family directory holding the reported run of `model` on dataset key `ds`.
    Errors when no run with a best.pt exists there (no fallback)."""
    fam = family_for(model, ds)
    if fam is None:
        raise SystemExit(f"[fleet_select] {ds}/{model}: no reported row on this dataset")
    if not glob.glob(str(Path(root) / ds / fam / "*" / "best.pt")):
        raise SystemExit(f"[fleet_select] {ds}/{model}: reported family {fam} has no run with best.pt")
    return fam


def sit_family(root: Path, ds: str) -> str:
    return reported_family("sit", root, ds)


def _n_obs(v):
    if v is None:
        return None
    return tuple(sorted({int(x) for x in (v if isinstance(v, (list, tuple)) else [v])}))


def why_not_canonical(d: dict, ds: str) -> str | None:
    """None if canonical, else the reason it is not."""
    c = CANON[base_key(ds)]
    if d.get("protocol") != PROTOCOL:
        return f"protocol={d.get('protocol')}"
    if d.get("split") != "test":
        return f"split={d.get('split')}"
    if d.get("clamp_hard") is not False:
        return f"clamp_hard={d.get('clamp_hard')}"
    if _n_obs(d.get("n_obs")) != c["n_obs"]:
        return f"n_obs={d.get('n_obs')}"
    m = str(d.get("model", ""))
    want_k = 1 if m in DETERMINISTIC else 8
    if int(d.get("K", -1)) != want_k:
        return f"K={d.get('K')} (canonical {want_k})"
    if int(d.get("n_frames", -1)) != c["n_frames"]:
        return f"n_frames={d.get('n_frames')}"
    if len(d.get("snapshots") or []) != c["n_frames"]:
        return f"{len(d.get('snapshots') or [])} snapshot records (canonical {c['n_frames']})"
    return None


def base_key(ds: str) -> str:
    """kolmogorov2d_seed7 -> kolmogorov2d: a seed replicate is canonical on the
    same identity as its base fleet."""
    return ds.split("_seed")[0]


def _drop_volatile(cfg: dict) -> dict:
    """Keys a launcher legitimately sets per job: resume, and the stage of a
    two-stage config (the run name carries the stage)."""
    cfg = json.loads(json.dumps(cfg))
    cfg.get("shared", {}).pop("reload", None)
    cfg.pop("training_stage", None)
    return cfg


def verify_run_config(run_dir: Path) -> Path:
    """The run's recorded configuration must be a file in configs/ whose content
    the run actually used. Returns that file. Baseline runs record
    run_metadata.json/config_path and the normalised run_config.yaml; DMF-Gen runs
    record args.json with the config path and every resolved argument. Upstream
    PhyCoFlow-MultiFieldReconstruction runs record resolved_config.yaml, kept verbatim
    as configs/upstream/<experiment_name>.yaml."""
    run_dir = Path(run_dir)
    resolved = run_dir / "resolved_config.yaml"
    if resolved.exists():
        got = yaml.safe_load(open(resolved))
        cfg_path = CONFIGS_DIR / "upstream" / f"{got['output']['experiment_name']}.yaml"
        if not cfg_path.is_file() or yaml.safe_load(open(cfg_path)) != got:
            raise SystemExit(f"[fleet_select] {run_dir.name}: resolved_config.yaml differs from {cfg_path}")
        return cfg_path
    meta = run_dir / "run_metadata.json"
    if meta.exists():
        from methods.common.config import validate_and_normalize_config
        cfg_path = Path(json.load(open(meta))["config_path"])
        if not cfg_path.is_absolute():
            cfg_path = REPO_ROOT / cfg_path
        if CONFIGS_DIR not in cfg_path.resolve().parents or not cfg_path.is_file():
            raise SystemExit(f"[fleet_select] {run_dir.name}: config {cfg_path} is not a file under {CONFIGS_DIR}")
        want = _drop_volatile(validate_and_normalize_config(yaml.safe_load(open(cfg_path))))
        got = _drop_volatile(yaml.safe_load(open(run_dir / "run_config.yaml")))
        if want != got:
            raise SystemExit(f"[fleet_select] {run_dir.name}: run_config.yaml differs from {cfg_path}")
        return cfg_path
    args_json = run_dir / "args.json"
    if not args_json.exists():
        raise SystemExit(f"[fleet_select] {run_dir.name}: neither run_metadata.json nor args.json")
    args = json.load(open(args_json))
    cfg_path = REPO_ROOT / str(args.get("config", ""))
    if CONFIGS_DIR not in cfg_path.resolve().parents or not cfg_path.is_file():
        raise SystemExit(f"[fleet_select] {run_dir.name}: config {cfg_path} is not a file under {CONFIGS_DIR}")
    bad = {}
    for k, v in (yaml.safe_load(open(cfg_path)) or {}).items():
        got = args.get(k)
        if v in (None, ""):          # left empty in the file: the trainer fills the default
            continue
        if v != got and not (isinstance(got, list) and len(got) == 1 and got[0] == v):
            bad[k] = (v, got)
    if bad:
        raise SystemExit(f"[fleet_select] {run_dir.name}: args.json differs from {cfg_path} on {bad}")
    return cfg_path


def _scan(dirkey: str, canon_key: str, root: Path, verbose: bool) -> dict:
    c = CANON[canon_key]
    found: dict[str, list] = {}
    for p in sorted(glob.glob(str(root / dirkey / "*" / "*" / "Evaluation" / f"{c['prefix']}_*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or "summary" not in d:
            continue
        reason = why_not_canonical(d, canon_key)
        if reason:
            if verbose:
                print(f"  [skip] {Path(p).name}: {reason}")
            continue
        d["_family"] = Path(p).parents[2].name        # <ds>/<family>/<run>/Evaluation/<file>
        d["_run_dir"] = str(Path(p).parents[1])
        found.setdefault(str(d["model"]).replace("_", ""), []).append((p, d))
    for m, rows in list(found.items()):
        if m not in REPORTED_FAMILY:
            raise SystemExit(f"[fleet_select] {dirkey}/{m}: no reported family declared")
        fam = family_for(m, dirkey)
        if fam is None:
            if verbose:
                print(f"  [family] {m}: not reported on {dirkey}; rows ignored")
            del found[m]
            continue
        keep = [(p, d) for p, d in rows if d["_family"] == fam]
        dropped = [Path(p).name for p, d in rows if d["_family"] != fam]
        if dropped and verbose:
            print(f"  [family] {m}: reported family {fam}; variant rows ignored: {dropped}")
        if not keep:
            raise SystemExit(f"[fleet_select] {dirkey}/{m}: canonical rows exist only under "
                             f"{sorted({d['_family'] for _, d in rows})}, not the reported family {fam}")
        found[m] = keep
    dup = {m: [Path(p).name for p, _ in v] for m, v in found.items() if len(v) > 1}
    if dup:
        raise SystemExit(f"[fleet_select] more than one canonical row for {dup} -- "
                         "refusing to pick by sort order")
    out = {m: v[0][1] for m, v in found.items()}
    for d in out.values():
        d["_config"] = str(verify_run_config(Path(d["_run_dir"])))
    return out


def load_canonical_fleet(ds: str, root: Path, verbose: bool = False) -> dict:
    """{model: payload} for the canonical rows of one 2D dataset key."""
    return _scan(ds, base_key(ds), root, verbose)
