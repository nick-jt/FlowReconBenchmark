# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import copy
import numpy as np
import pickle
import torch
import yaml
from pathlib import Path
from typing import Optional
from typing import Sequence
from methods.common.helpers_wing_baseline import surface_pool_enabled
from paths import REPO_ROOT


SUPPORTED_BASELINES = {"s3gm", "latent_fm", "sit", "senseiver", "geofno", "mlp_rbf", "confild"}

# ---------------------------------------------------------------------------
# SHIFT-WING (surface-to-volume) support.
#
# The wing is an unstructured 400k-node CFD mesh with grid_shape=None and a
# dedicated 65,536-point body-surface observation pool. Only coordinate/point
# native baselines can consume it; everything that rasterises onto a Cartesian
# array (latent_fm, s3gm, geofno, SiT with the `patch` tokenizer) must not be
# pointed at it. Conditioning for these three is built by
# helpers_wing_baseline, which draws from the SURFACE POOL exactly as our
# model does -- see that module's docstring for why the shared volume draw
# would invert the comparison.
# ---------------------------------------------------------------------------
_WING_POINT_NATIVE_BASELINES = {"senseiver", "mlp_rbf", "sit"}


def _require_no_surface_pool(cfg: dict, who: str) -> None:
    """Block the in-training visualize/benchmark paths on SHIFT-WING.

    Those helpers build their own conditioning with build_sparse_condition
    (VOLUME sensors) and rasterise onto a num_x x num_y grid, neither of which
    is meaningful here. The wing configs set save_every: 1000000 so they are
    never reached; this raises if someone lowers it, rather than producing a
    figure and an in-loop metric that quietly answer a different, easier
    inverse problem. Scoring goes through eval_wing_ensemble.py instead.
    """
    if surface_pool_enabled(cfg):
        raise NotImplementedError(
            f"{who}: the in-training visualize path conditions on random "
            "VOLUME points and needs a Cartesian grid; it is not valid on "
            "SHIFT-WING. Keep save_every large and score with "
            "eval_wing_ensemble.py.")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except (pickle.UnpicklingError, RuntimeError):
        return torch.load(path, map_location=map_location, weights_only=False)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_device(device_override: Optional[str], device_ids: Sequence[int]) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(device_ids[0])}")
    return torch.device("cpu")


def ensure_absolute(path_value: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def validate_and_normalize_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    baseline_model = str(cfg.get("baseline_model", "")).strip().lower()
    if baseline_model not in SUPPORTED_BASELINES:
        raise ValueError(
            f"Unknown baseline_model={baseline_model!r}. "
            f"Expected one of {sorted(SUPPORTED_BASELINES)}."
        )
    cfg["baseline_model"] = baseline_model

    try:
        training_stage = int(cfg.get("training_stage"))
    except Exception as exc:
        raise ValueError("training_stage must be explicitly set to 1 or 2.") from exc
    if training_stage not in (1, 2):
        raise ValueError("training_stage must be 1 or 2.")
    cfg["training_stage"] = training_stage

    if baseline_model in {"s3gm", "sit", "senseiver", "geofno", "mlp_rbf"} and training_stage != 1:
        raise ValueError(
            f"baseline_model={baseline_model!r} only supports training_stage=1."
        )

    shared = cfg.setdefault("shared", {})
    shared.setdefault("demo_num", 0)
    shared.setdefault("seed", 42)
    shared.setdefault("device_ids", [0])
    shared.setdefault("reload", False)
    shared.setdefault("paths", {})
    shared["paths"].setdefault("data_path", "data/Merged_CH4COTU1P.h5")
    shared["paths"].setdefault("save_root", "outputs/runs")
    shared["paths"].setdefault("config_backup_root", "configs/gen_baseline")
    shared.setdefault("data", {})
    shared["data"].setdefault("dataset_name", "turbulent_combustion")
    shared["data"].setdefault("train_ratio", 0.9)
    shared["data"].setdefault("time_stride", 1)
    shared["data"].setdefault("num_workers", 4)
    shared["data"].setdefault("num_x", 403)
    shared["data"].setdefault("num_y", 100)
    shared["data"].setdefault("num_z", None)
    shared["data"].setdefault("field_names", None)
    shared.setdefault("conditioning", {})
    shared["conditioning"].setdefault("cond_fields", [2, 3])
    shared["conditioning"].setdefault("n_obs_min_list", [128, 128])
    shared["conditioning"].setdefault("n_obs_max_list", [512, 512])
    # sensor_pool: None (canonical: sensors anywhere) | "surface" (cylinder
    # surface-to-field task: sensors restricted to the H5's surface_indices).
    shared["conditioning"].setdefault("sensor_pool", None)
    if shared["conditioning"].get("vis_cond_fields") is None:
        shared["conditioning"]["vis_cond_fields"] = list(shared["conditioning"]["cond_fields"])
    if shared["conditioning"].get("vis_n_obs_list") is None:
        shared["conditioning"]["vis_n_obs_list"] = list(shared["conditioning"]["n_obs_max_list"])
    shared.setdefault("logging", {})
    shared["logging"].setdefault("eval_every", 5)
    shared["logging"].setdefault("save_every", 200)
    shared.setdefault("overrides", {})
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay"):
        shared["overrides"].setdefault(key, None)

    # Datasets stored in the canonical H5 layout (coordinates/fields/time) all
    # flow through TurbulentCombustionH5Dataset, which is dimension-agnostic;
    # the name gate only guards against configs for datasets with no H5 export.
    # kolmogorov2d / cylinder2d are the PoF-benchmark 2D regimes (convert_
    # kolmogorov.py / convert_cylinder.py); their configs must set field_names,
    # num_x/num_y and leave num_z unset so the 2D adapter paths engage.
    _supported_h5 = {"turbulent_combustion", "kolmogorov2d", "cylinder2d"}
    # SHIFT-WING is per-case unstructured HDF5, not the canonical single-file
    # H5 layout, so it is exempt from the name gate: build_dataset routes
    # `data.dataset: shiftwing` to ShiftWingBaselineDataset before any of the
    # H5 machinery runs. Without this exemption the gate fires inside
    # validate_and_normalize_config -- which every driver calls BEFORE
    # build_dataset -- and no wing baseline can start at all.
    _is_shiftwing = str(shared["data"].get("dataset", "")).lower() == "shiftwing"
    if shared["data"]["dataset_name"] not in _supported_h5 and not _is_shiftwing:
        raise NotImplementedError(
            f"dataset_name {shared['data']['dataset_name']!r} has no canonical "
            f"H5 branch here (supported: {sorted(_supported_h5)}; shiftwing "
            "routes through build_dataset's dedicated branch)."
        )
    if _is_shiftwing:
        # Surface-pool conditioning is not optional on this dataset: the whole
        # point of the wing benchmark is that observations live ONLY on the
        # body surface. Sampling sensors from random volume points (what
        # helpers_baseline.build_sparse_condition does) would hand a baseline
        # interior information our model never receives and invert the
        # comparison, so refuse to run rather than produce a number that looks
        # valid and is not.
        if str(shared["conditioning"].get("source", "")).lower() != "surface_pool":
            raise ValueError(
                "dataset: shiftwing requires shared.conditioning.source: "
                "surface_pool. Any other setting draws sensors from RANDOM "
                "VOLUME POINTS, which leaks interior state the surface-to-"
                "volume protocol never provides. Refusing to train."
            )
        if str(cfg["baseline_model"]) not in _WING_POINT_NATIVE_BASELINES:
            raise NotImplementedError(
                f"baseline_model={cfg['baseline_model']!r} is not point-native; "
                f"SHIFT-WING is an unstructured mesh with grid_shape=None. "
                f"Supported here: {sorted(_WING_POINT_NATIVE_BASELINES)}."
            )
    return cfg


def resolve_stage_config(cfg: dict) -> dict:
    cfg = validate_and_normalize_config(cfg)
    baseline = cfg["baseline_model"]
    stage = cfg["training_stage"]
    shared = cfg["shared"]

    if baseline in {"latent_fm", "confild"}:
        stage_cfg = copy.deepcopy(cfg[f"{baseline}_params"][f"stage{stage}"])
    else:
        stage_cfg = copy.deepcopy(cfg[f"{baseline}_params"])

    training = stage_cfg.setdefault("training", {})
    training.setdefault("eval_every", shared["logging"]["eval_every"])
    training.setdefault("save_every", shared["logging"]["save_every"])

    overrides = shared["overrides"]
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay"):
        if overrides.get(key) is not None:
            training[key] = overrides[key]

    return stage_cfg


def build_run_name(cfg: dict, timestamp: str) -> str:
    return (
        f"Baseline_{cfg['baseline_model']}_Stage{cfg['training_stage']}"
        f"_DemoN{int(cfg['shared']['demo_num'])}_{timestamp}"
    )


def list_run_dirs(save_root: Path, cfg: dict) -> list[Path]:
    prefix = (
        f"Baseline_{cfg['baseline_model']}_Stage{cfg['training_stage']}"
        f"_DemoN{int(cfg['shared']['demo_num'])}_"
    )
    if not save_root.exists():
        return []
    return sorted([path for path in save_root.glob(f"{prefix}*") if path.is_dir()])


def find_latest_run_dir(save_root: Path, cfg: dict) -> Optional[Path]:
    run_dirs = list_run_dirs(save_root, cfg)
    return run_dirs[-1] if run_dirs else None


def copy_config_backup(config_path: Path, backup_root: Path, run_name: str) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    target = backup_root / f"{run_name}.yaml"
    target.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return target
