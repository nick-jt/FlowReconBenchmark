"""Parameter count of a training configuration, by the benchmark convention.

    python -m tools.count_params --config configs/kolmogorov2d/config_baseline_Det_kolm.yaml
    python -m tools.count_params --config configs/cylinder2d/config_bench_cyl_ffm_b128.yaml

Instantiates the model exactly as its trainer would (CPU, no checkpoint) and counts
`numel` over the parameters, a complex element counting once. Exit status 1 when the
count is outside the learned-row budget (6.5 M +- 10 %) unless --no-gate; a published
configuration outside the budget is reported as a separately labelled row.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from paths import REPO_ROOT
from protocol import PARAM_BUDGET, PARAM_TOL, count_parameters, within_budget


def _count_baseline(cfg: dict, stage: int | None, model: str | None) -> dict:
    from methods.common import config as C
    from methods.common.data import build_dataset
    from methods.common.registry import get_baseline_adapter
    if model:
        cfg["baseline_model"] = model
    if stage is not None:
        cfg["training_stage"] = stage
    cfg = C.validate_and_normalize_config(cfg)
    data_path = C.ensure_absolute(cfg["shared"]["paths"]["data_path"])
    ds = build_dataset(cfg, split="train", stats_path=data_path.with_suffix(".stats.pt"))
    adapter = get_baseline_adapter(cfg["baseline_model"])
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        s2 = cfg.get("latent_fm_params", {}).get("stage2", {})
        if cfg["baseline_model"] == "latent_fm" and cfg["training_stage"] == 2 and not s2.get("stage1_checkpoint"):
            # No trained autoencoder yet: build stage 1 from the config and point
            # stage 2 at a throwaway checkpoint of it, so the count is the
            # architecture's (the AE weights do not matter for counting).
            c1 = dict(cfg, training_stage=1)
            b1 = adapter.build_for_training(cfg=c1, device=device, run_dir=Path(tmp), train_set=ds, val_set=ds)
            ck = Path(tmp) / "stage1.pt"
            torch.save(adapter.build_checkpoint(b1, 0, 0.0, 0.0), ck)
            s2["stage1_checkpoint"] = str(ck)
        bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=Path(tmp), train_set=ds, val_set=ds)
        out = count_parameters(bundle.model)
    out["model"] = f"{cfg['baseline_model']} stage {cfg['training_stage']}"
    if cfg["baseline_model"] == "latent_fm" and cfg["training_stage"] == 2:
        out["note"] = "stage-2 count includes the frozen stage-1 autoencoder"
    return out


def _count_dmfgen(yaml_cfg: dict, config_path: Path) -> dict:
    from methods.dmfgen.evaluate import _build_model, _normalize_eval_config
    from methods.dmfgen.helpers import TurbulentCombustionH5Dataset
    from methods.dmfgen.train import normalize_conditioning_args, parse_args
    argv, sys.argv = sys.argv, [sys.argv[0], "--config", str(config_path)]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    for k, v in yaml_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    cfg = _normalize_eval_config(vars(normalize_conditioning_args(args)))
    data = Path(cfg["data"])
    if not data.is_absolute():
        data = REPO_ROOT / data
    ds = TurbulentCombustionH5Dataset(str(data), split="train", train_ratio=cfg["train_ratio"],
                                      field_names=cfg["field_names"], seed=cfg["seed"],
                                      time_stride=cfg["time_stride"],
                                      stats_path=str(data.with_suffix(".stats.pt")),
                                      sensor_pool=cfg.get("sensor_pool"))
    out = count_parameters(_build_model(cfg, ds))
    out["model"] = f"dmfgen/{cfg['backbone']}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline-model", default=None)
    ap.add_argument("--training-stage", type=int, default=None)
    ap.add_argument("--no-gate", action="store_true", help="report only; never exit 1")
    a = ap.parse_args()
    path = Path(a.config)
    if not path.is_absolute():
        path = REPO_ROOT / path
    cfg = yaml.safe_load(open(path)) or {}
    if "baseline_model" in cfg:
        out = _count_baseline(cfg, a.training_stage, a.baseline_model)
    else:
        out = _count_dmfgen(cfg, path)
    out["config"] = str(path.relative_to(REPO_ROOT)) if REPO_ROOT in path.parents else str(path)
    out["budget"] = PARAM_BUDGET
    out["within_budget"] = within_budget(out["total"])
    print(json.dumps(out))
    print(f"[count_params] {out['model']}: {out['total']:,} parameters ({out['convention']}); "
          f"budget {PARAM_BUDGET / 1e6:.1f}M +- {PARAM_TOL:.0%}: "
          f"{'OK' if out['within_budget'] else 'OUTSIDE'}", flush=True)
    if not out["within_budget"] and not a.no_gate:
        sys.exit(1)


if __name__ == "__main__":
    main()
