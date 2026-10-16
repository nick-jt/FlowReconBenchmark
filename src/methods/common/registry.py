# Extracted from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py (get_baseline_adapter).
from __future__ import annotations

from methods.common.adapters import BaseBaselineAdapter
from methods.confild.adapter import CoNFiLDAdapter
from methods.geofno.adapter import GeoFNOAdapter
from methods.latent_fm.adapter import LatentFMAdapter
from methods.mlp_rbf.adapter import MLPRBFAdapter
from methods.s3gm.adapter import S3GMAdapter
from methods.senseiver.adapter import SenseiverAdapter
from methods.sit.adapter import SiTAdapter

_OVERRIDES: dict[str, type] = {}


def register_adapter(baseline_model: str, adapter_cls: type) -> None:
    """Replace the adapter class used for ``baseline_model`` (e.g. the 3-D S3GM
    adapter for the JHU cube). Call before the trainer/evaluator asks for it."""
    _OVERRIDES[str(baseline_model).strip().lower()] = adapter_cls


def get_baseline_adapter(baseline_model: str) -> BaseBaselineAdapter:
    baseline_model = str(baseline_model).strip().lower()
    registry: dict[str, type] = {
        "s3gm": S3GMAdapter,
        "latent_fm": LatentFMAdapter,
        "sit": SiTAdapter,
        "senseiver": SenseiverAdapter,
        "mlp_rbf": MLPRBFAdapter,
        "geofno": GeoFNOAdapter,
        "confild": CoNFiLDAdapter,
    }
    registry.update(_OVERRIDES)
    if baseline_model not in registry:
        raise ValueError(
            f"Unsupported baseline_model={baseline_model!r}. "
            f"Available: {sorted(registry)}"
        )
    return registry[baseline_model]()
