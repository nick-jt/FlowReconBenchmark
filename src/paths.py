"""Repository-relative locations used by every script in FlowReconBench.

Nothing here is dataset- or cluster-specific: the repository root is found from
this file, and the conventional directories hang off it.  Cluster-local things
(dataset files, upstream checkouts, SLURM accounts) live in ``configs/site.yaml``
and are read through :func:`site`.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]      # src/paths.py -> repo root
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"                        # trained runs (was Save_TrainedModel/)
CONFIGS_DIR = REPO_ROOT / "configs"                    # training / eval configs (was Save_config/)
PAPER_DIR = OUTPUTS_DIR / "paper"                     # tables, figures and JSONs behind the paper
THIRD_PARTY_DIR = REPO_ROOT / "third_party"            # upstream baseline repositories (git submodules)


def resolve(path_value: str | os.PathLike, base: Path = REPO_ROOT) -> Path:
    """Absolute path for a config value: absolute values pass through, relative ones
    are taken from the repository root (never from the current working directory)."""
    p = Path(path_value)
    return p if p.is_absolute() else (base / p).resolve()


@functools.lru_cache(maxsize=None)
def site() -> dict:
    """``configs/site.yaml`` as a dict (empty if absent). Environment variables of the
    form ``FRB_<KEY>`` override the file, e.g. ``FRB_CONFILD_ROOT=/path``."""
    import yaml
    cfg = {}
    f = CONFIGS_DIR / "site.yaml"
    if f.is_file():
        cfg = yaml.safe_load(open(f)) or {}
    for k, v in os.environ.items():
        if k.startswith("FRB_"):
            cfg[k[4:].lower()] = v
    return cfg


def site_path(key: str, default: str | None = None) -> Path:
    """A path-valued site setting, resolved against the repository root."""
    v = site().get(key, default)
    if v is None:
        raise KeyError(f"site setting '{key}' is not set (configs/site.yaml or FRB_{key.upper()})")
    return resolve(v)
