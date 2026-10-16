# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import copy
import torch
from pathlib import Path
from typing import Optional
from methods.common.adapters import BaseBaselineAdapter
from methods.common.config import ensure_absolute, find_latest_run_dir


class CoNFiLDAdapter(BaseBaselineAdapter):
    """Route CoNFiLD through its upstream two-timescale, stateful lifecycle."""

    name = "confild"
    uses_custom_training_loop = True

    def _stage1_checkpoint_path(self, cfg: dict) -> Path:
        explicit = cfg["confild_params"]["stage2"].get("stage1_checkpoint")
        if explicit:
            checkpoint_path = ensure_absolute(explicit)
            if not checkpoint_path.exists():
                raise RuntimeError(f"Configured CoNFiLD stage-1 checkpoint does not exist: {checkpoint_path}")
            return checkpoint_path
        stage1_cfg = copy.deepcopy(cfg)
        stage1_cfg["training_stage"] = 1
        latest = find_latest_run_dir(ensure_absolute(cfg["shared"]["paths"]["save_root"]), stage1_cfg)
        if latest is None:
            raise RuntimeError("CoNFiLD stage 2 requires a completed unified stage-1 run.")
        checkpoint_path = latest / "best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = latest / "last.pt"
        if not checkpoint_path.exists():
            raise RuntimeError(f"No stage-1 checkpoint found in {latest}")
        return checkpoint_path

    def run_custom_training(self, cfg: dict, device: torch.device, run_dir: Path, reload_ckpt: Optional[dict]) -> dict:
        try:
            from methods.confild.upstream_training import run_confild_training
        except ImportError:
            from .confild_upstream_training import run_confild_training
        stage1_checkpoint = self._stage1_checkpoint_path(cfg) if int(cfg["training_stage"]) == 2 else None
        return run_confild_training(
            cfg=cfg,
            device=device,
            run_dir=run_dir,
            resume=reload_ckpt,
            stage1_checkpoint=stage1_checkpoint,
        )

    # CoNFiLD's latent table and once-per-epoch decoder update cannot be
    # represented by the generic stateless minibatch adapter interface.
    def build_for_training(self, cfg, device, run_dir, train_set, val_set):
        raise RuntimeError("CoNFiLD must use run_custom_training().")

    def load_checkpoint(self, bundle, checkpoint):
        raise RuntimeError("CoNFiLD must use run_custom_training().")

    def run_epoch(self, bundle, loader, training, epoch):
        raise RuntimeError("CoNFiLD must use run_custom_training().")

    def build_checkpoint(self, bundle, epoch, train_loss, val_loss):
        raise RuntimeError("CoNFiLD must use run_custom_training().")

    def visualize(self, bundle, dataset, save_dir, epoch, snapshot_index, n_steps=None, save_obs_consistency_plots=False):
        raise RuntimeError("Use the dedicated frozen CoNFiLD evaluator; training never optimizes held-out latents.")
