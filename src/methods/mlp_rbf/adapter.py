# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from methods.common.helpers_baseline import build_sparse_condition
from methods.common.helpers_wing_baseline import build_wing_condition
from methods.common.helpers_wing_baseline import describe_condition as describe_wing_condition
from methods.common.helpers_wing_baseline import surface_pool_enabled
from methods.common.helpers_wing_baseline import wing_cond_channels
from methods.common.adapters import BaseBaselineAdapter, BaselineBundle, _checkpoint_model_state
from methods.common.config import _require_no_surface_pool, resolve_stage_config
from methods.common.deterministic import visualize_reconstruction_deterministic
from methods.mlp_rbf.model import ConditionalPointMLPRBF, DeterministicMLPRBFRegressor


def run_epoch_mlp_rbf(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
    model = bundle.model
    optimizer = bundle.optimizer if training else None
    shared_cond = bundle.config["shared"]["conditioning"]
    stage_cfg = resolve_stage_config(bundle.config)
    n_query_points = int(stage_cfg["training"].get("n_query_points", 4096))

    _surface_pool = surface_pool_enabled(bundle.config)

    model.train(training)
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"MLP-RBF Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    for batch in pbar:
        coords = batch["coords"].to(bundle.device)
        fields = batch["fields"].to(bundle.device)
        batch_size, n_pts, _ = coords.shape
        valid_mask = batch.get("valid_sensor_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(bundle.device)

        if _surface_pool:
            obs_coords, obs_values, obs_mask, _, obs_field_ids = build_wing_condition(
                batch, bundle.device,
                n_obs_min=shared_cond["n_obs_min_list"],
                n_obs_max=shared_cond["n_obs_max_list"],
            )
        else:
            obs_coords, obs_values, obs_mask, _, obs_field_ids = build_sparse_condition(
                coords_full=coords,
                fields_full=fields,
                cond_fields=shared_cond["cond_fields"],
                n_obs_min=shared_cond["n_obs_min_list"],
                n_obs_max=shared_cond["n_obs_max_list"],
                valid_mask=valid_mask,
            )
        if training and epoch == 1 and count == 0:
            print(f"[cond] mlp_rbf source="
                  f"{'surface_pool' if _surface_pool else 'volume'} "
                  f"{describe_wing_condition(obs_mask[:1], obs_field_ids[:1])}",
                  flush=True)

        if 0 < n_query_points < n_pts:
            idx = torch.randperm(n_pts, device=bundle.device)[:n_query_points]
            query_coords = coords[:, idx]
            query_fields = fields[:, idx]
        else:
            query_coords = coords
            query_fields = fields

        pred = model(query_coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        loss = F.mse_loss(pred, query_fields)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss * batch_size
        count += batch_size
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


class MLPRBFAdapter(BaseBaselineAdapter):
    name = "mlp_rbf"

    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        stage_cfg = resolve_stage_config(cfg)
        arch = stage_cfg["architecture"]
        training = stage_cfg["training"]
        backbone = ConditionalPointMLPRBF(
            n_fields=train_set.num_fields,
            coord_dim=int(arch["coord_dim"]),
            hidden_dim=int(arch["hidden_dim"]),
            cond_dim=int(arch["cond_dim"]),
            field_embed_dim=int(arch["field_embed_dim"]),
            rbf_sigma=float(arch["rbf_sigma"]),
            use_fourier_pe=bool(arch.get("use_fourier_pe", False)),
            fourier_pe_num_bands=int(arch.get("fourier_pe_num_bands", 32)),
            fourier_pe_max_freq=float(arch.get("fourier_pe_max_freq", 64.0)),
            # None everywhere except SHIFT-WING (9 sensor field ids vs 4
            # output fields); see Senseiver.__init__.
            n_obs_field_types=(wing_cond_channels(train_set)
                               if surface_pool_enabled(cfg) else None),
        )
        model = DeterministicMLPRBFRegressor(backbone).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[mlp_rbf] trainable_params={n_params}", flush=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training["epochs"]),
        )
        return BaselineBundle(
            baseline_model=self.name,
            training_stage=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            device=device,
            run_dir=run_dir,
            config=cfg,
            dataset_train=train_set,
            dataset_val=val_set,
        )

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        bundle.model.load_state_dict(_checkpoint_model_state(checkpoint, "MLP-RBF"))
        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.scheduler is not None and checkpoint.get("scheduler") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler"])

    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
        return run_epoch_mlp_rbf(bundle, loader, training, epoch)

    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        stage_cfg = resolve_stage_config(bundle.config)
        arch = stage_cfg["architecture"]
        return {
            "baseline_model": bundle.baseline_model,
            "training_stage": bundle.training_stage,
            "model": bundle.model.state_dict(),
            "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
            "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "mean": bundle.dataset_train.mean,
            "std": bundle.dataset_train.std,
            "field_names": bundle.dataset_train.field_names,
            "method": "MLP_RBF_supervised",
            "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
            "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
            "n_fields": int(bundle.dataset_train.num_fields),
            "coord_dim": int(arch["coord_dim"]),
            "hidden_dim": int(arch["hidden_dim"]),
            "cond_dim": int(arch["cond_dim"]),
            "field_embed_dim": int(arch["field_embed_dim"]),
            "rbf_sigma": float(arch["rbf_sigma"]),
            "use_fourier_pe": bool(arch.get("use_fourier_pe", False)),
            "fourier_pe_num_bands": int(arch.get("fourier_pe_num_bands", 32)),
            "fourier_pe_max_freq": float(arch.get("fourier_pe_max_freq", 64.0)),
            "dataset": bundle.config["shared"]["data"]["dataset_name"],
        }

    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None, save_obs_consistency_plots: bool = False) -> dict[str, float]:
        _require_no_surface_pool(bundle.config, "mlp_rbf")
        return visualize_reconstruction_deterministic(
            bundle=bundle,
            dataset=dataset,
            save_dir=save_dir,
            epoch=epoch,
            snapshot_index=snapshot_index,
            file_tag="mlp_rbf",
            save_obs_consistency_plots=save_obs_consistency_plots,
        )
