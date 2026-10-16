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
from methods.common.helpers_baseline import build_obs_grid_mask
from methods.common.helpers_baseline import build_sparse_condition
from methods.common.helpers_baseline import pointcloud_to_grid_padded
from methods.common.helpers_baseline import validate_regular_grid_compatibility
from methods.common.adapters import BaseBaselineAdapter, BaselineBundle, _checkpoint_model_state
from methods.common.config import resolve_stage_config
from methods.common.deterministic import visualize_reconstruction_deterministic
from methods.geofno.model import FNOSupervisedGrid, FNOSupervisedIrregular
from methods.geofno.model_deform import GeoFNOSupervisedMesh


def predict_geofno_mesh(model, coords, obs_coords, obs_values, obs_mask, obs_field_ids) -> torch.Tensor:
    """Forward of the `deform` variant (Geo-FNO on the native mesh): sensors in at their
    own coordinates, prediction out at every mesh point. One mesh per dataset, so the
    batch shares a single query point cloud."""
    mesh = coords[:1] if bool((coords == coords[:1]).all()) else coords
    return model(mesh[..., :2], obs_coords[..., :2], obs_values, obs_mask, obs_field_ids)


def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The FNO reference's LpLoss (p = 2, averaged over the batch): per-sample
    ||pred - target|| / ||target|| over every point and field."""
    b = pred.shape[0]
    num = (pred - target).reshape(b, -1).norm(dim=1)
    den = target.reshape(b, -1).norm(dim=1).clamp_min(1e-12)
    return (num / den).mean()


def run_epoch_geofno(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
    model = bundle.model
    optimizer = bundle.optimizer if training else None
    shared_cond = bundle.config["shared"]["conditioning"]
    training_cfg = resolve_stage_config(bundle.config)["training"]
    # `loss: rel_l2` and `grad_clip: null` are the reference recipe; the defaults
    # (MSE, clip 1.0) reproduce the pre-audit runs.
    loss_fn = rel_l2_loss if str(training_cfg.get("loss", "mse")).lower() == "rel_l2" else F.mse_loss
    grad_clip = training_cfg.get("grad_clip", 1.0)
    num_y = int(bundle.config["shared"]["data"]["num_y"])
    num_x = int(bundle.config["shared"]["data"]["num_x"])
    variant = str(bundle.components["variant"])
    point_to_grid = bundle.components.get("point_to_grid")
    grid_order = bundle.components.get("grid_order")

    model.train(training)
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"Geo-FNO Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    for batch in pbar:
        coords = batch["coords"].to(bundle.device)
        fields = batch["fields"].to(bundle.device)
        valid_mask = batch.get("valid_sensor_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(bundle.device)

        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=fields,
            cond_fields=shared_cond["cond_fields"],
            n_obs_min=shared_cond["n_obs_min_list"],
            n_obs_max=shared_cond["n_obs_max_list"],
            valid_mask=valid_mask,
        )

        if variant == "irregular":
            pred = model(coords[..., :2], obs_values, obs_mask, obs_field_ids, obs_indices)
            loss = loss_fn(pred, fields)
        elif variant == "deform":
            pred = predict_geofno_mesh(model, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
            loss = loss_fn(pred, fields)
        else:
            n_fields = fields.shape[2]
            n_pts = fields.shape[1]
            obs_value_grid, obs_mask_grid = build_obs_grid_mask(
                obs_values,
                obs_mask,
                obs_field_ids,
                obs_indices,
                n_fields,
                n_pts,
                num_y,
                num_x,
                num_y,
                num_x,
                point_to_grid=point_to_grid,
            )
            target_grid = pointcloud_to_grid_padded(
                fields,
                num_y,
                num_x,
                num_y,
                num_x,
                grid_order=grid_order,
            )
            pred_grid = model(obs_value_grid, obs_mask_grid)
            loss = loss_fn(pred_grid, target_grid)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


class GeoFNOAdapter(BaseBaselineAdapter):
    name = "geofno"

    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        stage_cfg = resolve_stage_config(cfg)
        arch = stage_cfg["architecture"]
        training = stage_cfg["training"]
        variant = str(arch.get("geofno_variant", "fno")).lower()
        num_y = int(cfg["shared"]["data"]["num_y"])
        num_x = int(cfg["shared"]["data"]["num_x"])

        grid_info = None
        if variant == "fno":
            grid_info = validate_regular_grid_compatibility(train_set, num_x, num_y)
            if val_set is not train_set:
                validate_regular_grid_compatibility(val_set, num_x, num_y)
            model = FNOSupervisedGrid(
                n_fields=train_set.num_fields,
                Num_x=num_x,
                Num_y=num_y,
                n_modes_x=int(arch["fno_modes_x"]),
                n_modes_y=int(arch["fno_modes_y"]),
                hidden_channels=int(arch["fno_hidden_channels"]),
                n_layers=int(arch["fno_n_layers"]),
            ).to(device)
        elif variant == "irregular":
            model = FNOSupervisedIrregular(
                n_fields=train_set.num_fields,
                latent_Nx=int(arch["latent_Nx"]),
                latent_Ny=int(arch["latent_Ny"]),
                n_modes_x=int(arch["fno_modes_x"]),
                n_modes_y=int(arch["fno_modes_y"]),
                hidden_channels=int(arch["fno_hidden_channels"]),
                n_layers=int(arch["fno_n_layers"]),
            ).to(device)
        elif variant == "deform":
            # Geo-FNO proper on the native mesh. The deformation's angle/radius features
            # are taken about `iphi_center_xy` (physical coordinates, e.g. the body centre).
            raw_xy = train_set.coords_raw[:, :2]
            lo, hi = raw_xy.min(dim=0).values, raw_xy.max(dim=0).values
            extent = (hi - lo).clamp_min(1e-8)
            center_xy = torch.tensor([float(v) for v in arch.get("iphi_center_xy", ((lo + hi) / 2).tolist())])
            model = GeoFNOSupervisedMesh(
                n_fields=train_set.num_fields,
                latent_Nx=int(arch["latent_Nx"]),
                latent_Ny=int(arch["latent_Ny"]),
                n_modes_x=int(arch["fno_modes_x"]),
                n_modes_y=int(arch["fno_modes_y"]),
                hidden_channels=int(arch["fno_hidden_channels"]),
                n_layers=int(arch["fno_n_layers"]),
                iphi_width=int(arch.get("iphi_width", 32)),
                center=((center_xy - lo) / extent).tolist(),
                aspect=(extent / extent.max()).tolist(),
                domain_padding=float(arch.get("domain_padding", 0.1)),
            ).to(device)
        else:
            raise ValueError("geofno_params.architecture.geofno_variant must be 'fno', 'irregular' or 'deform'.")

        if variant == "deform":
            # The reference trains the deformation with its own, smaller learning rate
            # (1e-4 against 1e-3 for the operator).
            lr = float(training["learning_rate"])
            params = [
                {"params": model.fno.parameters(), "lr": lr},
                {"params": model.iphi.parameters(), "lr": lr * float(training.get("iphi_lr_ratio", 0.1))},
            ]
        else:
            params = model.parameters()
        # `optimizer: upstream` is the reference recipe (Adam with the stated weight
        # decay, StepLR halving every `lr_step_epochs`); the default (AdamW + cosine)
        # reproduces the pre-audit runs.
        if str(training.get("optimizer", "adamw")).lower() == "upstream":
            optimizer = torch.optim.Adam(params, lr=float(training["learning_rate"]),
                                         weight_decay=float(training["weight_decay"]))
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=int(training["lr_step_epochs"]),
                gamma=float(training.get("lr_step_gamma", 0.5)))
        else:
            optimizer = torch.optim.AdamW(
                params,
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
            components={
                "variant": variant,
                "grid_info": grid_info,
                "grid_order": None if grid_info is None else grid_info["grid_order"],
                "point_to_grid": None if grid_info is None else grid_info["point_to_grid"],
            },
        )

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        bundle.model.load_state_dict(_checkpoint_model_state(checkpoint, "Geo-FNO"))
        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.scheduler is not None and checkpoint.get("scheduler") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler"])

    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
        return run_epoch_geofno(bundle, loader, training, epoch)

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
            "method": "GeoFNO_supervised",
            "geofno_variant": str(bundle.components["variant"]),
            "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
            "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
            "fno_modes_x": int(arch["fno_modes_x"]),
            "fno_modes_y": int(arch["fno_modes_y"]),
            "fno_hidden_channels": int(arch["fno_hidden_channels"]),
            "fno_n_layers": int(arch["fno_n_layers"]),
            "dataset": bundle.config["shared"]["data"]["dataset_name"],
        }

    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None, save_obs_consistency_plots: bool = False) -> dict[str, float]:
        return visualize_reconstruction_deterministic(
            bundle=bundle,
            dataset=dataset,
            save_dir=save_dir,
            epoch=epoch,
            snapshot_index=snapshot_index,
            file_tag="geofno_supervised",
            save_obs_consistency_plots=save_obs_consistency_plots,
        )
