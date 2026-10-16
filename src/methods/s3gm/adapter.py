# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import argparse
import contextlib
import json
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Iterator
from typing import Optional
from typing import Tuple
from methods.common.helpers_baseline import _build_structured_triangulation
from methods.common.helpers_baseline import _save_single_field_plot
from methods.common.helpers_baseline import build_obs_grid_mask
from methods.common.helpers_baseline import build_sparse_condition
from methods.common.helpers_baseline import compute_pad_size
from methods.common.helpers_baseline import gather_from_grid
from methods.common.helpers_baseline import grid_to_pointcloud
from methods.common.helpers_baseline import midplane_slice
from methods.common.helpers_baseline import pointcloud_to_grid_padded
from methods.common.helpers_baseline import splat_obs_to_grid
from methods.common.helpers_baseline import validate_regular_grid_compatibility
from methods.common.adapters import BaseBaselineAdapter, BaselineBundle, _checkpoint_model_state
from methods.common.config import resolve_stage_config
from methods.common.deterministic import _attach_senconsis_outputs
from methods.s3gm.ema import ExponentialMovingAverage
from methods.s3gm.sde import VESDE
from methods.s3gm.unet_video import UNetVideoModel


def _net_to_score(net, out, std):
    """Map the network output to the score for the run's parametrisation.

    "raw" (upstream S3GM, the default for every checkpoint trained before
    2026-09-29): the network output IS the score, so its magnitude must scale
    like 1/sigma across the whole ladder and the training gradient at noise
    level sigma is scaled by sigma -- fine for sigma_max=20, unstable for the
    hundreds a 65k-240k-dimensional field needs (Song & Ermon rule).
    "eps" (score_sde's NCSN++ convention for the VE-SDE): the network predicts
    the negative noise and the score is output/sigma, so every noise level
    trains at the same gradient scale. Selected per run by
    s3gm_params.diffusion.score_param and stamped on the net at build time;
    old run configs lack the key and resolve to "raw".
    """
    if getattr(net, "frb_score_param", "raw") == "eps":
        std = std.view(-1, *([1] * (out.dim() - 1)))
        return out / std
    return out


def s3gm_loss(net, sde, x_grid, eps: float = 1e-5):
    batch_size = x_grid.shape[0]
    x = x_grid.unsqueeze(1)

    t = torch.rand(batch_size, device=x.device) * (sde.T - eps) + eps
    noise = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, t)
    std_e = std[:, None, None, None, None]
    perturbed = mean + std_e * noise

    obs_mask = torch.zeros(batch_size, 1, 1, 1, 1, device=x.device)
    latent_mask = torch.ones(batch_size, 1, 1, 1, 1, device=x.device)
    frame_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)

    out, _ = net(
        perturbed,
        x0=x,
        timesteps=std,
        frame_indices=frame_indices,
        obs_mask=obs_mask,
        latent_mask=latent_mask,
    )
    score = _net_to_score(net, out, std)
    losses = torch.square(score * std_e + noise)
    losses = losses.reshape(batch_size, -1).mean(dim=-1)
    return losses.mean()


def run_epoch_s3gm(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
    net = bundle.model
    optimizer = bundle.optimizer if training else None
    sde = bundle.components["sde"]
    num_y = int(bundle.config["shared"]["data"]["num_y"])
    num_x = int(bundle.config["shared"]["data"]["num_x"])
    h_pad = int(bundle.components["H_pad"])
    w_pad = int(bundle.components["W_pad"])
    grid_order = bundle.components.get("grid_order")
    ema = bundle.ema
    all_params_fn = bundle.components["all_params_fn"]
    grad_clip = bundle.components["grad_clip"]
    spike_skip = bundle.components["spike_skip"]

    net.train(training)
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"S3GM Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    for batch in pbar:
        fields_full = batch["fields"].to(bundle.device)
        x_grid = pointcloud_to_grid_padded(fields_full, num_y, num_x, h_pad, w_pad, grid_order=grid_order)
        loss = s3gm_loss(net, sde, x_grid)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                all_params_fn(), max_norm=float("inf") if grad_clip is None else float(grad_clip))
            if torch.isfinite(loss) and torch.isfinite(grad_norm) and (not spike_skip or grad_norm <= 50.0):
                optimizer.step()
                if ema is not None:
                    ema.update(all_params_fn())
            else:
                optimizer.zero_grad(set_to_none=True)

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


def _score_fn_nograd(net, sde, x_5d, t_vec):
    B = x_5d.shape[0]
    obs_mask = torch.zeros(B, 1, 1, 1, 1, device=x_5d.device)
    latent_mask = torch.ones(B, 1, 1, 1, 1, device=x_5d.device)
    frame_indices = torch.zeros(B, 1, dtype=torch.long, device=x_5d.device)
    _, std = sde.marginal_prob(torch.zeros_like(x_5d), t_vec)
    with torch.no_grad():
        out, _ = net(
            x_5d,
            x0=x_5d,
            timesteps=std,
            frame_indices=frame_indices,
            obs_mask=obs_mask,
            latent_mask=latent_mask,
        )
    return _net_to_score(net, out, std)


def _score_fn_grad(net, sde, x_5d, t_vec):
    B = x_5d.shape[0]
    obs_mask = torch.zeros(B, 1, 1, 1, 1, device=x_5d.device)
    latent_mask = torch.ones(B, 1, 1, 1, 1, device=x_5d.device)
    frame_indices = torch.zeros(B, 1, dtype=torch.long, device=x_5d.device)
    _, std = sde.marginal_prob(torch.zeros_like(x_5d), t_vec)
    out, _ = net(
        x_5d,
        x0=x_5d,
        timesteps=std,
        frame_indices=frame_indices,
        obs_mask=obs_mask,
        latent_mask=latent_mask,
    )
    return _net_to_score(net, out, std)


def dps_sample(
    net,
    sde,
    shape_5d,
    obs_value_grid,
    obs_mask_grid,
    device,
    N_steps=200,
    snr=0.128,
    n_corrector_steps=1,
    alpha_obs=1.0,
    eps=1e-3,
):
    B = shape_5d[0]
    x = sde.prior_sampling(list(shape_5d)).to(device).float()
    timesteps = torch.linspace(sde.T, eps, N_steps + 1, device=device).float()
    obs_5d = obs_value_grid.unsqueeze(1)
    mask_5d = obs_mask_grid.unsqueeze(1)

    for i in tqdm(range(N_steps), desc="S3GM sampling", leave=False):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        vec_t = torch.ones(B, device=device).float() * t_cur

        sigma_cur = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t_cur
        sigma_nxt = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t_next
        step_std = torch.sqrt(torch.clamp(sigma_cur ** 2 - sigma_nxt ** 2, min=0.0))

        for _ in range(n_corrector_steps):
            grad = _score_fn_nograd(net, sde, x, vec_t)
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(B, -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(B, -1), dim=-1).mean()
            step_size = (snr * noise_norm / (grad_norm + 1e-12)) ** 2 * 2.0
            step_size = torch.clamp(step_size, max=1.0)
            x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise

        with torch.enable_grad():
            x_inp = x.detach().clone().requires_grad_(True)
            score = _score_fn_grad(net, sde, x_inp, vec_t)

            with torch.no_grad():
                score_clean = torch.nan_to_num(score.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                G_e = step_std.view(1, 1, 1, 1, 1)
                rev_f = -(G_e ** 2) * score_clean
                x_mean = x_inp.detach() - rev_f
                if i < N_steps - 1:
                    z = torch.randn_like(x_inp)
                    x_next = x_mean + G_e * z
                else:
                    x_next = x_mean

            std_e = sigma_cur.view(1, 1, 1, 1, 1)
            x0_hat = std_e ** 2 * score + x_inp
            residual = (obs_5d - x0_hat) * mask_5d
            dps_loss = alpha_obs * torch.sum(residual ** 2)
            if torch.isfinite(dps_loss) and dps_loss > 0:
                dx = torch.autograd.grad(dps_loss, x_inp)[0]
                dx = torch.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
                dx = torch.clamp(dx, min=-1e8, max=1e8)
            else:
                dx = torch.zeros_like(x_inp)

        x = (x_next - dx).detach()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x.squeeze(1)


def visualize_reconstruction_s3gm(
    net,
    sde,
    dataset,
    Num_x,
    Num_y,
    H_pad,
    W_pad,
    epoch,
    device,
    save_dir,
    cond_fields=(2,),
    n_obs=256,
    N_steps=200,
    snr=0.128,
    n_corrector_steps=1,
    alpha_obs=1.0,
    snapshot_index=0,
    file_tag=None,
    save_metrics_json=True,
    irregular=False,
    deformer=None,
    point_to_grid=None,
    save_obs_consistency_plots=False,
):
    if isinstance(cond_fields, int):
        cond_fields = [cond_fields]
    if isinstance(n_obs, int):
        n_obs = [n_obs] * len(cond_fields)

    n_fields = dataset.num_fields
    n_pts = dataset.num_points
    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs,
        n_obs_max=n_obs,
        valid_mask=valid_mask,
    )

    if irregular:
        coords_2d = coords[..., :2]
        if deformer is not None:
            latent_coords = deformer(coords_2d)
            latent_obs_coords = deformer(obs_coords[..., :2])
        else:
            latent_coords = coords_2d
            latent_obs_coords = obs_coords[..., :2]
        obs_value_grid, obs_mask_grid = splat_obs_to_grid(
            latent_obs_coords,
            obs_values,
            obs_mask,
            obs_field_ids,
            n_fields,
            Num_y,
            Num_x,
        )
        if H_pad > Num_y or W_pad > Num_x:
            obs_value_grid = F.pad(obs_value_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)
            obs_mask_grid = F.pad(obs_mask_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)
    else:
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            n_fields,
            n_pts,
            Num_y,
            Num_x,
            H_pad,
            W_pad,
            point_to_grid=point_to_grid,
        )

    with torch.enable_grad():
        recon_grid = dps_sample(
            net=net,
            sde=sde,
            shape_5d=(1, 1, n_fields, H_pad, W_pad),
            obs_value_grid=obs_value_grid,
            obs_mask_grid=obs_mask_grid,
            device=device,
            N_steps=N_steps,
            snr=snr,
            n_corrector_steps=n_corrector_steps,
            alpha_obs=alpha_obs,
        )

    if irregular:
        recon = gather_from_grid(recon_grid[:, :, :Num_y, :Num_x], latent_coords)
    else:
        recon = grid_to_pointcloud(recon_grid, Num_y, Num_x, point_to_grid=point_to_grid)

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)
    recon_phys = recon_phys[0].cpu().numpy()
    truth_phys = truth_phys[0].cpu().numpy()

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    coords_np = coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]

    # Volumetric data: plot the z-midplane slice rather than every z-level
    # projected onto one plane (see midplane_slice).
    is_3d, slice_mask, coords_xy_plot, triang = midplane_slice(coords_np)
    body_polygon = None
    if not is_3d and hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if hasattr(dataset, "airfoil_body_indices") and dataset.airfoil_body_indices is not None:
        body_polygon = coords_xy[dataset.airfoil_body_indices]

    metrics = {}
    field_names = dataset.field_names if len(dataset.field_names) == truth_phys.shape[1] else tuple(f"field_{i}" for i in range(truth_phys.shape[1]))
    for c, name in enumerate(field_names):
        sensor_coords = None
        field_sensor_mask = obs_field_ids_cpu == c
        if np.any(field_sensor_mask):
            sensor_indices = obs_indices_cpu[field_sensor_mask]
            if is_3d:
                on_slice = slice_mask[sensor_indices]
                sensor_coords = (coords_xy[sensor_indices[on_slice]]
                                 if np.any(on_slice) else None)
            else:
                sensor_coords = coords_xy[sensor_indices]
        metrics[name] = _save_single_field_plot(
            true_f=truth_phys[:, c][slice_mask],
            pred_f=recon_phys[:, c][slice_mask],
            metric_true_f=truth_phys[:, c],
            metric_pred_f=recon_phys[:, c],
            coords_xy=coords_xy_plot,
            sensor_coords=sensor_coords,
            field_name=name,
            epoch=epoch,
            save_dir=save_dir,
            file_prefix=file_tag,
            triang=triang,
            body_polygon=body_polygon,
        )

    _attach_senconsis_outputs(
        metrics=metrics,
        recon=recon,
        truth=truth,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_indices=obs_indices,
        obs_field_ids=obs_field_ids,
        coords=coords,
        coords_xy=coords_xy,
        field_names=field_names,
        save_dir=save_dir,
        save_obs_consistency_plots=save_obs_consistency_plots,
    )

    if save_metrics_json:
        prefix = file_tag or f"epoch_{epoch:04d}"
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "N_steps": int(N_steps),
            "snr": float(snr),
            "n_corrector_steps": int(n_corrector_steps),
            "alpha_obs": float(alpha_obs),
            "method": "S3GM",
            "metrics": metrics,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return metrics


class S3GMAdapter(BaseBaselineAdapter):
    name = "s3gm"

    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        stage_cfg = resolve_stage_config(cfg)
        arch = stage_cfg["architecture"]
        diffusion = stage_cfg["diffusion"]
        training = stage_cfg["training"]
        num_y = int(cfg["shared"]["data"]["num_y"])
        num_x = int(cfg["shared"]["data"]["num_x"])
        grid_info = validate_regular_grid_compatibility(train_set, num_x, num_y)
        if val_set is not train_set:
            validate_regular_grid_compatibility(val_set, num_x, num_y)
        h_pad, w_pad = compute_pad_size(num_y, num_x, 2 ** (len(arch["ch_mult"]) - 1))

        net = UNetVideoModel(
            in_channels=train_set.num_fields,
            model_channels=int(arch["nf"]),
            out_channels=train_set.num_fields,
            num_res_blocks=int(arch["num_res_blocks"]),
            attention_resolutions=tuple(arch["attn_resolutions"]),
            image_size=max(h_pad, w_pad),
            dropout=float(arch["dropout"]),
            channel_mult=tuple(arch["ch_mult"]),
            conv_resample=True,
            dims=2,
            num_heads=int(arch["num_heads"]),
            use_rpe_net=True,
            use_checkpoint=bool(arch.get("use_checkpoint", False)),
        ).to(device)
        net.frb_score_param = str(diffusion.get("score_param", "raw"))
        if net.frb_score_param not in ("raw", "eps"):
            raise ValueError(f"s3gm diffusion.score_param must be 'raw' or 'eps', got {net.frb_score_param!r}")

        def all_params_fn():
            return list(net.parameters())

        optimizer = torch.optim.Adam(
            all_params_fn(),
            lr=float(training["learning_rate"]),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=float(training["weight_decay"]),
        )
        # `optimizer: upstream` is S3GM's own recipe (Adam at a constant learning
        # rate, no clipping, no step skipping); the default (100-epoch warm-up +
        # cosine, clip 0.5, skip on grad norm > 50) reproduces the pre-audit runs.
        upstream_opt = str(training.get("optimizer", "benchmark")).lower() == "upstream"
        if upstream_opt:
            scheduler = None
        else:
            warmup_epochs = 100

            def lr_lambda(epoch_index: int) -> float:
                if epoch_index < warmup_epochs:
                    return epoch_index / max(1, warmup_epochs)
                progress = (epoch_index - warmup_epochs) / max(1, int(training["epochs"]) - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        ema = ExponentialMovingAverage(all_params_fn(), decay=float(diffusion["ema_rate"]))
        sde = VESDE(
            config=argparse.Namespace(sde="vesde"),
            sigma_min=float(diffusion["sigma_min"]),
            sigma_max=float(diffusion["sigma_max"]),
            N=int(diffusion["num_scales"]),
        )
        return BaselineBundle(
            baseline_model=self.name,
            training_stage=1,
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            run_dir=run_dir,
            config=cfg,
            dataset_train=train_set,
            dataset_val=val_set,
            components={
                "sde": sde,
                "H_pad": h_pad,
                "W_pad": w_pad,
                "all_params_fn": all_params_fn,
                "grad_clip": None if upstream_opt else 0.5,
                "spike_skip": not upstream_opt,
                "grid_order": grid_info["grid_order"],
                "point_to_grid": grid_info["point_to_grid"],
                "grid_info": grid_info,
            },
        )

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        bundle.model.load_state_dict(_checkpoint_model_state(checkpoint, "S3GM"))
        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.scheduler is not None and checkpoint.get("scheduler") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler"])
        if bundle.ema is not None and checkpoint.get("ema") is not None:
            bundle.ema.load_state_dict(checkpoint["ema"])
            bundle.ema.shadow_params = [param.to(bundle.device) for param in bundle.ema.shadow_params]

    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> Tuple[float, float, float]:
        if training and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(bundle.device)
        t0 = time.perf_counter()
        loss = run_epoch_s3gm(bundle, loader, training, epoch)
        epoch_time_s = time.perf_counter() - t0
        peak_mem_mb = torch.cuda.max_memory_allocated(bundle.device) / 1024 ** 2 if torch.cuda.is_available() else 0.0
        return loss, epoch_time_s, peak_mem_mb

    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        return {
            "baseline_model": bundle.baseline_model,
            "training_stage": bundle.training_stage,
            "model": bundle.model.state_dict(),
            "ema": bundle.ema.state_dict() if bundle.ema is not None else None,
            "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
            "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "method": "S3GM_VESDE",
            "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
            "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
        }

    @contextlib.contextmanager
    def evaluation_weights(self, bundle: BaselineBundle) -> Iterator[None]:
        if bundle.ema is None:
            yield
            return
        all_params = bundle.components["all_params_fn"]()
        bundle.ema.store(all_params)
        bundle.ema.copy_to(all_params)
        try:
            yield
        finally:
            bundle.ema.restore(all_params)

    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None, save_obs_consistency_plots: bool = False) -> dict[str, float]:
        stage_cfg = resolve_stage_config(bundle.config)
        shared_cond = bundle.config["shared"]["conditioning"]
        sampling_cfg = stage_cfg["sampling"]
        n_steps = int(sampling_cfg["sampling_N"] if n_steps is None else n_steps)
        return visualize_reconstruction_s3gm(
            net=bundle.model,
            sde=bundle.components["sde"],
            dataset=dataset,
            Num_x=int(bundle.config["shared"]["data"]["num_x"]),
            Num_y=int(bundle.config["shared"]["data"]["num_y"]),
            H_pad=int(bundle.components["H_pad"]),
            W_pad=int(bundle.components["W_pad"]),
            epoch=epoch,
            device=bundle.device,
            save_dir=str(save_dir),
            cond_fields=shared_cond["vis_cond_fields"],
            n_obs=shared_cond["vis_n_obs_list"],
            N_steps=n_steps,
            snr=float(sampling_cfg["snr"]),
            n_corrector_steps=int(sampling_cfg["n_corrector_steps"]),
            alpha_obs=float(sampling_cfg["alpha_obs"]),
            snapshot_index=snapshot_index,
            file_tag=f"s3gm_N{n_steps}",
            save_metrics_json=True,
            point_to_grid=bundle.components.get("point_to_grid"),
            save_obs_consistency_plots=save_obs_consistency_plots,
        )
