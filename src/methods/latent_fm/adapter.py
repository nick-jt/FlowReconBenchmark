# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import contextlib
import copy
import json
import matplotlib.pyplot as plt
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
from methods.common.helpers_baseline import build_obs_grid_mask3d
from methods.common.helpers_baseline import build_sparse_condition
from methods.common.helpers_baseline import gather_from_grid
from methods.common.helpers_baseline import grid3d_to_pointcloud
from methods.common.helpers_baseline import grid_to_pointcloud
from methods.common.helpers_baseline import pointcloud_to_grid
from methods.common.helpers_baseline import pointcloud_to_grid3d
from methods.common.helpers_baseline import splat_obs_to_grid
from methods.common.helpers_baseline import splat_to_grid
from methods.common.adapters import BaseBaselineAdapter, BaselineBundle, LatentEMA
from methods.common.config import ensure_absolute, find_latest_run_dir, resolve_stage_config, safe_torch_load
from methods.common.deterministic import _attach_senconsis_outputs
from methods.common.viz import _apply_eval_measurement_ops, _infer_structured_grid_from_coords, _reshape_flat_field_to_grid, _ssim2d
from methods.latent_fm.model import ConvAE, ConvAE3D, LatentFMUNet, LatentFMUNet3D, LatentFlowMatching, PointNetSensorEncoder


def run_epoch_ae(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> Tuple[float, float, float]:
    """Run one AE epoch. Returns (mean_loss, epoch_time_s, peak_gpu_mem_mb)."""
    ae = bundle.model
    optimizer = bundle.optimizer if training else None
    num_y = int(bundle.config["shared"]["data"]["num_y"])
    num_x = int(bundle.config["shared"]["data"]["num_x"])
    num_z = bundle.config["shared"]["data"].get("num_z")
    num_z = None if num_z is None else int(num_z)

    ae.train(training)
    if training and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(bundle.device)

    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"LatentFM-AE Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    t0 = time.perf_counter()
    for batch in pbar:
        fields_full = batch["fields"].to(bundle.device)
        x_grid = pointcloud_to_grid3d(fields_full, num_z, num_y, num_x) if num_z is not None else pointcloud_to_grid(fields_full, num_y, num_x)
        x_hat, _ = ae(x_grid)
        x_hat = x_hat[:, :, :num_z, :num_y, :num_x] if num_z is not None else x_hat[:, :, :num_y, :num_x]
        loss = 0.5 * F.l1_loss(x_hat, x_grid) + 0.5 * F.mse_loss(x_hat, x_grid)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    epoch_time_s = time.perf_counter() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(bundle.device) / 1024 ** 2 if torch.cuda.is_available() else 0.0
    return total_loss / max(count, 1), epoch_time_s, peak_mem_mb


def run_epoch_latentfm(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> Tuple[float, float, float]:
    """Run one Latent FM epoch. Returns (mean_loss, epoch_time_s, peak_gpu_mem_mb)."""
    model = bundle.model
    optimizer = bundle.optimizer if training else None
    ema = bundle.ema
    velocity_net = bundle.components["velocity_net"]
    num_y = int(bundle.config["shared"]["data"]["num_y"])
    num_x = int(bundle.config["shared"]["data"]["num_x"])
    num_z = bundle.config["shared"]["data"].get("num_z")
    num_z = None if num_z is None else int(num_z)
    conditioning = bundle.config["shared"]["conditioning"]

    cond_fields = conditioning["cond_fields"]
    n_obs_min = conditioning["n_obs_min_list"]
    n_obs_max = conditioning["n_obs_max_list"]

    model.velocity_net.train(training)
    if training and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(bundle.device)

    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"LatentFM Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    n_fields = model.n_fields
    n_pts = (num_z * num_y * num_x) if num_z is not None else (num_y * num_x)

    t0 = time.perf_counter()
    for batch in pbar:
        coords = batch["coords"].to(bundle.device)
        fields_full = batch["fields"].to(bundle.device)
        valid_mask = batch.get("valid_sensor_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(bundle.device)
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min,
            n_obs_max=n_obs_max,
            valid_mask=valid_mask,
        )

        if num_z is not None:
            fields_grid = pointcloud_to_grid3d(fields_full, num_z, num_y, num_x)
            obs_value_grid, obs_mask_grid = build_obs_grid_mask3d(
                obs_values, obs_mask, obs_field_ids, obs_indices,
                n_fields, n_pts, num_z, num_y, num_x, num_z, num_y, num_x,
            )
            ix = (obs_indices % num_x).float() / max(num_x - 1, 1)
            iy = (obs_indices.div(num_x, rounding_mode="floor") % num_y).float() / max(num_y - 1, 1)
            iz = (obs_indices.div(num_x * num_y, rounding_mode="floor")).float() / max(num_z - 1, 1)
            obs_coords_2d = torch.stack([ix, iy, iz], dim=-1)
        else:
            fields_grid = pointcloud_to_grid(fields_full, num_y, num_x)
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
            )
            ix = (obs_indices % num_x).float() / max(num_x - 1, 1)
            iy = (obs_indices.div(num_x, rounding_mode="floor")).float() / max(num_y - 1, 1)
            obs_coords_2d = torch.stack([ix, iy], dim=-1)

        cond_inputs = {
            "obs_value_grid": obs_value_grid,
            "obs_mask_grid": obs_mask_grid,
            "obs_coords_2d": obs_coords_2d,
            "obs_values": obs_values,
            "obs_mask": obs_mask,
            "obs_field_ids": obs_field_ids,
        }
        loss, _ = model.training_loss(fields_grid, cond_inputs)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.velocity_net.parameters(), max_norm=1.0)
            optimizer.step()
            if ema is not None:
                ema.update(velocity_net)

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    epoch_time_s = time.perf_counter() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(bundle.device) / 1024 ** 2 if torch.cuda.is_available() else 0.0
    return total_loss / max(count, 1), epoch_time_s, peak_mem_mb


@torch.no_grad()
def visualize_ae_reconstruction(
    ae,
    dataset,
    epoch,
    device,
    save_dir,
    Num_y,
    Num_x,
    Num_z=None,
    snapshot_index=0,
    file_tag=None,
    irregular=False,
):
    ae.eval()
    sample = dataset[snapshot_index]
    fields = sample["fields"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"]
    is_3d = Num_z is not None

    if is_3d:
        Num_z = int(Num_z)
        x_grid = pointcloud_to_grid3d(fields, Num_z, Num_y, Num_x)
        x_hat, _ = ae(x_grid)
        x_hat = x_hat[:, :, :Num_z, :Num_y, :Num_x]
        recon_pc = grid3d_to_pointcloud(x_hat, Num_z, Num_y, Num_x)
    elif irregular:
        coords = sample["coords"].unsqueeze(0).to(device)
        if ae.deformer is not None:
            coords_2d = ae.deform(coords)
        else:
            coords_2d = coords[..., :2]
        x_grid = splat_to_grid(coords_2d, fields, Num_y, Num_x)[0]
        x_hat_grid, _ = ae(x_grid)
        recon_pc = gather_from_grid(x_hat_grid[:, :, :Num_y, :Num_x], coords_2d)
    else:
        x_grid = pointcloud_to_grid(fields, Num_y, Num_x)
        x_hat, _ = ae(x_grid)
        x_hat = x_hat[:, :, :Num_y, :Num_x]
        recon_pc = x_hat.permute(0, 2, 3, 1).reshape(1, -1, ae.n_fields)

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    truth = fields * std.view(1, 1, -1) + mean.view(1, 1, -1)
    recon = recon_pc * std.view(1, 1, -1) + mean.view(1, 1, -1)

    truth_np = truth[0].cpu().numpy()
    recon_np = recon[0].cpu().numpy()
    coords_xy = coords_raw.cpu().numpy()[:, :2]

    import matplotlib.tri as mtri
    if is_3d:
        z_vals = coords_raw.cpu().numpy()[:, 2]
        _z_levels = np.unique(z_vals)
        slice_mask = z_vals == _z_levels[len(_z_levels) // 2]
        plot_coords_xy = coords_xy[slice_mask]
        tri = mtri.Triangulation(plot_coords_xy[:, 0], plot_coords_xy[:, 1])
    else:
        slice_mask = slice(None)
        plot_coords_xy = coords_xy
        tri = mtri.Triangulation(coords_xy[:, 0], coords_xy[:, 1])

    metrics = {}
    field_names = dataset.field_names if len(dataset.field_names) == ae.n_fields else tuple(f"field_{i}" for i in range(ae.n_fields))
    for c, name in enumerate(field_names):
        true_f = truth_np[:, c]
        pred_f = recon_np[:, c]
        l2_err = float(np.linalg.norm(true_f - pred_f) / (np.linalg.norm(true_f) + 1e-8))
        metrics[name] = l2_err
        true_plot = true_f[slice_mask]
        pred_plot = pred_f[slice_mask]

        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        vmin = min(true_plot.min(), pred_plot.min())
        vmax = max(true_plot.max(), pred_plot.max())
        for ax, data, title in zip(
            axs,
            [true_plot, pred_plot, np.abs(true_plot - pred_plot)],
            [f"{name} Truth", f"{name} AE Recon", f"{name} |Error| (L2={l2_err:.4e})"],
        ):
            im = ax.tricontourf(
                tri,
                data,
                levels=100,
                cmap="coolwarm" if "Error" not in title else "inferno",
                vmin=vmin if "Error" not in title else None,
                vmax=vmax if "Error" not in title else None,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title)
            ax.set_aspect("equal")

        prefix = file_tag if file_tag else f"epoch_{epoch:04d}"
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{prefix}_ae_field_{name}.png"), dpi=150)
        plt.close(fig)

    return metrics


def visualize_reconstruction_latentfm(
    model,
    dataset,
    Num_x,
    Num_y,
    epoch,
    device,
    save_dir,
    Num_z=None,
    cond_fields=(2,),
    n_obs=256,
    n_steps=8,
    ode_solver="euler",
    snapshot_index=0,
    file_tag=None,
    save_metrics_json=True,
    irregular=False,
    extra_metrics_list=None,
    save_analysis_npz=False,
    cfg=None,
    save_obs_consistency_plots=False,
):
    model.eval()
    if isinstance(cond_fields, int):
        cond_fields = [cond_fields]
    if isinstance(n_obs, int):
        n_obs = [n_obs] * len(cond_fields)

    n_fields = dataset.num_fields
    n_pts = dataset.num_points
    is_3d = Num_z is not None
    if is_3d:
        Num_z = int(Num_z)
    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    if is_3d:
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=truth,
            cond_fields=cond_fields,
            n_obs_min=n_obs,
            n_obs_max=n_obs,
            valid_mask=valid_mask,
        )
        obs_values, obs_mask = _apply_eval_measurement_ops(obs_coords, obs_values, obs_mask, obs_field_ids)
        obs_value_grid, obs_mask_grid = build_obs_grid_mask3d(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            n_fields,
            n_pts,
            Num_z,
            Num_y,
            Num_x,
            Num_z,
            Num_y,
            Num_x,
        )
        ix = (obs_indices % Num_x).float() / max(Num_x - 1, 1)
        iy = (obs_indices.div(Num_x, rounding_mode="floor") % Num_y).float() / max(Num_y - 1, 1)
        iz = (obs_indices.div(Num_x * Num_y, rounding_mode="floor")).float() / max(Num_z - 1, 1)
        obs_coords_2d = torch.stack([ix, iy, iz], dim=-1)
    elif irregular:
        if model.ae.deformer is not None:
            coords_2d = model.ae.deform(coords)
        else:
            coords_2d = coords[..., :2]
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=truth,
            cond_fields=cond_fields,
            n_obs_min=n_obs,
            n_obs_max=n_obs,
            valid_mask=valid_mask,
            coords_2d=coords_2d,
            Ny=Num_y,
            Nx=Num_x,
        )
        B_, M_ = obs_indices.shape
        batch_idx = torch.arange(B_, device=device).unsqueeze(-1).expand(-1, M_)
        obs_coords_2d = coords_2d[batch_idx, obs_indices]
        obs_value_grid, obs_mask_grid = splat_obs_to_grid(
            obs_coords_2d,
            obs_values,
            obs_mask,
            obs_field_ids,
            n_fields,
            Num_y,
            Num_x,
        )
    else:
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=truth,
            cond_fields=cond_fields,
            n_obs_min=n_obs,
            n_obs_max=n_obs,
            valid_mask=valid_mask,
        )
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            n_fields,
            n_pts,
            Num_y,
            Num_x,
            Num_y,
            Num_x,
        )
        ix = (obs_indices % Num_x).float() / max(Num_x - 1, 1)
        iy = (obs_indices.div(Num_x, rounding_mode="floor")).float() / max(Num_y - 1, 1)
        obs_coords_2d = torch.stack([ix, iy], dim=-1)

    cond_inputs = {
        "obs_value_grid": obs_value_grid,
        "obs_mask_grid": obs_mask_grid,
        "obs_coords_2d": obs_coords_2d,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_field_ids": obs_field_ids,
    }
    _bench = os.environ.get("BENCH_SAMPLE", "") == "1"
    if _bench:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); _t0 = time.perf_counter()
    recon_grid = model.sample(cond_inputs, n_steps=n_steps, ode_solver=ode_solver)
    if _bench:
        torch.cuda.synchronize()
        print(f"[bench] latentfm sample: {time.perf_counter() - _t0:.3f} s, "
              f"peak {torch.cuda.max_memory_allocated() / 1024 ** 2:.1f} MB, "
              f"n_steps={n_steps}", flush=True)

    if is_3d:
        recon = grid3d_to_pointcloud(recon_grid, Num_z, Num_y, Num_x)
    elif irregular:
        recon = gather_from_grid(recon_grid[:, :, :Num_y, :Num_x], coords_2d)
    else:
        recon = grid_to_pointcloud(recon_grid, Num_y, Num_x)

    # Posterior-ensemble scoring. relL2 rewards collapse toward the posterior
    # mean, so a proper scoring rule is needed to compare generative
    # reconstruction methods fairly. Draws K independent samples through the
    # baseline's own sampler and scores them with the same estimator used for
    # our model.
    _K = int(os.environ.get("ENSEMBLE_K", "0") or 0)
    if _K > 1:
        import json as _json
        from methods.dmfgen.ensemble_eval import ensemble_metrics as _ens_metrics

        def _draw():
            rg = model.sample(cond_inputs, n_steps=n_steps, ode_solver=ode_solver)
            if is_3d:
                r = grid3d_to_pointcloud(rg, Num_z, Num_y, Num_x)
            elif irregular:
                r = gather_from_grid(rg[:, :, :Num_y, :Num_x], coords_2d)
            else:
                r = grid_to_pointcloud(rg, Num_y, Num_x)
            return r[0].detach().float().cpu().numpy()

        _ens = [recon[0].detach().float().cpu().numpy()]
        for _ in range(_K - 1):
            _ens.append(_draw())
        _ens = np.stack(_ens, axis=0)
        _names = [str(n) for n in getattr(dataset, "field_names",
                                          [f"field_{i}" for i in range(_ens.shape[-1])])]
        _m = _ens_metrics(_ens, truth[0].detach().float().cpu().numpy(), _names)
        _out = os.environ.get("ENSEMBLE_OUT", "")
        print(f"[ensemble] K={_K} " + " ".join(
            f"{k}={v:.5f}" for k, v in _m["aggregate"].items()), flush=True)
        if _out:
            os.makedirs(os.path.dirname(_out), exist_ok=True)
            _json.dump(_m, open(_out, "w"), indent=1)
        # Optional raw-field dump so paper figures can reuse this ensemble
        # instead of reimplementing each baseline's sampler (a reimplementation
        # is what silently dropped latent-FM from the first panel run).
        _npz = os.environ.get("ENSEMBLE_NPZ", "")
        if _npz:
            os.makedirs(os.path.dirname(_npz), exist_ok=True)
            np.savez_compressed(
                _npz, ens=_ens.astype(np.float32),
                truth=truth[0].detach().float().cpu().numpy().astype(np.float32),
                names=np.array(_names))

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)
    recon_phys = recon_phys[0].cpu().numpy()
    truth_phys = truth_phys[0].cpu().numpy()

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    coords_raw_np = coords_raw[0].cpu().numpy()
    coords_xy = coords_raw_np[:, :2]

    triang = None
    body_polygon = None
    if is_3d:
        z_vals = coords_raw_np[:, 2]
        _z_levels = np.unique(z_vals)
        slice_mask = z_vals == _z_levels[len(_z_levels) // 2]
        coords_xy_plot = coords_xy[slice_mask]
        import matplotlib.tri as mtri
        triang = mtri.Triangulation(coords_xy_plot[:, 0], coords_xy_plot[:, 1])
    else:
        slice_mask = slice(None)
        coords_xy_plot = coords_xy
    if (not is_3d) and hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if hasattr(dataset, "airfoil_body_indices") and dataset.airfoil_body_indices is not None:
        body_polygon = coords_xy[dataset.airfoil_body_indices]

    metrics = {}
    field_names = dataset.field_names if len(dataset.field_names) == n_fields else tuple(f"field_{i}" for i in range(n_fields))
    for c, name in enumerate(field_names):
        sensor_coords = None
        field_sensor_mask = obs_field_ids_cpu == c
        if np.any(field_sensor_mask):
            sensor_indices = obs_indices_cpu[field_sensor_mask]
            if is_3d:
                # For 3D, only overlay sensors that lie on the displayed z-midplane
                # slice; projecting all z-levels onto 2D floods the ground-truth
                # panel with green markers.
                on_slice = slice_mask[sensor_indices]
                sensor_coords = coords_xy[sensor_indices[on_slice]] if np.any(on_slice) else None
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

    extra_metrics_list = extra_metrics_list or []
    extra_metrics = {}
    if len(extra_metrics_list) > 0 or save_analysis_npz:
        try:
            grid_info = _infer_structured_grid_from_coords(
                coords_xy,
                num_x=(cfg or {}).get("Num_x", None),
                num_y=(cfg or {}).get("Num_y", None),
            )
        except ValueError:
            grid_info = None
        if grid_info is not None:
            prefix = file_tag or f"epoch_{epoch:04d}"
            for c, name in enumerate(field_names):
                u = _reshape_flat_field_to_grid(truth_phys[:, c], grid_info)
                v = _reshape_flat_field_to_grid(recon_phys[:, c], grid_info)
                field_metrics = {}
                if "ssim" in extra_metrics_list:
                    field_metrics["ssim"] = _ssim2d(u, v, data_range=float(u.max() - u.min()))
                extra_metrics[name] = field_metrics
                if save_analysis_npz:
                    np.savez_compressed(
                        os.path.join(save_dir, f"{prefix}_field_{name}_analysis.npz"),
                        true_grid=u,
                        pred_grid=v,
                        abs_err_grid=np.abs(v - u),
                        x_unique=grid_info["x_unique"],
                        y_unique=grid_info["y_unique"],
                    )

    if save_metrics_json:
        prefix = file_tag or f"epoch_{epoch:04d}"
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "n_steps": int(n_steps),
            "ode_solver": ode_solver,
            "method": "LatentFM",
            "metrics": metrics,
            "extra_metrics": extra_metrics,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return metrics, extra_metrics


class LatentFMAdapter(BaseBaselineAdapter):
    name = "latent_fm"

    def _stage1_checkpoint_path(self, cfg: dict) -> Path:
        stage2_cfg = cfg["latent_fm_params"]["stage2"]
        explicit = stage2_cfg.get("stage1_checkpoint")
        if explicit:
            checkpoint_path = ensure_absolute(explicit)
            if not checkpoint_path.exists():
                raise RuntimeError(
                    f"Latent FM stage 2 requires a valid stage-1 checkpoint, but the configured path does not exist: {checkpoint_path}"
                )
            return checkpoint_path

        stage1_cfg = copy.deepcopy(cfg)
        stage1_cfg["training_stage"] = 1
        save_root = ensure_absolute(cfg["shared"]["paths"]["save_root"])
        latest_stage1_run = find_latest_run_dir(save_root, stage1_cfg)
        if latest_stage1_run is None:
            raise RuntimeError(
                "Latent FM stage 2 requires stage 1 to be completed first. No matching unified stage-1 run directory was found."
            )
        checkpoint_path = latest_stage1_run / "best.pt"
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Latent FM stage 2 requires stage 1 to be completed first. Expected checkpoint not found: {checkpoint_path}"
            )
        return checkpoint_path

    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        stage_cfg = resolve_stage_config(cfg)
        stage = int(cfg["training_stage"])
        training = stage_cfg["training"]
        num_y = int(cfg["shared"]["data"]["num_y"])
        num_x = int(cfg["shared"]["data"]["num_x"])
        num_z_raw = cfg["shared"]["data"].get("num_z")
        num_z = None if num_z_raw is None else int(num_z_raw)
        spatial_dim = int(stage_cfg.get("architecture", {}).get("spatial_dim", 3 if num_z is not None else 2))

        if stage == 1:
            arch = stage_cfg["architecture"]
            if spatial_dim == 3:
                if num_z is None:
                    raise ValueError("LatentFM spatial_dim=3 requires shared.data.num_z.")
                ae = ConvAE3D(
                    n_fields=train_set.num_fields,
                    base_ch=int(arch["base_ch"]),
                    latent_ch=int(arch["latent_ch"]),
                    n_levels=int(arch["n_levels"]),
                    Num_z=num_z,
                    Num_y=num_y,
                    Num_x=num_x,
                    num_res_blocks=int(arch.get("num_res_blocks", 1)),
                ).to(device)
            else:
                ae = ConvAE(
                    n_fields=train_set.num_fields,
                    base_ch=int(arch["base_ch"]),
                    latent_ch=int(arch["latent_ch"]),
                    n_levels=int(arch["n_levels"]),
                    Num_y=num_y,
                    Num_x=num_x,
                    deform_coord_dim=0,
                    num_res_blocks=int(arch.get("num_res_blocks", 2)),
                ).to(device)
            optimizer = torch.optim.AdamW(ae.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]), eta_min=1e-6)
            return BaselineBundle(
                baseline_model=self.name,
                training_stage=1,
                model=ae,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=None,
                device=device,
                run_dir=run_dir,
                config=cfg,
                dataset_train=train_set,
                dataset_val=val_set,
            )

        stage1_checkpoint = self._stage1_checkpoint_path(cfg)
        stage1_ckpt = safe_torch_load(stage1_checkpoint, map_location=device)
        stage1_arch = cfg["latent_fm_params"]["stage1"]["architecture"]
        stage2_arch = stage_cfg["architecture"]
        conditioning_cfg = stage_cfg["conditioning"]
        spatial_dim = int(stage1_ckpt.get("spatial_dim", stage1_arch.get("spatial_dim", spatial_dim)))

        # Infer the actual num_res_blocks from the model state dict keys: after
        # encoder.0 (always a strided Conv), the first subsequent index whose
        # key has a plain ".weight" suffix (not ".net.*.weight") identifies the
        # next strided Conv, so num_res_blocks = that_index - 1.  This is more
        # reliable than the stored "ae_num_res_blocks" metadata, which may be
        # wrong when the checkpoint was saved by an older code version that had
        # a mismatched default between build_for_training and build_checkpoint.
        _sd_keys = set(stage1_ckpt["model"].keys())
        _inferred_nrb_3d = next(
            (i - 1 for i in range(1, 20) if f"encoder.{i}.weight" in _sd_keys),
            int(stage1_ckpt.get("ae_num_res_blocks", stage1_arch.get("num_res_blocks", 1))),
        )
        _inferred_nrb_2d = next(
            (i - 1 for i in range(1, 20) if f"encoder.{i}.weight" in _sd_keys),
            int(stage1_ckpt.get("ae_num_res_blocks", stage1_arch.get("num_res_blocks", 2))),
        )

        if spatial_dim == 3:
            if num_z is None:
                raise ValueError("LatentFM stage 2 loaded a 3D AE but shared.data.num_z is missing.")
            ae = ConvAE3D(
                n_fields=train_set.num_fields,
                base_ch=int(stage1_ckpt.get("ae_base_ch", stage1_arch["base_ch"])),
                latent_ch=int(stage1_ckpt.get("ae_latent_ch", stage1_arch["latent_ch"])),
                n_levels=int(stage1_ckpt.get("ae_n_levels", stage1_arch["n_levels"])),
                Num_z=int(stage1_ckpt.get("Num_z", num_z)),
                Num_y=num_y,
                Num_x=num_x,
                num_res_blocks=_inferred_nrb_3d,
            ).to(device)
        else:
            ae = ConvAE(
                n_fields=train_set.num_fields,
                base_ch=int(stage1_ckpt.get("ae_base_ch", stage1_arch["base_ch"])),
                latent_ch=int(stage1_ckpt.get("ae_latent_ch", stage1_arch["latent_ch"])),
                n_levels=int(stage1_ckpt.get("ae_n_levels", stage1_arch["n_levels"])),
                Num_y=num_y,
                Num_x=num_x,
                deform_coord_dim=int(stage1_ckpt.get("deform_coord_dim", 0)),
                deform_hidden=int(stage1_ckpt.get("deform_hidden", 128)),
                deform_depth=int(stage1_ckpt.get("deform_depth", 3)),
                num_res_blocks=_inferred_nrb_2d,
            ).to(device)
        ae.load_state_dict(stage1_ckpt["model"])
        ae.eval()
        ae.requires_grad_(False)

        velocity_cls = LatentFMUNet3D if spatial_dim == 3 else LatentFMUNet
        velocity_net = velocity_cls(
            latent_ch=int(stage1_ckpt.get("ae_latent_ch", stage1_arch["latent_ch"])),
            n_fields=train_set.num_fields,
            base_ch=int(stage2_arch["base_ch"]),
            ch_mult=tuple(stage2_arch["ch_mult"]),
            num_res_blocks=int(stage2_arch["num_res_blocks"]),
            num_heads=int(stage2_arch["num_heads"]),
        ).to(device)

        pointnet_encoder = None
        cond_mode = str(conditioning_cfg["cond_mode"])
        if cond_mode == "pointnet":
            pointnet_encoder = PointNetSensorEncoder(
                n_fields=train_set.num_fields,
                coord_dim=2,
                hidden_mult=float(conditioning_cfg["pointnet_hidden_mult"]),
            ).to(device)

        model = LatentFlowMatching(
            ae=ae,
            velocity_net=velocity_net,
            Num_x=num_x,
            Num_y=num_y,
            cond_mode=cond_mode,
            pointnet_encoder=pointnet_encoder,
        ).to(device)

        trainable_params = list(velocity_net.parameters())
        if pointnet_encoder is not None:
            trainable_params += list(pointnet_encoder.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]), eta_min=1e-6)
        ema = LatentEMA(velocity_net, decay=float(stage2_arch["ema_decay"]))
        bundle = BaselineBundle(
            baseline_model=self.name,
            training_stage=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            run_dir=run_dir,
            config=cfg,
            dataset_train=train_set,
            dataset_val=val_set,
            components={
                "stage1_checkpoint": str(stage1_checkpoint),
                "velocity_net": velocity_net,
                "pointnet_encoder": pointnet_encoder,
                "cond_mode": cond_mode,
            },
        )
        self._set_latent_scale(bundle, cfg, device, train_set)
        return bundle

    @staticmethod
    def _set_latent_scale(bundle: BaselineBundle, cfg: dict, device, train_set) -> None:
        """Fidelity repair 2: `latent_scale_mode: global | per_channel` rescales the AE
        latents to unit std (statistics from the first 24 TRAIN snapshots). With the
        shipped configs (`none`) the model keeps latent_scale = 1 and is unchanged."""
        arch = cfg["latent_fm_params"]["stage2"]["architecture"]
        mode = str(arch.get("latent_scale_mode", "none")).lower()
        if mode == "none":
            return
        model = bundle.model
        print(f"[latent_fm] cond_mode={model.cond_mode} latent_scale_mode={mode}", flush=True)
        num_x = int(cfg["shared"]["data"]["num_x"])
        num_y = int(cfg["shared"]["data"]["num_y"])
        num_z_raw = cfg["shared"]["data"].get("num_z")
        num_z = None if num_z_raw is None else int(num_z_raw)

        zs = []
        with torch.no_grad():
            for i in range(min(len(train_set), 24)):
                f = train_set[i]["fields"][None].to(device)
                g = (pointcloud_to_grid3d(f, num_z, num_y, num_x) if num_z
                     else pointcloud_to_grid(f, num_y, num_x))
                zs.append(model.ae.encode(g).float())
        z = torch.cat(zs)

        if mode == "per_channel":
            dims = [0] + list(range(2, z.ndim))
            scale = 1.0 / z.std(dim=dims, keepdim=True).clamp(min=1e-6)
        elif mode == "global":
            scale = (1.0 / z.std().clamp(min=1e-6)).reshape(1)
        else:
            raise ValueError(f"latent_scale_mode={mode!r} must be "
                             "'none', 'global' or 'per_channel'")

        model.latent_scale = scale.to(device=device, dtype=torch.float32)
        ch_std = z.std(dim=[0] + list(range(2, z.ndim)))
        print(f"[latent_fm] raw latent std={float(z.std()):.4f} "
              f"per-ch std [{float(ch_std.min()):.3f},{float(ch_std.max()):.3f}] -> "
              f"scale[:6]={[round(v, 4) for v in scale.flatten()[:6].tolist()]}",
              flush=True)

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        if bundle.training_stage == 1:
            bundle.model.load_state_dict(checkpoint["model"])
        else:
            bundle.components["velocity_net"].load_state_dict(checkpoint["velocity_net"])
            pointnet_encoder = bundle.components["pointnet_encoder"]
            if pointnet_encoder is not None and checkpoint.get("pointnet_encoder") is not None:
                pointnet_encoder.load_state_dict(checkpoint["pointnet_encoder"])
            if bundle.ema is not None and checkpoint.get("ema") is not None:
                bundle.ema.load_state_dict(checkpoint["ema"])

        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.scheduler is not None and checkpoint.get("scheduler") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler"])
        ls = checkpoint.get("latent_scale")
        if int(bundle.training_stage) == 2 and ls is not None:
            # store the exact tensor the run trained with (recomputing from the train
            # split would resample the augmentation)
            bundle.model.latent_scale = torch.as_tensor(ls).to(
                device=bundle.device, dtype=torch.float32)
            print(f"[latent_fm] restored latent_scale from checkpoint "
                  f"(numel={bundle.model.latent_scale.numel()})", flush=True)

    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> Tuple[float, float, float]:
        if bundle.training_stage == 1:
            return run_epoch_ae(bundle, loader, training, epoch)
        return run_epoch_latentfm(bundle, loader, training, epoch)

    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        if bundle.training_stage == 1:
            stage1_arch = bundle.config["latent_fm_params"]["stage1"]["architecture"]
            num_z = bundle.config["shared"]["data"].get("num_z")
            # Derive spatial_dim the same way build_for_training does: use num_z
            # as the ground truth rather than stage1_arch["spatial_dim"] which
            # may be absent.  This keeps ae_num_res_blocks consistent with the
            # default used when the model was actually constructed.
            _spatial_dim_1 = int(stage1_arch.get("spatial_dim", 3 if num_z is not None else 2))
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
                "method": "LatentFM_AE",
                "ae_base_ch": int(stage1_arch["base_ch"]),
                "ae_latent_ch": int(stage1_arch["latent_ch"]),
                "ae_n_levels": int(stage1_arch["n_levels"]),
                "ae_num_res_blocks": int(stage1_arch.get("num_res_blocks", 1 if _spatial_dim_1 == 3 else 2)),
                "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
                "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
                "Num_z": None if num_z is None else int(num_z),
                "spatial_dim": _spatial_dim_1,
                "dataset": bundle.config["shared"]["data"]["dataset_name"],
            }

        stage1_arch = bundle.config["latent_fm_params"]["stage1"]["architecture"]
        stage2_arch = bundle.config["latent_fm_params"]["stage2"]["architecture"]
        num_z = bundle.config["shared"]["data"].get("num_z")
        _spatial_dim_2 = int(stage1_arch.get("spatial_dim", 3 if num_z is not None else 2))
        ck = {
            "baseline_model": bundle.baseline_model,
            "training_stage": bundle.training_stage,
            "velocity_net": bundle.components["velocity_net"].state_dict(),
            "pointnet_encoder": bundle.components["pointnet_encoder"].state_dict() if bundle.components["pointnet_encoder"] is not None else None,
            "ema": bundle.ema.state_dict() if bundle.ema is not None else None,
            "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
            "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "ae_checkpoint": bundle.components["stage1_checkpoint"],
            "mean": bundle.dataset_train.mean,
            "std": bundle.dataset_train.std,
            "field_names": bundle.dataset_train.field_names,
            "method": "LatentFM",
            "ae_base_ch": int(stage1_arch["base_ch"]),
            "ae_latent_ch": int(stage1_arch["latent_ch"]),
            "ae_n_levels": int(stage1_arch["n_levels"]),
            "ae_num_res_blocks": int(stage1_arch.get("num_res_blocks", 1 if _spatial_dim_2 == 3 else 2)),
            "fm_base_ch": int(stage2_arch["base_ch"]),
            "fm_ch_mult": list(stage2_arch["ch_mult"]),
            "fm_num_res_blocks": int(stage2_arch["num_res_blocks"]),
            "fm_num_heads": int(stage2_arch["num_heads"]),
            "ema_decay": float(stage2_arch["ema_decay"]),
            "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
            "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
            "Num_z": None if num_z is None else int(num_z),
            "spatial_dim": _spatial_dim_2,
            "cond_mode": bundle.components["cond_mode"],
        }
        if bundle.model._latent_scale_active():
            ck["latent_scale"] = bundle.model.latent_scale.detach().cpu()
        # Checkpoint-noise window (LFM_CKPT_WINDOW="lo:hi:step", epochs inclusive):
        # both trainers keep only best.pt/last.pt, so a dense window around the budget
        # point is archived on request to measure checkpoint-to-checkpoint variance.
        spec = os.environ.get("LFM_CKPT_WINDOW", "")
        if spec:
            lo, hi, step = (int(v) for v in spec.split(":"))
            if lo <= epoch <= hi and (epoch - lo) % step == 0:
                d = Path(bundle.run_dir) / "ckpt_window"
                d.mkdir(parents=True, exist_ok=True)
                torch.save(ck, d / f"epoch_{epoch:05d}.pt")
                print(f"[latent_fm] archived window checkpoint epoch={epoch}", flush=True)
        return ck

    @contextlib.contextmanager
    def evaluation_weights(self, bundle: BaselineBundle) -> Iterator[None]:
        if bundle.training_stage != 2 or bundle.ema is None:
            yield
            return
        velocity_net = bundle.components["velocity_net"]
        bundle.ema.apply(velocity_net)
        try:
            yield
        finally:
            bundle.ema.restore(velocity_net)

    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None, save_obs_consistency_plots: bool = False) -> dict[str, float]:
        shared_cond = bundle.config["shared"]["conditioning"]
        num_z = bundle.config["shared"]["data"].get("num_z")
        if bundle.training_stage == 1:
            return visualize_ae_reconstruction(
                ae=bundle.model,
                dataset=dataset,
                epoch=epoch,
                device=bundle.device,
                save_dir=str(save_dir),
                Num_y=int(bundle.config["shared"]["data"]["num_y"]),
                Num_x=int(bundle.config["shared"]["data"]["num_x"]),
                Num_z=None if num_z is None else int(num_z),
                snapshot_index=snapshot_index,
                file_tag="latent_fm_ae",
                irregular=False,
            )

        stage2_cfg = resolve_stage_config(bundle.config)
        sampling_cfg = stage2_cfg["sampling"]
        n_steps = int(sampling_cfg["benchmark_n_steps"][0] if n_steps is None else n_steps)
        metrics, _ = visualize_reconstruction_latentfm(
            model=bundle.model,
            dataset=dataset,
            Num_x=int(bundle.config["shared"]["data"]["num_x"]),
            Num_y=int(bundle.config["shared"]["data"]["num_y"]),
            Num_z=None if num_z is None else int(num_z),
            epoch=epoch,
            device=bundle.device,
            save_dir=str(save_dir),
            cond_fields=shared_cond["vis_cond_fields"],
            n_obs=shared_cond["vis_n_obs_list"],
            n_steps=n_steps,
            ode_solver=str(sampling_cfg["ode_solver"]),
            snapshot_index=snapshot_index,
            file_tag=f"latentfm_N{n_steps}",
            save_metrics_json=True,
            irregular=False,
            save_obs_consistency_plots=save_obs_consistency_plots,
        )
        return metrics
