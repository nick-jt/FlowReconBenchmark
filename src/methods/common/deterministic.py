# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import json
import numpy as np
import os
import torch
from pathlib import Path
from typing import Optional
from typing import Sequence
from methods.common.helpers_baseline import _build_structured_triangulation
from methods.common.helpers_baseline import _save_single_field_plot
from methods.common.helpers_baseline import build_obs_grid_mask
from methods.common.helpers_baseline import build_sparse_condition
from methods.common.helpers_baseline import grid_to_pointcloud
from methods.dmfgen.helpers import save_sensor_parity_plot, save_sensor_residual_plot
from methods.dmfgen.obs_consistency import observation_consistency_metrics
from methods.common.adapters import BaselineBundle
from methods.common.viz import _apply_eval_measurement_ops


def _attach_senconsis_outputs(
    *,
    metrics: dict,
    recon: torch.Tensor,
    truth: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_field_ids: torch.Tensor,
    coords: torch.Tensor,
    coords_xy: np.ndarray,
    field_names: Sequence[str],
    save_dir,
    save_obs_consistency_plots: bool,
) -> dict:
    """SenConsis is sensor consistency between generated and observed sensors."""
    obs_metrics = observation_consistency_metrics(
        recon=recon,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_indices=obs_indices,
        obs_field_ids=obs_field_ids,
        field_names=field_names,
    )
    metrics.update(obs_metrics)
    payload = {
        "coords": coords.detach().cpu(),
        "coords_xy": coords_xy,
        "truth": truth.detach().cpu(),
        "target": truth.detach().cpu(),
        "recon": recon.detach().cpu(),
        "obs_coords": obs_coords.detach().cpu(),
        "obs_values": obs_values.detach().cpu(),
        "obs_mask": obs_mask.detach().cpu(),
        "obs_indices": obs_indices.detach().cpu(),
        "obs_field_ids": obs_field_ids.detach().cpu(),
        "field_names": list(field_names),
    }
    if save_obs_consistency_plots:
        senconsis_dir = Path(save_dir) / "SenConsis"
        senconsis_dir.mkdir(parents=True, exist_ok=True)
        with open(senconsis_dir / "obs_consistency_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(obs_metrics, handle, indent=2)
        save_sensor_parity_plot(payload, str(senconsis_dir))
        save_sensor_residual_plot(payload, str(senconsis_dir))
    return payload


@torch.no_grad()
def visualize_reconstruction_deterministic(
    bundle: BaselineBundle,
    dataset,
    save_dir: Path,
    epoch: int,
    snapshot_index: int = 0,
    file_tag: Optional[str] = None,
    save_obs_consistency_plots: bool = False,
) -> dict[str, float]:
    model = bundle.model
    model.eval()
    shared_cond = bundle.config["shared"]["conditioning"]
    cond_fields = shared_cond["vis_cond_fields"]
    n_obs = shared_cond["vis_n_obs_list"]
    if isinstance(cond_fields, int):
        cond_fields = [cond_fields]
    if isinstance(n_obs, int):
        n_obs = [n_obs] * len(cond_fields)

    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(bundle.device)
    coords_raw = sample["coords_raw"].unsqueeze(0).to(bundle.device)
    truth = sample["fields"].unsqueeze(0).to(bundle.device)
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(bundle.device)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs,
        n_obs_max=n_obs,
        valid_mask=valid_mask,
    )
    obs_values, obs_mask = _apply_eval_measurement_ops(obs_coords, obs_values, obs_mask, obs_field_ids)

    method = bundle.baseline_model
    if method in {"senseiver", "mlp_rbf"}:
        recon = model(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        method_label = "Senseiver" if method == "senseiver" else "MLP_RBF_supervised"
    elif method == "geofno":
        variant = str(bundle.components["variant"])
        if variant == "irregular":
            recon = model(coords[..., :2], obs_values, obs_mask, obs_field_ids, obs_indices)
        elif variant == "deform":
            from methods.geofno.adapter import predict_geofno_mesh
            recon = predict_geofno_mesh(model, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        else:
            n_fields = dataset.num_fields
            n_pts = dataset.num_points
            num_y = int(bundle.config["shared"]["data"]["num_y"])
            num_x = int(bundle.config["shared"]["data"]["num_x"])
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
                point_to_grid=bundle.components.get("point_to_grid"),
            )
            pred_grid = model(obs_value_grid, obs_mask_grid)
            recon = grid_to_pointcloud(
                pred_grid,
                num_y,
                num_x,
                point_to_grid=bundle.components.get("point_to_grid"),
            )
        method_label = "GeoFNO_supervised"
    else:
        raise ValueError(f"Deterministic visualizer does not support {method!r}.")

    # A deterministic forecast is a degenerate predictive distribution, for
    # which CRPS reduces exactly to the mean absolute error. Reporting it that
    # way puts the point-estimate baseline on the same axis as the generative
    # ones instead of excluding it.
    _K = int(os.environ.get("ENSEMBLE_K", "0") or 0)
    if _K >= 1:
        import json as _json
        _r = recon[0].detach().float().cpu().numpy()
        _y = truth[0].detach().float().cpu().numpy()
        _crps = float(np.abs(_r - _y).mean())
        _rel = float(np.linalg.norm(_r - _y) / (np.linalg.norm(_y) + 1e-12))
        print(f"[ensemble] deterministic crps(=MAE)={_crps:.5f} rel_l2={_rel:.5f}",
              flush=True)
        _out = os.environ.get("ENSEMBLE_OUT", "")
        if _out:
            os.makedirs(os.path.dirname(_out), exist_ok=True)
            _json.dump({"aggregate": {"crps": _crps, "rel_l2_mean": _rel,
                                      "spread": 0.0, "deterministic": True}},
                       open(_out, "w"), indent=1)

    mean = dataset.mean.to(bundle.device)
    std = dataset.std.to(bundle.device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)
    recon_np = recon_phys[0].cpu().numpy()
    truth_np = truth_phys[0].cpu().numpy()

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    coords_np = coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]

    # For volumetric data, plot a z-midplane slice rather than every z-level
    # projected onto one plane. Without this the triangulation sees ~125
    # coincident points per (x, y) location and the figure is meaningless.
    # Mirrors the slicing already done in the latent-FM and AE visualizers so
    # every model's figures show the same plane.
    is_3d = coords_np.shape[1] >= 3 and len(np.unique(coords_np[:, 2])) > 1
    if is_3d:
        z_vals = coords_np[:, 2]
        _z_levels = np.unique(z_vals)
        slice_mask = z_vals == _z_levels[len(_z_levels) // 2]
        coords_xy_plot = coords_xy[slice_mask]
    else:
        slice_mask = slice(None)
        coords_xy_plot = coords_xy

    triang = None
    body_polygon = None
    if (not is_3d) and hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if hasattr(dataset, "airfoil_body_indices") and dataset.airfoil_body_indices is not None:
        body_polygon = coords_xy[dataset.airfoil_body_indices]

    field_names = (
        dataset.field_names
        if len(dataset.field_names) == truth_np.shape[1]
        else tuple(f"field_{idx}" for idx in range(truth_np.shape[1]))
    )
    metrics: dict[str, float] = {}
    for field_idx, field_name in enumerate(field_names):
        sensor_coords = None
        field_sensor_mask = obs_field_ids_cpu == field_idx
        if np.any(field_sensor_mask):
            sensor_indices = obs_indices_cpu[field_sensor_mask]
            if is_3d:
                # Only overlay sensors lying on the displayed slice; drawing
                # every z-level would bury the field under sensor markers.
                on_slice = slice_mask[sensor_indices]
                sensor_coords = (coords_xy[sensor_indices[on_slice]]
                                 if np.any(on_slice) else None)
            else:
                sensor_coords = coords_xy[sensor_indices]
        metrics[field_name] = _save_single_field_plot(
            true_f=truth_np[:, field_idx][slice_mask],
            pred_f=recon_np[:, field_idx][slice_mask],
            metric_true_f=truth_np[:, field_idx],
            metric_pred_f=recon_np[:, field_idx],
            coords_xy=coords_xy_plot,
            sensor_coords=sensor_coords,
            field_name=field_name,
            epoch=epoch,
            save_dir=str(save_dir),
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

    prefix = file_tag or f"epoch_{epoch:04d}"
    payload = {
        "epoch": int(epoch),
        "snapshot_index": int(snapshot_index),
        "cond_fields": [int(v) for v in cond_fields],
        "n_obs": [int(v) for v in n_obs],
        "method": method_label,
        "metrics": metrics,
    }
    with open(Path(save_dir) / f"{prefix}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return metrics
