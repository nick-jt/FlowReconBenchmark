# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import contextlib
import json
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Tuple
from methods.common.helpers_baseline import _build_structured_triangulation
from methods.common.helpers_baseline import _save_single_field_plot
from methods.common.helpers_baseline import build_obs_grid_mask
from methods.common.helpers_baseline import build_sparse_condition
from methods.common.helpers_baseline import compute_pad_size
from methods.common.helpers_baseline import grid_to_pointcloud
from methods.common.helpers_baseline import midplane_slice
from methods.common.helpers_baseline import nearest_fill_grid
from methods.common.helpers_baseline import nearest_sensor_fill_nodes
from methods.common.helpers_baseline import pointcloud_to_grid_padded
from methods.common.helpers_baseline import scatter_sensors_to_nodes
from methods.common.helpers_baseline import validate_regular_grid_compatibility
from methods.common.helpers_wing_baseline import build_wing_condition
from methods.common.helpers_wing_baseline import describe_condition as describe_wing_condition
from methods.common.helpers_wing_baseline import surface_pool_enabled
from methods.common.helpers_wing_baseline import wing_cond_channels
from methods.common.helpers_wing_baseline import wing_fill_nodes
from methods.sit.transport import Transport as LinearVelocityTransport, create_transport, Sampler
from methods.common.adapters import BaseBaselineAdapter, BaselineBundle
from methods.common.config import _require_no_surface_pool, resolve_stage_config
from methods.common.deterministic import _attach_senconsis_outputs
from methods.sit.model import SiTEMA, SiTPhysics


def _sit_wing_tokens(bundle, coords, fields_full, obs_coords, obs_values,
                     obs_mask, obs_field_ids, node_subsample):
    """Token set + surface-pool conditioning for one SHIFT-WING SiT step.

    A fresh uniform node subsample becomes the transformer token set (the wing
    has 400k nodes; full self-attention over them is not a thing). Sensors live
    on the body surface and are essentially never members of that subset, so
    conditioning reaches the tokens geometrically, through
    helpers_wing_baseline.wing_fill_nodes: per FIELD ID, the nearest valid
    sensor's value plus a soft support weight, with the Mach/alpha tokens
    broadcast globally at support 1.

    Returns (obs_value_nodes, obs_mask_nodes, coords_tok, x_tokens).
    """
    n_cond_ch = int(bundle.components["cond_value_channels"])
    idx = torch.randperm(fields_full.shape[1], device=bundle.device)[:node_subsample]
    coords_tok = coords[:, idx]
    x_tokens = fields_full[:, idx]
    val, sup = wing_fill_nodes(
        coords_tok, obs_coords, obs_values, obs_mask, obs_field_ids,
        n_cond_ch=n_cond_ch,
        sigma=float(bundle.components.get("cond_fill_sigma", 0.05)),
    )
    return val, sup, coords_tok, x_tokens


def run_epoch_sit(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
    net = bundle.model
    optimizer = bundle.optimizer if training else None
    transport = bundle.components["transport"]
    h_pad = int(bundle.components["H_pad"])
    w_pad = int(bundle.components["W_pad"])
    grid_order = bundle.components.get("grid_order")
    point_to_grid = bundle.components.get("point_to_grid")
    cond_mode = bundle.components["cond_mode"]
    tokenizer = bundle.components["tokenizer"]
    spike_state = bundle.components["spike_state"]
    grad_clip = bundle.components["grad_clip"]
    spike_skip = bundle.components["spike_skip"]
    ema = bundle.ema
    all_params_fn = bundle.components["all_params_fn"]
    n_fields = bundle.components["n_fields"]

    num_y = int(bundle.config["shared"]["data"]["num_y"])
    num_x = int(bundle.config["shared"]["data"]["num_x"])
    cond_fields = bundle.config["shared"]["conditioning"]["cond_fields"]
    n_obs_min = bundle.config["shared"]["conditioning"]["n_obs_min_list"]
    n_obs_max = bundle.config["shared"]["conditioning"]["n_obs_max_list"]
    huber_beta = bundle.components["huber_beta"]
    _surface_pool = surface_pool_enabled(bundle.config)

    net.train(training)
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"SiT Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    for batch in pbar:
        fields_full = batch["fields"].to(bundle.device)
        coords = batch["coords"].to(bundle.device)
        model_kwargs: dict[str, Any] = {}
        valid_mask = batch.get("valid_sensor_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(bundle.device)

        if tokenizer == "pointnet":
            if _surface_pool:
                obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = (
                    build_wing_condition(batch, bundle.device,
                                         n_obs_min=n_obs_min, n_obs_max=n_obs_max))
            else:
                obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
                    coords_full=coords,
                    fields_full=fields_full,
                    cond_fields=cond_fields,
                    n_obs_min=n_obs_min,
                    n_obs_max=n_obs_max,
                    valid_mask=valid_mask,
                )
            if training and epoch == 1 and count == 0:
                print(f"[cond] sit source="
                      f"{'surface_pool' if _surface_pool else 'volume'} "
                      f"{describe_wing_condition(obs_mask[:1], obs_field_ids[:1])}",
                      flush=True)
            node_subsample = int(bundle.components.get("node_subsample") or 0)
            if _surface_pool:
                # Surface-pool obs_indices point into the 65,536-node SURFACE
                # pool, a different point set from the volume tokens, so
                # scatter_sensors_to_nodes (which uses them as volume indices)
                # is meaningless here -- it would write sensor values onto
                # arbitrary interior nodes. The geometric nearest-sensor fill
                # is the only correct route, and it is also the only one that
                # can carry field ids 4-8. Require a token budget so the fill
                # is the path that runs.
                if not (node_subsample and node_subsample < fields_full.shape[1]):
                    raise ValueError(
                        "SiT + surface_pool requires sit_params.architecture."
                        "node_subsample < num_points; the full-node path uses "
                        "scatter_sensors_to_nodes, which indexes the volume "
                        "with SURFACE-pool indices.")
                obs_value_nodes, obs_mask_nodes, coords_tok, x_grid = _sit_wing_tokens(
                    bundle, coords, fields_full, obs_coords, obs_values,
                    obs_mask, obs_field_ids, node_subsample)
                model_kwargs["coords"] = coords_tok
                model_kwargs["obs_value_nodes"] = obs_value_nodes
                model_kwargs["obs_mask_nodes"] = obs_mask_nodes
            elif node_subsample and node_subsample < fields_full.shape[1]:
                # Token-budgeted training: a fresh uniform node subset becomes
                # the token set; sensors (drawn from the FULL field, protocol
                # unchanged) reach the tokens through nearest-sensor fill,
                # since they are generally not members of the subset.
                idx = torch.randperm(fields_full.shape[1], device=bundle.device)[:node_subsample]
                coords_tok = coords[:, idx]
                x_grid = fields_full[:, idx]
                obs_value_nodes, obs_mask_nodes = nearest_sensor_fill_nodes(
                    coords_tok, obs_coords, obs_values, obs_mask, obs_field_ids,
                    n_fields, sigma=float(bundle.components.get("cond_fill_sigma", 0.05)),
                )
                model_kwargs["coords"] = coords_tok
            else:
                x_grid = fields_full
                obs_value_nodes, obs_mask_nodes = scatter_sensors_to_nodes(
                    obs_values,
                    obs_mask,
                    obs_field_ids,
                    obs_indices,
                    fields_full.shape[0],
                    fields_full.shape[1],
                    n_fields,
                    bundle.device,
                    fields_full.dtype,
                )
                model_kwargs["coords"] = coords
            model_kwargs["obs_value_nodes"] = obs_value_nodes
            model_kwargs["obs_mask_nodes"] = obs_mask_nodes
        else:
            x_grid = pointcloud_to_grid_padded(
                fields_full, num_y, num_x, h_pad, w_pad, grid_order=grid_order
            )
            obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
                coords_full=coords,
                fields_full=fields_full,
                cond_fields=cond_fields,
                n_obs_min=n_obs_min,
                n_obs_max=n_obs_max,
                valid_mask=valid_mask,
            )
            obs_value_grid, obs_mask_grid = build_obs_grid_mask(
                obs_values,
                obs_mask,
                obs_field_ids,
                obs_indices,
                n_fields,
                fields_full.shape[1],
                num_y,
                num_x,
                h_pad,
                w_pad,
                point_to_grid=point_to_grid,
            )
            if cond_mode == "interp":
                obs_value_grid = nearest_fill_grid(obs_value_grid, obs_mask_grid)
            model_kwargs["obs_value_grid"] = obs_value_grid
            model_kwargs["obs_mask_grid"] = obs_mask_grid

        loss_dict = transport.training_losses(
            net,
            x_grid,
            model_kwargs=model_kwargs,
            huber_beta=huber_beta,
        )
        loss = loss_dict["loss"].float().mean()
        current_loss = float(loss.detach().cpu())

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # max_norm=inf measures the norm without clipping
            grad_norm = float(nn.utils.clip_grad_norm_(
                all_params_fn(), max_norm=float("inf") if grad_clip is None else float(grad_clip)))

            skip = not (math.isfinite(current_loss) and math.isfinite(grad_norm))
            ema_loss = spike_state.get("ema_loss")
            ema_grad = spike_state.get("ema_grad")
            if spike_skip and not skip and ema_loss is not None and current_loss > 5.0 * ema_loss:
                skip = True
            if spike_skip and not skip and ema_grad is not None and grad_norm > 10.0 * ema_grad:
                skip = True

            if skip:
                optimizer.zero_grad(set_to_none=True)
                spike_state["skipped"] = spike_state.get("skipped", 0) + 1
            else:
                optimizer.step()
                if ema is not None:
                    ema.update(all_params_fn())
                beta = 0.99
                spike_state["ema_loss"] = (
                    current_loss if ema_loss is None else beta * ema_loss + (1.0 - beta) * current_loss
                )
                spike_state["ema_grad"] = (
                    grad_norm if ema_grad is None else beta * ema_grad + (1.0 - beta) * grad_norm
                )

        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


@torch.no_grad()
def sit_conditional_sample(
    net: nn.Module,
    transport: LinearVelocityTransport,
    shape: Sequence[int],
    obs_value_grid: Optional[torch.Tensor] = None,
    obs_mask_grid: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    obs_value_nodes: Optional[torch.Tensor] = None,
    obs_mask_nodes: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    n_steps: int = 50,
    sampler_type: str = "euler",
) -> torch.Tensor:
    if isinstance(sampler_type, str):
        sampler_type = {"euler": "Euler", "heun": "Heun"}.get(sampler_type.lower(), sampler_type)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method=sampler_type, num_steps=n_steps)
    init = torch.randn(tuple(shape), device=device)
    kwargs: dict[str, Any] = {}
    if coords is not None:
        kwargs["coords"] = coords
    if obs_value_grid is not None:
        kwargs["obs_value_grid"] = obs_value_grid
        kwargs["obs_mask_grid"] = obs_mask_grid
    if obs_value_nodes is not None:
        kwargs["obs_value_nodes"] = obs_value_nodes
        kwargs["obs_mask_nodes"] = obs_mask_nodes
    traj = sample_fn(init, net, **kwargs)
    return traj[-1]


@torch.no_grad()
def sit_conditional_sample_points_chunked(
    net: nn.Module,
    transport: LinearVelocityTransport,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    n_fields: int,
    device: torch.device,
    n_steps: int,
    sampler_type: str,
    chunk: int = 8192,
    sigma: float = 0.05,
    cond_value_channels: Optional[int] = None,
    surface_pool: bool = False,
    rechunk: bool = False,
) -> torch.Tensor:
    """Full-field conditional sample for the point-token SiT, chunked to the
    training token budget. Conditioning per chunk uses the same nearest-sensor
    fill as training. Chunks are sampled independently, so pointwise metrics
    (rel-L2, CRPS, coverage) are well defined while single-sample coherence is
    limited to the chunk size -- documented as a property of this baseline.

    ``rechunk=True`` integrates all N points as one ODE state instead: every
    drift evaluation draws a fresh random partition into chunks and runs each
    chunk through the network, so no point stays in one independent chunk for
    the whole trajectory. Solver, time grid, network calls per sample and
    tokens per call are those of the chunked sampler.

    ``surface_pool=True`` (SHIFT-WING) swaps the fill for
    helpers_wing_baseline.wing_fill_nodes over ``cond_value_channels`` field
    ids, matching what training did; everything else is unchanged."""
    cond_ch = int(cond_value_channels) if cond_value_channels else int(n_fields)
    B, N, _ = coords.shape

    def fill(coords_c):
        if surface_pool:
            return wing_fill_nodes(coords_c, obs_coords, obs_values, obs_mask, obs_field_ids,
                                   n_cond_ch=cond_ch, sigma=sigma)
        return nearest_sensor_fill_nodes(coords_c, obs_coords, obs_values, obs_mask,
                                         obs_field_ids, n_fields, sigma=sigma)

    if rechunk:
        # Both fills are pointwise, so one fill over all N points sliced per
        # chunk equals the fill of each chunk.
        val, sup = fill(coords)
        sampler = Sampler(transport)
        drift = sampler.drift

        def rechunked_drift(x, t, model):
            perm = torch.randperm(N, device=device)
            v = torch.empty_like(x)
            for s in range(0, N, chunk):
                sel = perm[s:s + chunk]
                v[:, sel] = drift(x[:, sel], t, model, coords=coords[:, sel],
                                  obs_value_nodes=val[:, sel], obs_mask_nodes=sup[:, sel])
            return v

        # Replacing the drift (not the loop) keeps sample_ode's interval,
        # time grid and solver exactly those of the chunked sampler.
        sampler.drift = rechunked_drift
        sample_fn = sampler.sample_ode(sampling_method=sampler_type, num_steps=n_steps)
        return sample_fn(torch.randn(B, N, n_fields, device=device), net)[-1]

    out = torch.empty(B, N, n_fields, device=device)
    # Chunk over a RANDOM PERMUTATION, not contiguous indices. The H5 point
    # ordering is raster with z fastest and x slowest, so a contiguous slice of
    # 8192 points spans exactly one x-plane: measured per-axis extent
    # [0.000, 0.399, 0.761] versus [0.761, 0.761, 0.761] for the uniform-random
    # token sets used in training. Sampling on planar token clouds is far out of
    # distribution for a model trained on space-filling ones, and the resulting
    # raster aliasing appeared in the figures as sheared vertical striping on
    # the unobserved channels. Permuting makes each chunk a uniform 1/n-density
    # subsample of the cube, matching training. Chunks remain independent, which
    # is a real and documented limit on single-sample coherence.
    perm = torch.randperm(N, device=device)
    for s in range(0, N, chunk):
        sel = perm[s:s + chunk]
        coords_c = coords[:, sel]
        val, sup = fill(coords_c)
        out[:, sel] = sit_conditional_sample(
            net=net,
            transport=transport,
            shape=(B, coords_c.shape[1], n_fields),
            coords=coords_c,
            obs_value_nodes=val,
            obs_mask_nodes=sup,
            device=device,
            n_steps=n_steps,
            sampler_type=sampler_type,
        )
    return out


@torch.no_grad()
def visualize_reconstruction_sit(
    net: nn.Module,
    transport: LinearVelocityTransport,
    dataset,
    num_x: int,
    num_y: int,
    h_pad: int,
    w_pad: int,
    epoch: int,
    device: torch.device,
    save_dir: Path,
    cond_fields: Sequence[int],
    n_obs: Sequence[int],
    n_steps: int,
    sampler_type: str,
    snapshot_index: int = 0,
    file_tag: Optional[str] = None,
    tokenizer: str = "patch",
    cond_mode: str = "image",
    point_to_grid: Optional[torch.Tensor] = None,
    save_obs_consistency_plots: bool = False,
    node_subsample: int = 0,
    cond_fill_sigma: float = 0.05,
) -> dict[str, float]:
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

    if tokenizer == "pointnet":
        if node_subsample and node_subsample < n_pts:
            # Token-budgeted model: chunked sampling with the same
            # nearest-sensor-fill conditioning used during training.
            def _draw_point_sample():
                return sit_conditional_sample_points_chunked(
                    net=net,
                    transport=transport,
                    coords=coords,
                    obs_coords=obs_coords,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_field_ids=obs_field_ids,
                    n_fields=n_fields,
                    device=device,
                    n_steps=n_steps,
                    sampler_type=sampler_type,
                    chunk=int(node_subsample),
                    sigma=float(cond_fill_sigma),
                )
            recon = _draw_point_sample()
        else:
            obs_value_nodes, obs_mask_nodes = scatter_sensors_to_nodes(
                obs_values,
                obs_mask,
                obs_field_ids,
                obs_indices,
                1,
                n_pts,
                n_fields,
                device,
                truth.dtype,
            )
            def _draw_point_sample():
                return sit_conditional_sample(
                    net=net,
                    transport=transport,
                    shape=(1, n_pts, n_fields),
                    coords=coords,
                    obs_value_nodes=obs_value_nodes,
                    obs_mask_nodes=obs_mask_nodes,
                    device=device,
                    n_steps=n_steps,
                    sampler_type=sampler_type,
                )
            recon = _draw_point_sample()
    else:
        _draw_point_sample = None
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            n_fields,
            n_pts,
            num_y,
            num_x,
            h_pad,
            w_pad,
            point_to_grid=point_to_grid,
        )
        if cond_mode == "interp":
            obs_value_grid = nearest_fill_grid(obs_value_grid, obs_mask_grid)
        recon_grid = sit_conditional_sample(
            net=net,
            transport=transport,
            shape=(1, n_fields, h_pad, w_pad),
            obs_value_grid=obs_value_grid,
            obs_mask_grid=obs_mask_grid,
            device=device,
            n_steps=n_steps,
            sampler_type=sampler_type,
        )
        recon = grid_to_pointcloud(recon_grid, num_y, num_x, point_to_grid=point_to_grid)

    # Posterior-ensemble scoring for the SiT baseline, mirroring the
    # latent-FM hook: K independent samples through the model's own sampler,
    # scored with the same estimator used for our model.
    _K = int(os.environ.get("ENSEMBLE_K", "0") or 0)
    if _K > 1 and _draw_point_sample is not None:
        import json as _json
        from methods.dmfgen.ensemble_eval import ensemble_metrics as _ens_metrics
        _ens = [recon[0].detach().float().cpu().numpy()]
        for _ in range(_K - 1):
            _ens.append(_draw_point_sample()[0].detach().float().cpu().numpy())
        _ens = np.stack(_ens, axis=0)
        _names = [str(nm) for nm in getattr(dataset, "field_names",
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
    coords_np = coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]

    # Volumetric data: plot the z-midplane slice. Handing every z-level to a
    # 2-D triangulation superposes ~125 coincident points per (x, y) and the
    # figure collapses into a smear. Matches the latent-FM / deterministic
    # visualizers so all models show the same plane.
    is_3d, slice_mask, coords_xy_plot, triang = midplane_slice(coords_np)
    if not is_3d and hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)

    metrics = {}
    field_names = dataset.field_names if len(dataset.field_names) == n_fields else tuple(f"field_{i}" for i in range(n_fields))
    for idx, name in enumerate(field_names):
        sensor_coords = None
        field_sensor_mask = obs_field_ids_cpu == idx
        if np.any(field_sensor_mask):
            sensor_indices = obs_indices_cpu[field_sensor_mask]
            if is_3d:
                # Only overlay sensors lying on the displayed slice.
                on_slice = slice_mask[sensor_indices]
                sensor_coords = (coords_xy[sensor_indices[on_slice]]
                                 if np.any(on_slice) else None)
            else:
                sensor_coords = coords_xy[sensor_indices]
        metrics[name] = _save_single_field_plot(
            true_f=truth_phys[:, idx][slice_mask],
            pred_f=recon_phys[:, idx][slice_mask],
            metric_true_f=truth_phys[:, idx],
            metric_pred_f=recon_phys[:, idx],
            coords_xy=coords_xy_plot,
            sensor_coords=sensor_coords,
            field_name=name,
            epoch=epoch,
            save_dir=str(save_dir),
            file_prefix=file_tag,
            triang=triang,
            body_polygon=None,
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

    payload = {
        "epoch": int(epoch),
        "snapshot_index": int(snapshot_index),
        "cond_fields": [int(v) for v in cond_fields],
        "n_obs": [int(v) for v in n_obs],
        "n_steps": int(n_steps),
        "sampler_type": sampler_type,
        "method": "SiT",
        "metrics": metrics,
    }
    prefix = file_tag or f"epoch_{epoch:04d}"
    with open(save_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return metrics


class SiTAdapter(BaseBaselineAdapter):
    name = "sit"

    def _checkpoint_context(self, bundle: BaselineBundle, checkpoint: dict) -> str:
        stage_cfg = resolve_stage_config(bundle.config)
        arch = stage_cfg["architecture"]
        fields = [
            f"checkpoint method={checkpoint.get('method', '<missing>')!r}",
            f"checkpoint baseline_model={checkpoint.get('baseline_model', '<missing>')!r}",
            f"checkpoint training_stage={checkpoint.get('training_stage', '<missing>')!r}",
            f"checkpoint epoch={checkpoint.get('epoch', '<missing>')!r}",
            f"checkpoint grid=({checkpoint.get('Num_y', '<missing>')}, {checkpoint.get('Num_x', '<missing>')})",
            f"checkpoint padded=({checkpoint.get('H_pad', '<missing>')}, {checkpoint.get('W_pad', '<missing>')})",
            f"checkpoint patch_size={checkpoint.get('patch_size', '<missing>')!r}",
            f"checkpoint hidden_size={checkpoint.get('hidden_size', '<missing>')!r}",
            f"checkpoint depth={checkpoint.get('depth', '<missing>')!r}",
            f"checkpoint num_heads={checkpoint.get('sit_num_heads', '<missing>')!r}",
            f"checkpoint tokenizer={checkpoint.get('tokenizer', '<missing>')!r}",
            f"checkpoint cond_channels={checkpoint.get('cond_channels', '<missing>')!r}",
            f"config grid=({bundle.config['shared']['data']['num_y']}, {bundle.config['shared']['data']['num_x']})",
            f"config padded=({bundle.components['H_pad']}, {bundle.components['W_pad']})",
            f"config patch_size={arch['patch_size']!r}",
            f"config hidden_size={arch['hidden_size']!r}",
            f"config depth={arch['depth']!r}",
            f"config num_heads={arch['num_heads']!r}",
            f"config tokenizer={arch['tokenizer']!r}",
            f"config cond_channels={bundle.components['cond_channels']!r}",
        ]
        return "; ".join(fields)

    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        stage_cfg = resolve_stage_config(cfg)
        training = stage_cfg["training"]
        arch = stage_cfg["architecture"]
        transport_cfg = stage_cfg["transport"]
        conditioning_cfg = stage_cfg["conditioning"]
        num_y = int(cfg["shared"]["data"]["num_y"])
        num_x = int(cfg["shared"]["data"]["num_x"])
        tokenizer_kind = str(arch["tokenizer"])
        node_subsample = int(arch.get("node_subsample") or 0)
        patch_size = int(arch["patch_size"])
        if tokenizer_kind == "pointnet":
            # Point-token path: no grid, no patchification. The token count is
            # the per-step node subsample budget (training) and the sampling
            # chunk (inference); without a subsample it is the full node count,
            # which is only viable for small discretizations.
            grid_info = {"grid_order": None, "point_to_grid": None}
            h_pad, w_pad = 0, 0
            token_count = node_subsample if node_subsample else int(train_set.num_points)
            token_message = (
                f"SiT point-token count is {token_count} "
                f"(node_subsample={node_subsample or 'full'}, "
                f"num_points={int(train_set.num_points)})."
            )
        else:
            grid_info = validate_regular_grid_compatibility(train_set, num_x, num_y)
            if val_set is not train_set:
                validate_regular_grid_compatibility(val_set, num_x, num_y)
            h_pad, w_pad = compute_pad_size(num_y, num_x, patch_size)
            token_count = (h_pad // patch_size) * (w_pad // patch_size)
            token_message = (
                f"SiT patch token count is {token_count} "
                f"(({h_pad}//{patch_size}) * ({w_pad}//{patch_size})) for padded grid "
                f"{h_pad}x{w_pad}."
            )
        allow_large_token_count = bool(arch.get("allow_large_token_count", False))
        if token_count > 8192 and not allow_large_token_count:
            raise ValueError(
                token_message
                + " This is likely too large for stable/efficient SiT baseline training. "
                "Set sit_params.architecture.allow_large_token_count: true only after "
                "explicitly accepting the memory/runtime cost."
            )
        if token_count > 8192:
            warnings.warn(
                token_message + " Proceeding because allow_large_token_count=true.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif token_count > 4096:
            warnings.warn(
                token_message + " This may be memory-heavy for SiT; consider a larger patch_size.",
                RuntimeWarning,
                stacklevel=2,
            )
        n_fields = train_set.num_fields
        # The conditioning stack has one value channel and one mask/support
        # channel per SENSOR FIELD ID, plus one "any sensor here" channel.
        # On the volume datasets sensor ids and field ids coincide, so this is
        # the historical 2*n_fields+1. On SHIFT-WING a sensor can be one of 9
        # types (0-3 volume, 4-6 wall shear, 7-8 Mach/alpha) against 4 output
        # fields; a 4-wide stack cannot even hold a wall-shear reading.
        cond_value_channels = (wing_cond_channels(train_set)
                               if surface_pool_enabled(cfg) else n_fields)
        cond_channels = 2 * cond_value_channels + 1

        net = SiTPhysics(
            input_size_h=h_pad,
            input_size_w=w_pad,
            patch_size=patch_size,
            in_channels=n_fields,
            cond_channels=cond_channels,
            hidden_size=int(arch["hidden_size"]),
            depth=int(arch["depth"]),
            num_heads=int(arch["num_heads"]),
            mlp_ratio=float(arch["mlp_ratio"]),
            tokenizer=str(arch["tokenizer"]),
            coord_dim=int(arch["coord_dim"]),
            fourier_num_freqs=int(arch["fourier_num_freqs"]),
            fourier_scale=float(arch["fourier_scale"]),
        ).to(device)

        def all_params_fn():
            return list(net.parameters())

        def build_param_groups(params: Iterable[torch.nn.Parameter]) -> list[dict[str, Any]]:
            decay, no_decay = [], []
            for param in params:
                if not param.requires_grad:
                    continue
                if param.ndim < 2:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {"params": decay, "weight_decay": float(training["weight_decay"])},
                {"params": no_decay, "weight_decay": 0.0},
            ]

        # `optimizer: upstream` is SiT's own recipe (train.py: AdamW, lr 1e-4, no weight
        # decay, constant LR, no gradient clipping, no step skipping); the default
        # (AdamW betas 0.9/0.95, warm-up + cosine, clip 0.5, spike skipping)
        # reproduces the pre-audit runs.
        upstream_opt = str(training.get("optimizer", "benchmark")).lower() == "upstream"
        if upstream_opt:
            optimizer = torch.optim.AdamW(all_params_fn(), lr=float(training["learning_rate"]),
                                          weight_decay=float(training.get("weight_decay", 0.0)))
            scheduler = None
        else:
            optimizer = torch.optim.AdamW(
                build_param_groups(all_params_fn()),
                lr=float(training["learning_rate"]),
                betas=(0.9, 0.95),
                eps=float(training["adam_eps"]),
            )
            # Configurable only so a 3-epoch smoke can actually learn something:
            # with the default 200-epoch linear warmup the LR is ~0 for the first
            # few epochs, so a short run's loss trace is pure noise and a
            # "descending loss" gate on it would be meaningless. Defaults to 200,
            # so no existing or full-length run changes.
            warmup_epochs = int(training.get("warmup_epochs", 200))

            def lr_lambda(epoch_index: int) -> float:
                if epoch_index < warmup_epochs:
                    return epoch_index / max(1, warmup_epochs)
                progress = (epoch_index - warmup_epochs) / max(1, int(training["epochs"]) - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        ema = SiTEMA(all_params_fn(), decay=float(arch["ema_rate"]))
        transport = create_transport(
            path_type=str(transport_cfg["path_type"]),
            prediction=str(transport_cfg["prediction"]),
            loss_weight=transport_cfg.get("loss_weight"),
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
                "transport": transport,
                "H_pad": h_pad,
                "W_pad": w_pad,
                "cond_channels": cond_channels,
                "cond_value_channels": cond_value_channels,
                "n_fields": n_fields,
                "cond_mode": str(conditioning_cfg["cond_mode"]),
                "tokenizer": tokenizer_kind,
                "node_subsample": node_subsample,
                "cond_fill_sigma": float(arch.get("cond_fill_sigma", 0.05)),
                "grid_order": grid_info["grid_order"],
                "point_to_grid": grid_info["point_to_grid"],
                "grid_info": grid_info,
                "huber_beta": float(training.get("huber_beta", 0.0)),
                "grad_clip": None if upstream_opt else 0.5,
                "spike_skip": not upstream_opt,
                "spike_state": {"ema_loss": None, "ema_grad": None, "skipped": 0},
                "all_params_fn": all_params_fn,
            },
        )

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        if "model" not in checkpoint:
            raise KeyError("SiT checkpoint is missing required 'model' state_dict.")
        model_state = checkpoint["model"]
        if not isinstance(model_state, dict):
            raise TypeError(f"SiT checkpoint['model'] must be a state_dict, got {type(model_state)!r}.")
        model_state = dict(model_state)
        model_state.pop("_metadata", None)

        expected_keys = set(bundle.model.state_dict().keys())
        found_keys = set(model_state.keys())
        missing_keys = sorted(expected_keys - found_keys)
        unexpected_keys = sorted(found_keys - expected_keys)
        benign_missing: set[str] = set()
        benign_unexpected: set[str] = set()
        bad_missing = [key for key in missing_keys if key not in benign_missing]
        bad_unexpected = [key for key in unexpected_keys if key not in benign_unexpected]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "SiT checkpoint/model state_dict mismatch. Refusing to load with "
                "silent strict=False behavior because this usually means the run "
                "config changed (patch_size, hidden_size, depth, cond_channels, "
                "tokenizer, or related architecture settings). "
                f"Missing keys: {bad_missing[:20]}"
                f"{' ...' if len(bad_missing) > 20 else ''}. "
                f"Unexpected keys: {bad_unexpected[:20]}"
                f"{' ...' if len(bad_unexpected) > 20 else ''}. "
                f"Context: {self._checkpoint_context(bundle, checkpoint)}"
            )
        try:
            bundle.model.load_state_dict(model_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "SiT checkpoint tensors do not match the current model even though "
                "the key sets match. This commonly indicates an architecture/config "
                "mismatch. "
                f"Context: {self._checkpoint_context(bundle, checkpoint)}"
            ) from exc
        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.scheduler is not None and checkpoint.get("scheduler") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler"])
        if bundle.ema is not None and checkpoint.get("ema") is not None:
            ema_state = checkpoint["ema"]
            shadow = ema_state.get("shadow") if isinstance(ema_state, dict) else None
            live_params = bundle.components["all_params_fn"]()
            if shadow is not None:
                if len(shadow) != len(live_params):
                    raise RuntimeError(
                        "SiT EMA checkpoint length does not match the current model "
                        f"({len(shadow)} shadow tensors vs {len(live_params)} parameters). "
                        f"Context: {self._checkpoint_context(bundle, checkpoint)}"
                    )
                mismatched = [
                    (idx, tuple(saved.shape), tuple(param.shape))
                    for idx, (saved, param) in enumerate(zip(shadow, live_params))
                    if tuple(saved.shape) != tuple(param.shape)
                ]
                if mismatched:
                    preview = mismatched[:10]
                    raise RuntimeError(
                        "SiT EMA checkpoint tensor shapes do not match the current model. "
                        f"First mismatches: {preview}"
                        f"{' ...' if len(mismatched) > 10 else ''}. "
                        f"Context: {self._checkpoint_context(bundle, checkpoint)}"
                    )
            bundle.ema.load_state_dict(checkpoint["ema"])
        if checkpoint.get("spike_state") is not None:
            bundle.components["spike_state"] = dict(checkpoint["spike_state"])

    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> Tuple[float, float, float]:
        if training and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(bundle.device)
        t0 = time.perf_counter()
        loss = run_epoch_sit(bundle, loader, training, epoch)
        epoch_time_s = time.perf_counter() - t0
        peak_mem_mb = torch.cuda.max_memory_allocated(bundle.device) / 1024 ** 2 if torch.cuda.is_available() else 0.0
        return loss, epoch_time_s, peak_mem_mb

    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        stage_cfg = resolve_stage_config(bundle.config)
        arch = stage_cfg["architecture"]
        return {
            "baseline_model": bundle.baseline_model,
            "training_stage": bundle.training_stage,
            "model": bundle.model.state_dict(),
            "ema": bundle.ema.state_dict() if bundle.ema is not None else None,
            "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
            "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
            "spike_state": dict(bundle.components["spike_state"]),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "method": "SiT_FlowMatching",
            "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
            "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
            "H_pad": int(bundle.components["H_pad"]),
            "W_pad": int(bundle.components["W_pad"]),
            "n_fields": int(bundle.components["n_fields"]),
            "cond_channels": int(bundle.components["cond_channels"]),
            "patch_size": int(arch["patch_size"]),
            "hidden_size": int(arch["hidden_size"]),
            "depth": int(arch["depth"]),
            "sit_num_heads": int(arch["num_heads"]),
            "mlp_ratio": float(arch["mlp_ratio"]),
            "tokenizer": str(arch["tokenizer"]),
            "coord_dim": int(arch["coord_dim"]),
            "fourier_num_freqs": int(arch["fourier_num_freqs"]),
            "fourier_scale": float(arch["fourier_scale"]),
            "cond_mode": str(bundle.components["cond_mode"]),
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
        _require_no_surface_pool(bundle.config, "sit")
        return visualize_reconstruction_sit(
            net=bundle.model,
            transport=bundle.components["transport"],
            dataset=dataset,
            num_x=int(bundle.config["shared"]["data"]["num_x"]),
            num_y=int(bundle.config["shared"]["data"]["num_y"]),
            h_pad=int(bundle.components["H_pad"]),
            w_pad=int(bundle.components["W_pad"]),
            epoch=epoch,
            device=bundle.device,
            save_dir=save_dir,
            cond_fields=shared_cond["vis_cond_fields"],
            n_obs=shared_cond["vis_n_obs_list"],
            n_steps=n_steps,
            sampler_type=str(sampling_cfg["ode_solver"]),
            snapshot_index=snapshot_index,
            file_tag=f"sit_N{n_steps}",
            tokenizer=bundle.components["tokenizer"],
            cond_mode=bundle.components["cond_mode"],
            point_to_grid=bundle.components.get("point_to_grid"),
            save_obs_consistency_plots=save_obs_consistency_plots,
            node_subsample=int(bundle.components.get("node_subsample") or 0),
            cond_fill_sigma=float(bundle.components.get("cond_fill_sigma", 0.05)),
        )
