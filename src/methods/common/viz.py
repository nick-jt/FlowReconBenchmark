# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


def _infer_structured_grid_from_coords(
    coords_xy: np.ndarray,
    decimals: int = 8,
    num_x: Optional[int] = None,
    num_y: Optional[int] = None,
):
    x = np.round(coords_xy[:, 0], decimals=decimals)
    y = np.round(coords_xy[:, 1], decimals=decimals)
    n_pts = coords_xy.shape[0]

    if num_x is not None and num_y is not None:
        nx = int(num_x)
        ny = int(num_y)
        if nx > 0 and ny > 0 and nx * ny == n_pts:
            sort_idx = np.lexsort((x, y))
            unique_x = np.unique(x)
            unique_y = np.unique(y)
            dx = float(np.mean(np.diff(unique_x))) if len(unique_x) > 1 else 1.0
            dy = float(np.mean(np.diff(unique_y))) if len(unique_y) > 1 else 1.0
            return {
                "nx": nx,
                "ny": ny,
                "sort_idx": sort_idx,
                "x_unique": unique_x,
                "y_unique": unique_y,
                "dx": dx,
                "dy": dy,
            }

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    nx = len(unique_x)
    ny = len(unique_y)
    if nx * ny != n_pts:
        raise ValueError(
            f"Coordinates do not form a complete structured 2D grid. Inferred nx={nx}, ny={ny}, N={n_pts}."
        )
    sort_idx = np.lexsort((x, y))
    dx = float(np.mean(np.diff(unique_x))) if nx > 1 else 1.0
    dy = float(np.mean(np.diff(unique_y))) if ny > 1 else 1.0
    return {
        "nx": nx,
        "ny": ny,
        "sort_idx": sort_idx,
        "x_unique": unique_x,
        "y_unique": unique_y,
        "dx": dx,
        "dy": dy,
    }


def _reshape_flat_field_to_grid(field_flat: np.ndarray, grid_info: dict) -> np.ndarray:
    vals = field_flat[grid_info["sort_idx"]]
    return vals.reshape(grid_info["ny"], grid_info["nx"])


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: str = "cpu"):
    ax = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, window_size, window_size)


def _ssim2d(u: np.ndarray, v: np.ndarray, data_range: Optional[float] = None, window_size: int = 11, sigma: float = 1.5) -> float:
    x = torch.from_numpy(u).float().unsqueeze(0).unsqueeze(0)
    y = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0)
    if data_range is None:
        data_range = float(max(u.max(), v.max()) - min(u.min(), v.min()))
    data_range = max(float(data_range), 1e-8)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    kernel = _gaussian_kernel(window_size=window_size, sigma=sigma)
    pad = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return float(ssim_map.mean().item())


@torch.no_grad()


def _apply_eval_measurement_ops(obs_coords, obs_values, obs_mask, obs_field_ids):
    """Env-gated test-time measurement operators for matched robustness evals.

    ENSEMBLE_NOISE    additive Gaussian sensor noise sigma (z-score units)
    ENSEMBLE_OCCLUDE  "slab:0.25" or "ball:0.25" - remove sensors in a region
    ENSEMBLE_DROPFIELD  comma list of field ids whose sensors are removed
    ENSEMBLE_OP_SEED  seed so every model sees the identical realization
    """
    import os as _os
    _ns = float(_os.environ.get("ENSEMBLE_NOISE", "0") or 0)
    _occ = _os.environ.get("ENSEMBLE_OCCLUDE", "")
    _drop = _os.environ.get("ENSEMBLE_DROPFIELD", "")
    if not (_ns > 0 or _occ or _drop):
        return obs_values, obs_mask
    _seed = int(_os.environ.get("ENSEMBLE_OP_SEED", "0") or 0)
    _g = torch.Generator(device=obs_values.device).manual_seed(_seed)
    if _ns > 0:
        _noise = torch.randn(obs_values.shape, device=obs_values.device,
                             dtype=obs_values.dtype, generator=_g)
        obs_values = obs_values + _ns * _noise * obs_mask.unsqueeze(-1)
    if _occ:
        from methods.dmfgen.measurement_ops import apply_occlusion as _occl
        _kind, _frac = _occ.split(":")
        obs_mask = _occl(obs_coords, obs_mask, prob=1.0, kind=_kind,
                         frac_min=float(_frac), frac_max=float(_frac), generator=_g)
    if _drop:
        _ids = torch.tensor([int(x) for x in _drop.split(",")],
                            device=obs_mask.device)
        _rm = (obs_field_ids.unsqueeze(-1) == _ids.view(1, 1, -1)).any(-1)
        obs_mask = obs_mask * (~_rm).to(obs_mask.dtype)
    print(f"[eval_ops] noise={_ns} occlude={_occ!r} drop={_drop!r} seed={_seed} "
          f"valid={int(obs_mask.sum())}", flush=True)
    return obs_values, obs_mask
