"""
Sparse-observation consistency usage:

- default_hard is the default and preserves current pointwise hard replacement behavior.
- endpoint applies rectified-flow clean-endpoint pointwise observation masking.
- endpoint_smooth applies rectified-flow clean-endpoint Gaussian/RBF smooth observation masking.
- All added SenConsis outputs are saved under SenConsis/, 
                 activate this using --obs-consistency-mode & visualize by --save-obs-consistency-plots.
- SenConsis metrics are relative L2 sensor-consistency errors.
"""
from paths import DATA_DIR

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import yaml
import pickle
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phycoflow_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

from methods.dmfgen.helpers import TurbulentCombustionH5Dataset, save_obs_consistency_comparison, visualize_reconstruction, reconstruct_snapshot

from methods.dmfgen.Model import ConditionalPointMLPRBF, ConditionalPointPerceiver, ConditionalPointHybridLocalGlobalRBF, PointCloudFFM
try:
    from methods.dmfgen.Model import FNO, FNOFFM
except ImportError:
    FNO = None
    FNOFFM = None

def parse_args():
    p = argparse.ArgumentParser(
        "Standalone evaluator for trained FFM models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--Demo-Num", dest="Demo_Num", type=int, required=True, 
                   help="Demo ID to recover.")
    p.add_argument("--demo-root", type=str, default=".", 
                   help="Project/demo root directory.")
    p.add_argument("--split", type=str, default="test", 
                   choices=["train", "val", "test"])
    p.add_argument("--snapshot-index", type=int, default=0, 
                   help="Index within the selected split.")
    
    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Override visualization cond_fields. Defaults to YAML vis_cond_fields or cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Override visualization n_obs list. Defaults to YAML vis_n_obs_list or n_obs_max_list.")
    
    p.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"],
                   help="Which checkpoint to load from the recovered run directory.")
    p.add_argument("--n-steps-generation", type=int, default = 4,
                   help="Override generation steps. Defaults to YAML n_steps_generation if present.")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    
    p.add_argument("--extra-metrics", type=str, nargs="*", default=[], choices=["ssim", "grad", "spectrum"], 
                   help="Optional extra metrics to compute on structured grids (2D or 3D).",)
    p.add_argument("--save-analysis-npz", action="store_true",
                   help="If set, save per-field intermediate arrays (grids, gradients, spectra) to .npz files.",
    )
    p.add_argument(
        "--obs-consistency-mode",
        choices=["none", "default_hard", "endpoint", "endpoint_smooth"],
        default="default_hard",
        help="Sparse-observation consistency mode used during sampling.",
    )
    p.add_argument("--obs-consistency-strength", type=float, default=1.0)
    p.add_argument("--obs-consistency-sigma", type=float, default=0.05)
    p.add_argument("--obs-consistency-schedule-power", type=float, default=2.0)
    p.add_argument(
        "--no-obs-consistency-final-clamp",
        action="store_true",
        help="Disable the final exact sensor clamp for observation-consistency modes.",
    )
    p.add_argument(
        "--save-obs-consistency-plots",
        action="store_true",
        help="Save SenConsis metrics and sensor-consistency figures.",
    )
    p.add_argument(
        "--obs-consistency-compare-modes",
        nargs="+",
        default=None,
        choices=["none", "default_hard", "endpoint", "endpoint_smooth"],
        help="Evaluate multiple sparse-observation consistency modes using the same sensor set.",
    )

    return p.parse_args()

class IIDGaussianPrior(torch.nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


class RFFGaussianPrior(torch.nn.Module):
    def __init__(self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * np.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return np.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)

def _extract_timestamp(path: Path) -> Optional[str]:
    m = re.search(r"DemoN(\d+)_(\d{8}_\d{6})", path.name)
    if m is None:
        m = re.search(r"demo_N(\d+)_(\d{8}_\d{6})", path.name)
    return m.group(2) if m else None


def _find_latest_yaml(cfg_dir: Path, demo_num: int) -> Path:
    pattern = f"config_pointcloud_ffm_DemoN{demo_num}_*.yaml"
    candidates = sorted(cfg_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No config backup found for Demo_Num={demo_num} in {cfg_dir}"
        )

    def _sort_key(p: Path):
        ts = _extract_timestamp(p)
        return ts if ts is not None else p.stat().st_mtime

    candidates = sorted(candidates, key=_sort_key)
    return candidates[-1]


def _normalize_eval_config(cfg: dict) -> dict:
    cfg = dict(cfg)

    if cfg.get("cond_fields") is None:
        cfg["cond_fields"] = [cfg.get("cond_field", 2)]
    if cfg.get("n_obs_min_list") is None:
        cfg["n_obs_min_list"] = [cfg.get("n_obs_min", 64)]
    if cfg.get("n_obs_max_list") is None:
        cfg["n_obs_max_list"] = [cfg.get("n_obs_max", 256)]

    if cfg.get("vis_cond_fields") in (None, ""):
        cfg["vis_cond_fields"] = list(cfg["cond_fields"])
    if cfg.get("vis_n_obs_list") in (None, ""):
        cfg["vis_n_obs_list"] = list(cfg["n_obs_max_list"])

    if cfg.get("backbone") is None:
        cfg["backbone"] = "mlp_rbf"

    return cfg


def _build_prior(cfg: dict):
    if cfg.get("prior", "rff") == "iid":
        return IIDGaussianPrior()
    if cfg.get("prior", "rff") in ("rff_powerlaw", "rff_kolmogorov"):
        from methods.dmfgen.spectral_prior import PowerLawRFFPrior
        return PowerLawRFFPrior(
            coord_dim=3,
            n_features=cfg.get("rff_features", 1024),
            slope=float(cfg.get("prior_slope", 5.0 / 3.0)),
            k_min=float(cfg.get("prior_k_min", 1.0)),
            k_max=float(cfg.get("prior_k_max", 48.0)),
            seed=int(cfg.get("seed", 0) or 0),
        )
    return RFFGaussianPrior(
        coord_dim=3,
        n_features=cfg.get("rff_features", 256),
        lengthscale=cfg.get("rff_lengthscale", 0.15),
    )


def _build_model(cfg: dict, dataset) -> torch.nn.Module:
    prior = _build_prior(cfg)
    backbone_name = cfg.get("backbone", "mlp_rbf")

    if backbone_name == "perceiver":
        backbone = ConditionalPointPerceiver(
            n_fields=dataset.num_fields,
            coord_dim=3,
            latent_dim=cfg.get("latent_dim", 256),
            num_latents=cfg.get("num_latents", 128),
            num_heads=cfg.get("num_heads", 8),
            num_latent_blocks=cfg.get("num_latent_blocks", 4),
            field_embed_dim=cfg.get("field_embed_dim", 128),
            ff_mult=cfg.get("ff_mult", 4),
            attn_dropout=cfg.get("attn_dropout", 0.0),
            mlp_dropout=cfg.get("mlp_dropout", 0.0),
            decode_chunk_size=cfg.get("decode_chunk_size", 4096),
            share_query_proj=cfg.get("share_query_proj", False),
        )
        model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    if backbone_name == "fno":
        if FNO is None or FNOFFM is None:
            raise RuntimeError("YAML says backbone='fno' but FNO/FNOFFM are not available in Model.py")
        Num_x = cfg.get("Num_x", None)
        Num_y = cfg.get("Num_y", None)
        if Num_x is None or Num_y is None:
            raise ValueError("FNO evaluation requires Num_x and Num_y in YAML.")
        backbone = FNO(
            n_fields=dataset.num_fields,
            Num_x=Num_x,
            Num_y=Num_y,
            n_modes_x=cfg.get("fno_modes_x", 32),
            n_modes_y=cfg.get("fno_modes_y", 8),
            hidden_channels=cfg.get("fno_hidden_channels", 64),
            n_layers=cfg.get("fno_n_layers", 4),
            condition_blur=cfg.get("condition_blur", False),
            condition_blur_kernel=cfg.get("condition_blur_kernel", 5),
            condition_blur_sigma=cfg.get("condition_blur_sigma", 1.0),
        )
        model = FNOFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    if backbone_name == "fno3d":
        from methods.dmfgen.fno3d_backbone import FNO3D, FNO3DFFM, FNO3DRegression
        Num_x, Num_y, Num_z = cfg.get("Num_x"), cfg.get("Num_y"), cfg.get("Num_z")
        if Num_x is None or Num_y is None or Num_z is None:
            raise ValueError("FNO3D evaluation requires Num_x, Num_y and Num_z in the checkpoint cfg / YAML.")
        backbone = FNO3D(
            n_fields=dataset.num_fields,
            Num_x=Num_x, Num_y=Num_y, Num_z=Num_z,
            n_modes_x=cfg.get("fno_modes_x", 16),
            n_modes_y=cfg.get("fno_modes_y", 16),
            n_modes_z=cfg.get("fno_modes_z", 16),
            hidden_channels=cfg.get("fno_hidden_channels", 27),
            n_layers=cfg.get("fno_n_layers", 4),
            condition_blur=cfg.get("condition_blur", False),
            condition_blur_kernel=cfg.get("condition_blur_kernel", 5),
            condition_blur_sigma=cfg.get("condition_blur_sigma", 1.0),
            domain_padding=cfg.get("fno_domain_padding", None),
        )
        wrapper = FNO3DRegression if cfg.get("objective", "flow") == "regression" else FNO3DFFM
        model = wrapper(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    if backbone_name == "GL_rbf_CQ":
        # Vendored portable compact-query backbone. The saved args.json holds
        # the merged CQ config, so rebuilding through the adapter (which
        # re-merges the core defaults, then the same arg overrides) restores
        # an identical schema; t_sampling only affects training.
        from types import SimpleNamespace
        raise NotImplementedError("the GL_rbf_CQ backbone ablation is not part of FlowReconBench (see PhyCoFlow_demo)")
        build_cq_model = None  # noqa: F841
        model, _ = build_cq_model(SimpleNamespace(**cfg), dataset, "cpu")
        return model

    if backbone_name in ["GL_rbf", "GL_rbf_ENH"]:
        enhanced = backbone_name == "GL_rbf_ENH"
        sensor_coord_encoding = cfg.get("sensor_coord_encoding", "fourier" if enhanced else "raw")
        latent_sensor_reinject = cfg.get("latent_sensor_reinject", enhanced)
        query_latent_readout = cfg.get("query_latent_readout", enhanced)
        enhanced_head_norm = cfg.get("enhanced_head_norm", enhanced)
        query_readout_scale_init = cfg.get("query_readout_scale_init", 1e-2 if enhanced else 0.0)
        glres_scale_init = cfg.get("glres_scale_init", 1e-2 if enhanced else 0.0)
        query_readout_type = cfg.get(
            "query_readout_type",
            "coord" if enhanced or query_latent_readout else "point",
        )

        backbone = ConditionalPointHybridLocalGlobalRBF(
            n_fields=dataset.num_fields,
            coord_dim=3,
            hidden_dim=cfg.get("hidden_dim", 256),
            cond_dim=cfg.get("cond_dim", 128),
            field_embed_dim=cfg.get("field_embed_dim", 128),
            latent_dim=cfg.get("latent_dim", 256),
            num_latents=cfg.get("num_latents", 128),
            num_heads=cfg.get("num_heads", 8),
            num_latent_blocks=cfg.get("num_latent_blocks", 4),
            ff_mult=cfg.get("ff_mult", 4),
            attn_dropout=cfg.get("attn_dropout", 0),
            mlp_dropout=cfg.get("mlp_dropout", 0),
            rbf_sigma=cfg.get("rbf_sigma", 0.05),
            summary_type=cfg.get("summary_type", "cls"),

            gather_mode=cfg.get("gather_mode", "rbf"),
            gather_topk=cfg.get("gather_topk", 32),
            gather_query_chunk_size=cfg.get("gather_query_chunk_size", None),
            learnable_rbf_sigma=cfg.get("learnable_rbf_sigma", False),
            neighbor_backend=cfg.get("neighbor_backend", "torch"),

            sensor_local_topk=cfg.get("sensor_local_topk", 32),
            sensor_local_dropout=cfg.get("sensor_local_dropout", 0.0),
            use_fourier_pe=cfg.get("USE_FOURIER_PE", False),
            fourier_pe_num_bands=cfg.get("fourier_pe_num_bands", 32),
            fourier_pe_max_freq=cfg.get("fourier_pe_max_freq", 64.0),
            enhanced_backbone=enhanced,
            sensor_coord_encoding=sensor_coord_encoding,
            latent_sensor_reinject=latent_sensor_reinject,
            latent_reinject_every=cfg.get("latent_reinject_every", 1),
            query_latent_readout=query_latent_readout,
            query_readout_type=query_readout_type,
            query_readout_scale_init=query_readout_scale_init,
            enhanced_head_norm=enhanced_head_norm,
            glres_scale_init=glres_scale_init,
            n_obs_field_types=cfg.get("n_obs_field_types",
                                      getattr(dataset, "n_obs_field_types", None)),
        )
        model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    backbone = ConditionalPointMLPRBF(
        n_fields=dataset.num_fields,
        coord_dim=3,
        hidden_dim=cfg.get("hidden_dim", 256),
        cond_dim=cfg.get("cond_dim", 128),
        field_embed_dim=cfg.get("field_embed_dim", 128),
        rbf_sigma=cfg.get("rbf_sigma", 0.05),
        use_fourier_pe=cfg.get("USE_FOURIER_PE", False),
        fourier_pe_num_bands=cfg.get("fourier_pe_num_bands", 32),
        fourier_pe_max_freq=cfg.get("fourier_pe_max_freq", 64.0),
    )
    model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))

    return model


# ---------------------------------------------------------------------------
# Structured-grid inference (2D and 3D)
# ---------------------------------------------------------------------------

def _infer_structured_grid(
    coords: np.ndarray,
    decimals: int = 8,
    num_x: Optional[int] = None,
    num_y: Optional[int] = None,
    num_z: Optional[int] = None,
):
    """
    Recover a structured 2D or 3D grid description from point coordinates.

    coords : [N, D] with D >= 2.  If D >= 3 and z contains multiple unique
             values the grid is treated as 3D; otherwise 2D.

    Returns a dict with keys:
        ndim, nx, ny, nz (None for 2D), sort_idx,
        x_unique, y_unique, z_unique (None for 2D),
        dx, dy, dz (None for 2D).
    """
    n_pts, D = coords.shape

    x = np.round(coords[:, 0], decimals=decimals)
    y = np.round(coords[:, 1], decimals=decimals)

    unique_x = np.unique(x)
    unique_y = np.unique(y)

    if D >= 3:
        z = np.round(coords[:, 2], decimals=decimals)
        unique_z = np.unique(z)
    else:
        z = None
        unique_z = np.array([0.0])

    is_3d = len(unique_z) > 1

    # ------------------------------------------------------------------
    # 3D grid
    # ------------------------------------------------------------------
    if is_3d:
        # Option 1: explicit shape from config
        if (num_x is not None and num_y is not None and num_z is not None
                and int(num_x) > 0 and int(num_y) > 0 and int(num_z) > 0):
            nx, ny, nz = int(num_x), int(num_y), int(num_z)
            if nx * ny * nz == n_pts:
                sort_idx = np.lexsort((x, y, z))
                dx = float(np.mean(np.diff(unique_x))) if len(unique_x) > 1 else 1.0
                dy = float(np.mean(np.diff(unique_y))) if len(unique_y) > 1 else 1.0
                dz = float(np.mean(np.diff(unique_z))) if len(unique_z) > 1 else 1.0
                return {
                    "ndim": 3, "nx": nx, "ny": ny, "nz": nz,
                    "sort_idx": sort_idx,
                    "x_unique": unique_x, "y_unique": unique_y, "z_unique": unique_z,
                    "dx": dx, "dy": dy, "dz": dz,
                }
            else:
                print(
                    f"[Warning: !] Provided Num_x={nx}, Num_y={ny}, Num_z={nz} are "
                    f"inconsistent with N={n_pts}; falling back to coordinate inference."
                )

        # Option 2: infer from coordinates
        nx = len(unique_x)
        ny = len(unique_y)
        nz = len(unique_z)

        if nx * ny * nz != n_pts:
            raise ValueError(
                f"Coordinates do not form a complete structured 3D grid and no valid "
                f"(Num_x, Num_y, Num_z) was provided.  "
                f"Inferred nx={nx}, ny={ny}, nz={nz}, "
                f"nx*ny*nz={nx * ny * nz}, N={n_pts}"
            )

        sort_idx = np.lexsort((x, y, z))
        dx = float(np.mean(np.diff(unique_x))) if nx > 1 else 1.0
        dy = float(np.mean(np.diff(unique_y))) if ny > 1 else 1.0
        dz = float(np.mean(np.diff(unique_z))) if nz > 1 else 1.0

        return {
            "ndim": 3, "nx": nx, "ny": ny, "nz": nz,
            "sort_idx": sort_idx,
            "x_unique": unique_x, "y_unique": unique_y, "z_unique": unique_z,
            "dx": dx, "dy": dy, "dz": dz,
        }

    # ------------------------------------------------------------------
    # 2D grid
    # ------------------------------------------------------------------
    if num_x is not None and num_y is not None:
        nx, ny = int(num_x), int(num_y)
        if nx > 0 and ny > 0 and nx * ny == n_pts:
            sort_idx = np.lexsort((x, y))
            dx = float(np.mean(np.diff(unique_x))) if len(unique_x) > 1 else 1.0
            dy = float(np.mean(np.diff(unique_y))) if len(unique_y) > 1 else 1.0
            return {
                "ndim": 2, "nx": nx, "ny": ny, "nz": None,
                "sort_idx": sort_idx,
                "x_unique": unique_x, "y_unique": unique_y, "z_unique": None,
                "dx": dx, "dy": dy, "dz": None,
            }
        elif nx > 0 and ny > 0:
            print(
                f"[Warning: !] Provided Num_x={nx}, Num_y={ny} are inconsistent with "
                f"N={n_pts}; falling back to coordinate inference."
            )

    nx = len(unique_x)
    ny = len(unique_y)

    if nx * ny != n_pts:
        raise ValueError(
            f"Coordinates do not form a complete structured 2D grid and no valid "
            f"(Num_x, Num_y) was provided.  "
            f"Inferred nx={nx}, ny={ny}, nx*ny={nx * ny}, N={n_pts}"
        )

    sort_idx = np.lexsort((x, y))
    dx = float(np.mean(np.diff(unique_x))) if nx > 1 else 1.0
    dy = float(np.mean(np.diff(unique_y))) if ny > 1 else 1.0

    return {
        "ndim": 2, "nx": nx, "ny": ny, "nz": None,
        "sort_idx": sort_idx,
        "x_unique": unique_x, "y_unique": unique_y, "z_unique": None,
        "dx": dx, "dy": dy, "dz": None,
    }




def _reshape_flat_field_to_grid(field_flat: np.ndarray, grid_info: dict) -> np.ndarray:
    vals = field_flat[grid_info["sort_idx"]]
    if grid_info["ndim"] == 3:
        return vals.reshape(grid_info["nz"], grid_info["ny"], grid_info["nx"])
    return vals.reshape(grid_info["ny"], grid_info["nx"])


# ---------------------------------------------------------------------------
# SSIM (2D / 3D)
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, ndim: int = 2, device: str = "cpu"):
    ax = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    if ndim == 3:
        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    else:
        kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, *kernel.shape)


def _ssim(u: np.ndarray, v: np.ndarray, data_range: Optional[float] = None,
          window_size: int = 11, sigma: float = 1.5) -> float:
    """Single-scale SSIM for one scalar 2D or 3D field."""
    ndim = u.ndim
    device = "cpu"
    x = torch.from_numpy(u).float().to(device)
    y = torch.from_numpy(v).float().to(device)

    # [1, 1, ...spatial dims...]
    for _ in range(2):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    if data_range is None:
        data_range = float(max(u.max(), v.max()) - min(u.min(), v.min()))
    data_range = max(float(data_range), 1e-8)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel(window_size=window_size, sigma=sigma, ndim=ndim, device=device)
    pad = window_size // 2
    conv_fn = F.conv3d if ndim == 3 else F.conv2d

    mu_x = conv_fn(x, kernel, padding=pad)
    mu_y = conv_fn(y, kernel, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = conv_fn(x * x, kernel, padding=pad) - mu_x2
    sigma_y2 = conv_fn(y * y, kernel, padding=pad) - mu_y2
    sigma_xy = conv_fn(x * y, kernel, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12
    )
    return float(ssim_map.mean().item())




# ---------------------------------------------------------------------------
# Gradient metrics (2D / 3D)
# ---------------------------------------------------------------------------

def _gradient_metrics(
    u: np.ndarray, v: np.ndarray,
    dx: float, dy: float, dz: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Gradient-based metrics using finite differences with physical spacing."""

    if u.ndim == 3 and dz is not None:
        # 3D gradients: np.gradient returns (grad_z, grad_y, grad_x) for shape (nz, ny, nx)
        uz, uy, ux = np.gradient(u, dz, dy, dx, edge_order=2)
        vz, vy, vx = np.gradient(v, dz, dy, dx, edge_order=2)

        diff_x = vx - ux
        diff_y = vy - uy
        diff_z = vz - uz

        sq_diff = diff_x ** 2 + diff_y ** 2 + diff_z ** 2
        sq_true = ux ** 2 + uy ** 2 + uz ** 2

        grad_mse = float(np.mean(sq_diff))
        grad_rel_l2 = float(np.sqrt(np.sum(sq_diff)) / (np.sqrt(np.sum(sq_true)) + 1e-12))

        val_diff = v - u
        h1_num = np.sum(val_diff ** 2) + np.sum(sq_diff)
        h1_den = np.sum(u ** 2) + np.sum(sq_true)
        h1_rel = float(np.sqrt(h1_num) / (np.sqrt(h1_den) + 1e-12))

        metrics = {"grad_mse": grad_mse, "grad_rel_l2": grad_rel_l2, "h1_rel": h1_rel}
        payload = {
            "grad_true_x": ux, "grad_true_y": uy, "grad_true_z": uz,
            "grad_pred_x": vx, "grad_pred_y": vy, "grad_pred_z": vz,
            "grad_abs_err": np.sqrt(sq_diff),
        }
        return metrics, payload

    # 2D fallback
    uy, ux = np.gradient(u, dy, dx, edge_order=2)
    vy, vx = np.gradient(v, dy, dx, edge_order=2)

    diff_x = vx - ux
    diff_y = vy - uy

    grad_mse = float(np.mean(diff_x ** 2 + diff_y ** 2))
    grad_rel_l2 = float(
        np.sqrt(np.sum(diff_x ** 2 + diff_y ** 2)) /
        (np.sqrt(np.sum(ux ** 2 + uy ** 2)) + 1e-12)
    )

    val_diff = v - u
    h1_num = np.sum(val_diff ** 2) + np.sum(diff_x ** 2 + diff_y ** 2)
    h1_den = np.sum(u ** 2) + np.sum(ux ** 2 + uy ** 2)
    h1_rel = float(np.sqrt(h1_num) / (np.sqrt(h1_den) + 1e-12))

    metrics = {"grad_mse": grad_mse, "grad_rel_l2": grad_rel_l2, "h1_rel": h1_rel}
    payload = {
        "grad_true_x": ux, "grad_true_y": uy,
        "grad_pred_x": vx, "grad_pred_y": vy,
        "grad_abs_err": np.sqrt(diff_x ** 2 + diff_y ** 2),
    }
    return metrics, payload


# ---------------------------------------------------------------------------
# Radial power spectrum (2D / 3D)
# ---------------------------------------------------------------------------

def _radial_spectrum(u: np.ndarray, dx: float, dy: float, dz: Optional[float] = None):
    """
    Shell-averaged radial power spectrum of a zero-mean field.
    Supports both 2D (ny, nx) and 3D (nz, ny, nx) arrays.
    """

    if u.ndim == 3 and dz is not None:
        nz, ny, nx = u.shape
        uu = u - np.mean(u)

        fft = np.fft.fftshift(np.fft.fftn(uu))
        psd = (np.abs(fft) ** 2) / (nx * ny * nz)

        kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
        kz = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nz, d=dz))

        # meshgrid with 'ij' indexing so shapes are (nz, ny, nx) matching the FFT
        KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
        kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

        dkx = np.min(np.abs(np.diff(np.unique(kx)))) if nx > 1 else 1.0
        dky = np.min(np.abs(np.diff(np.unique(ky)))) if ny > 1 else 1.0
        dkz = np.min(np.abs(np.diff(np.unique(kz)))) if nz > 1 else 1.0
        dk = float(min(abs(dkx), abs(dky), abs(dkz)))
        dk = max(dk, 1e-12)

        shell_id = np.rint(kmag / dk).astype(np.int64)
        n_shells = int(shell_id.max()) + 1

        shell_sum = np.bincount(shell_id.ravel(), weights=psd.ravel(), minlength=n_shells)
        shell_count = np.bincount(shell_id.ravel(), minlength=n_shells)
        shell_k_sum = np.bincount(shell_id.ravel(), weights=kmag.ravel(), minlength=n_shells)

        radial = shell_sum / np.maximum(shell_count, 1)
        k = shell_k_sum / np.maximum(shell_count, 1)

        valid = shell_count > 0
        k = k[valid]
        radial = radial[valid]

        if len(k) > 1:
            k = k[1:]
            radial = radial[1:]

        # Truncate at the per-axis Nyquist: corner shells (|k| up to
        # sqrt(3) x Nyquist) are populated by a handful of modes with
        # essentially zero physical energy, and band ratios computed over
        # them are spurious.
        k_nyq = np.pi / max(float(dx), float(dy), float(dz))
        keep = k <= k_nyq
        return {"k": k[keep], "psd": psd, "radial_spectrum": radial[keep]}

    # 2D path
    ny, nx = u.shape
    uu = u - np.mean(u)

    fft = np.fft.fftshift(np.fft.fft2(uu))
    psd = (np.abs(fft) ** 2) / (nx * ny)

    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    kmag = np.sqrt(KX ** 2 + KY ** 2)

    dkx = np.min(np.diff(np.unique(kx))) if nx > 1 else 1.0
    dky = np.min(np.diff(np.unique(ky))) if ny > 1 else 1.0
    dk = float(min(abs(dkx), abs(dky))) if (nx > 1 and ny > 1) else 1.0
    dk = max(dk, 1e-12)

    shell_id = np.rint(kmag / dk).astype(np.int64)
    n_shells = int(shell_id.max()) + 1

    shell_sum = np.bincount(shell_id.ravel(), weights=psd.ravel(), minlength=n_shells)
    shell_count = np.bincount(shell_id.ravel(), minlength=n_shells)
    shell_k_sum = np.bincount(shell_id.ravel(), weights=kmag.ravel(), minlength=n_shells)

    radial = shell_sum / np.maximum(shell_count, 1)
    k = shell_k_sum / np.maximum(shell_count, 1)

    valid = shell_count > 0
    k = k[valid]
    radial = radial[valid]

    if len(k) > 1:
        k = k[1:]
        radial = radial[1:]

    # Same Nyquist truncation as the 3D path (corner shells are spurious).
    k_nyq = np.pi / max(float(dx), float(dy))
    keep = k <= k_nyq
    return {"k": k[keep], "psd": psd, "radial_spectrum": radial[keep]}


def _band_energy_breakdown(k: np.ndarray, spectrum: np.ndarray):
    """Split radial spectrum into low / mid / high wavenumber bands."""
    if len(k) == 0:
        return {
            "band_names": ["large", "medium", "small"],
            "band_edges": [0.0, 0.0, 0.0, 0.0],
            "band_energy": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "band_fraction": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        }

    kmax = float(np.max(k))
    e1 = kmax / 3.0
    e2 = 2.0 * kmax / 3.0

    masks = [
        (k <= e1),
        ((k > e1) & (k <= e2)),
        (k > e2),
    ]

    energies = []
    for mask in masks:
        if np.count_nonzero(mask) >= 2:
            energies.append(float(np.trapezoid(spectrum[mask], k[mask])))
        elif np.count_nonzero(mask) == 1:
            energies.append(float(spectrum[mask][0]))
        else:
            energies.append(0.0)

    energies = np.asarray(energies, dtype=np.float64)
    total = float(np.sum(energies)) + 1e-12
    fractions = energies / total

    return {
        "band_names": ["large", "medium", "small"],
        "band_edges": [0.0, e1, e2, kmax],
        "band_energy": energies,
        "band_fraction": fractions,
    }


def _spectral_metrics(u: np.ndarray, v: np.ndarray, dx: float, dy: float,
                      dz: Optional[float] = None):
    """Spectral comparison using shell-averaged radial spectra (2D or 3D)."""
    su = _radial_spectrum(u, dx=dx, dy=dy, dz=dz)
    sv = _radial_spectrum(v, dx=dx, dy=dy, dz=dz)

    eps = 1e-12
    k = su["k"]
    ru = su["radial_spectrum"]
    rv = sv["radial_spectrum"]

    n = min(len(ru), len(rv))
    k = k[:n]
    ru = ru[:n]
    rv = rv[:n]

    spectral_lsd = float(np.sqrt(np.mean((np.log(rv + eps) - np.log(ru + eps)) ** 2)))

    gt_band = _band_energy_breakdown(k, ru)
    pr_band = _band_energy_breakdown(k, rv)

    band_ratio = pr_band["band_energy"] / (gt_band["band_energy"] + eps)
    band_rel_err = np.abs(pr_band["band_energy"] - gt_band["band_energy"]) / (gt_band["band_energy"] + eps)

    metrics = {
        "spectral_lsd": spectral_lsd,
        "spectral_ratio_large": float(band_ratio[0]),
        "spectral_ratio_medium": float(band_ratio[1]),
        "spectral_ratio_small": float(band_ratio[2]),
        "spectral_relerr_large": float(band_rel_err[0]),
        "spectral_relerr_medium": float(band_rel_err[1]),
        "spectral_relerr_small": float(band_rel_err[2]),
    }

    payload = {
        "k": k,
        "spectrum_true": ru,
        "spectrum_pred": rv,
        "psd_true": su["psd"],
        "psd_pred": sv["psd"],
        "band_names": np.array(gt_band["band_names"]),
        "band_edges": np.array(gt_band["band_edges"], dtype=np.float64),
        "band_energy_true": gt_band["band_energy"],
        "band_energy_pred": pr_band["band_energy"],
        "band_fraction_true": gt_band["band_fraction"],
        "band_fraction_pred": pr_band["band_fraction"],
        "band_ratio_pred_over_true": band_ratio,
    }
    return metrics, payload


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_spectrum_plot(
    k: np.ndarray,
    s_true: np.ndarray,
    s_pred: np.ndarray,
    band_edges: np.ndarray,
    save_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.axvspan(band_edges[0], band_edges[1], color="#c9c9f5", alpha=0.25)
    ax.axvspan(band_edges[1], band_edges[2], color="#cfe8cf", alpha=0.25)
    ax.axvspan(band_edges[2], band_edges[3], color="#f3d6d6", alpha=0.25)

    ax.semilogy(k + 1e-12, s_true + 1e-12, color="black", linewidth=2.5, label="Ground Truth")
    ax.semilogy(k + 1e-12, s_pred + 1e-12, color="red", linewidth=2.2, label="Reconstruction")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)

    ymax = max(np.max(s_true), np.max(s_pred)) + 1e-12
    ymin = max(min(np.min(s_true[s_true > 0]) if np.any(s_true > 0) else 1e-12,
                   np.min(s_pred[s_pred > 0]) if np.any(s_pred > 0) else 1e-12), 1e-12)
    ax.set_ylim(bottom=ymin * 0.7, top=ymax * 1.25)

    ax.text(0.5 * (band_edges[0] + band_edges[1]), ymin * 1.2, "large scales",
            color="#303080", fontsize=12, ha="center", va="bottom", fontstyle="italic")
    ax.text(0.5 * (band_edges[1] + band_edges[2]), ymin * 1.2, "medium scales",
            color="#2f6f2f", fontsize=12, ha="center", va="bottom", fontstyle="italic")
    ax.text(0.5 * (band_edges[2] + band_edges[3]), ymin * 1.2, "small scales",
            color="#7a3030", fontsize=12, ha="center", va="bottom", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _save_band_energy_plot(
    band_names: np.ndarray,
    band_ratio: np.ndarray,
    save_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(band_names))

    ax.bar(x, band_ratio, width=0.6)
    ax.axhline(1.0, linestyle=":", linewidth=1.8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v).capitalize() for v in band_names])
    ax.set_ylabel(r"$E_{\mathrm{pred}} / E_{\mathrm{GT}}$")
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)




def _save_3d_slice_plots(
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    grid_info: dict,
    field_name: str,
    save_dir: Path,
    prefix: str,
    dpi: int = 300,
    cmap_field: str = "coolwarm",
    cmap_err: str = "inferno",
    contour_levels: Optional[int] = 20,
    contour_linewidth: float = 0.5,
    contour_alpha: float = 0.5,
):
    """
    Save three orthogonal mid-plane slices (XY, XZ, YZ) for a single 3D field.
    Each slice is a 3-row panel: Ground Truth | Reconstruction | |Error|.
    """
    nz, ny, nx = true_grid.shape
    slices = {
        "xy": {
            "idx": nz // 2,
            "true": true_grid[nz // 2, :, :],
            "pred": pred_grid[nz // 2, :, :],
            "xlabel": "x", "ylabel": "y",
            "extent": [
                float(grid_info["x_unique"][0]), float(grid_info["x_unique"][-1]),
                float(grid_info["y_unique"][0]), float(grid_info["y_unique"][-1]),
            ],
            "x_coords": grid_info["x_unique"],
            "y_coords": grid_info["y_unique"],
            "label": f"z={float(grid_info['z_unique'][nz // 2]):.4g}",
        },
        "xz": {
            "idx": ny // 2,
            "true": true_grid[:, ny // 2, :],
            "pred": pred_grid[:, ny // 2, :],
            "xlabel": "x", "ylabel": "z",
            "extent": [
                float(grid_info["x_unique"][0]), float(grid_info["x_unique"][-1]),
                float(grid_info["z_unique"][0]), float(grid_info["z_unique"][-1]),
            ],
            "x_coords": grid_info["x_unique"],
            "y_coords": grid_info["z_unique"],
            "label": f"y={float(grid_info['y_unique'][ny // 2]):.4g}",
        },
        "yz": {
            "idx": nx // 2,
            "true": true_grid[:, :, nx // 2],
            "pred": pred_grid[:, :, nx // 2],
            "xlabel": "y", "ylabel": "z",
            "extent": [
                float(grid_info["y_unique"][0]), float(grid_info["y_unique"][-1]),
                float(grid_info["z_unique"][0]), float(grid_info["z_unique"][-1]),
            ],
            "x_coords": grid_info["y_unique"],
            "y_coords": grid_info["z_unique"],
            "label": f"x={float(grid_info['x_unique'][nx // 2]):.4g}",
        },
    }

    for plane_name, s in slices.items():
        u = s["true"]
        v = s["pred"]
        err = np.abs(v - u)
        l2 = float(np.linalg.norm(u.ravel() - v.ravel()) / (np.linalg.norm(u.ravel()) + 1e-8))

        field_min = float(min(u.min(), v.min()))
        field_max = float(max(u.max(), v.max()))
        err_pos = err[err > 0]
        err_min = float(err_pos.min()) if err_pos.size > 0 else 0.0
        err_max = float(err.max()) if err.size > 0 else 1.0

        fig, axes = plt.subplots(3, 1, figsize=(7.5, 12))

        im0 = axes[0].imshow(u, origin="lower", extent=s["extent"], aspect="equal",
                              cmap=cmap_field, vmin=field_min, vmax=field_max)
        if contour_levels is not None:
            axes[0].contour(
                s["x_coords"], s["y_coords"], u,
                levels=contour_levels, colors="white",
                linewidths=contour_linewidth, alpha=contour_alpha,
            )
        axes[0].set_title(f"Ground Truth  ({s['label']})")

        im1 = axes[1].imshow(v, origin="lower", extent=s["extent"], aspect="equal",
                              cmap=cmap_field, vmin=field_min, vmax=field_max)
        if contour_levels is not None:
            axes[1].contour(
                s["x_coords"], s["y_coords"], v,
                levels=contour_levels, colors="white",
                linewidths=contour_linewidth, alpha=contour_alpha,
            )
        axes[1].set_title(f"Reconstruction  ({s['label']})")

        im2 = axes[2].imshow(err, origin="lower", extent=s["extent"], aspect="equal",
                              cmap=cmap_err, vmin=err_min, vmax=err_max)
        axes[2].set_title(f"|Error|  ({s['label']})")

        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(im0, ax=axes[0], shrink=0.7, pad=0.02)
        fig.colorbar(im1, ax=axes[1], shrink=0.7, pad=0.02)
        fig.colorbar(im2, ax=axes[2], shrink=0.7, pad=0.02)

        fig.suptitle(
            f"{field_name}  |  {plane_name.upper()} slice  |  L2={l2:.3e}",
            y=0.98, fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = save_dir / f"{prefix}_field_{field_name}_slice_{plane_name}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    demo_root = Path(args.demo_root).resolve()
    cfg_dir = demo_root / "configs" / "pointcloud_ffm"

    try:
        yaml_path = _find_latest_yaml(cfg_dir, args.Demo_Num)
    except FileNotFoundError as e:
        print(f"[Warning: !] {e}")
        raise SystemExit(1)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _normalize_eval_config(cfg)

    train_timestamp = _extract_timestamp(yaml_path)
    if train_timestamp is None:
        print(f"[Warning: !] Could not parse timestamp from config filename: {yaml_path.name}")
        raise SystemExit(1)

    save_dir_cfg = Path(cfg.get("save_dir", "outputs/runs/ffm_tc_pointcloud"))
    model_root = demo_root / save_dir_cfg.parent / f"{save_dir_cfg.name}_DemoN{args.Demo_Num}_{train_timestamp}"

    if not model_root.exists():
        print(f"[Warning: !] Matching model directory not found: {model_root}")
        raise SystemExit(1)

    ckpt_path = model_root / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        print(f"[Warning: !] Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    device = torch.device(args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    dataset = TurbulentCombustionH5Dataset(
        cfg.get("data", f"{DATA_DIR}/Merged_CH4COTU1P.h5"),
        split=args.split,
        train_ratio=cfg.get("train_ratio", 0.9),
        seed=cfg.get("seed", 42),
        time_stride=cfg.get("time_stride", 1),
        field_names=tuple(cfg.get("field_names", ["CH4", "CO", "T", "U_1", "p"])),
        stats_path=str(model_root / "dataset_stats.pt"),
    )

    try:
        model = _build_model(cfg, dataset)
    except Exception as e:
        print(f"[Warning: !] Model construction failed: {e}")
        raise SystemExit(1)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except pickle.UnpicklingError:
        print("[Warning: !] Restricted torch.load failed; retrying with weights_only=False for a trusted local checkpoint.")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("_metadata", None)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[Warning: !] Checkpoint is incompatible with the reconstructed model: {e}")
        raise SystemExit(1)

    epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    del state_dict
    del ckpt

    model = model.to(device)
    model.eval()

    vis_cond_fields = args.vis_cond_fields if args.vis_cond_fields is not None else cfg["vis_cond_fields"]
    vis_n_obs_list = args.vis_n_obs_list if args.vis_n_obs_list is not None else cfg["vis_n_obs_list"]

    print(f'\nvis_n_obs_list is {vis_n_obs_list}\n')
    
    n_steps_generation = (
        args.n_steps_generation if args.n_steps_generation is not None
        else cfg.get("n_steps_generation", 100)
    )
    print(f'\nResults are generated from n_steps={n_steps_generation}\n')

    from datetime import datetime
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = model_root / "Evaluation" / f"eval_N{args.Demo_Num}_{eval_timestamp}_from_{train_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    need_extra = (len(args.extra_metrics) > 0) or args.save_analysis_npz
    prefix = f"snapshot_{args.snapshot_index:04d}"
    metrics_by_mode = None

    # ------------------------------------------------------------------
    # Detect dimensionality from the dataset coordinates
    # ------------------------------------------------------------------
    _sample0 = dataset[0]
    _coords_raw = _sample0["coords_raw"].numpy()
    _coord_dim = _coords_raw.shape[1]
    _unique_z = np.unique(np.round(_coords_raw[:, 2], 6)) if _coord_dim >= 3 else np.array([0.0])
    is_3d = len(_unique_z) > 1
    del _sample0, _coords_raw, _unique_z

    field_names = list(getattr(dataset, "field_names",
                               cfg.get("field_names", [f"f{i}" for i in range(dataset.num_fields)])))

    # ------------------------------------------------------------------
    # 3D evaluation path
    # ------------------------------------------------------------------
    if is_3d:
        print("[*] Detected 3D grid; using 3D evaluation path.\n")

        recon_result = reconstruct_snapshot(
            model=model,
            dataset=dataset,
            device=device,
            snapshot_index=args.snapshot_index,
            cond_fields=vis_cond_fields,
            n_obs_list=vis_n_obs_list,
            n_steps=n_steps_generation,
        )

        mean = dataset.mean.to(device)
        std = dataset.std.to(device)
        truth_phys = (recon_result["truth"] * std.view(1, 1, -1) + mean.view(1, 1, -1))[0].cpu().numpy()
        recon_phys = (recon_result["recon"] * std.view(1, 1, -1) + mean.view(1, 1, -1))[0].cpu().numpy()

        sample_data = dataset[args.snapshot_index]
        coords_raw = sample_data["coords_raw"].numpy()

        metrics = {}
        for c, name in enumerate(field_names):
            t_f = truth_phys[:, c]
            p_f = recon_phys[:, c]
            metrics[name] = float(np.linalg.norm(t_f - p_f) / (np.linalg.norm(t_f) + 1e-8))

        # Save basic metrics JSON
        basic_metrics_path = out_dir / f"{prefix}_metrics.json"
        with open(basic_metrics_path, "w") as f:
            json.dump({
                "epoch": int(epoch),
                "snapshot_index": int(args.snapshot_index),
                "cond_fields": [int(v) for v in vis_cond_fields],
                "n_obs": [int(v) for v in vis_n_obs_list],
                "n_steps": int(n_steps_generation),
                "metrics": metrics,
            }, f, indent=2)

        # Infer 3D grid and save slice visualizations
        try:
            grid_info = _infer_structured_grid(
                coords_raw,
                num_x=cfg.get("Num_x", None),
                num_y=cfg.get("Num_y", None),
                num_z=cfg.get("Num_z", None),
            )
            print(f"[*] Inferred 3D grid: nx={grid_info['nx']}, ny={grid_info['ny']}, nz={grid_info['nz']}")

            for c, name in enumerate(field_names):
                true_grid = _reshape_flat_field_to_grid(truth_phys[:, c], grid_info)
                pred_grid = _reshape_flat_field_to_grid(recon_phys[:, c], grid_info)
                _save_3d_slice_plots(
                    true_grid, pred_grid, grid_info,
                    field_name=name, save_dir=out_dir, prefix=prefix,
                )
        except ValueError as e:
            print(f"[Warning: !] 3D grid inference failed; slice plots and grid metrics skipped: {e}")
            grid_info = None

    # ------------------------------------------------------------------
    # 2D evaluation path (existing behavior)
    # ------------------------------------------------------------------
    else:
        print("[*] Detected 2D grid; using standard evaluation path.\n")
        grid_info = None

        result = visualize_reconstruction(
            model=model,
            dataset=dataset,
            epoch=epoch,
            device=device,
            save_dir=str(out_dir),
            cond_fields=vis_cond_fields,
            n_obs=vis_n_obs_list,
            n_steps=n_steps_generation,
            snapshot_index=args.snapshot_index,
            file_tag=prefix,
            save_metrics_json=True,
            return_payload=need_extra,
        )

        if need_extra:
            metrics, payload_2d = result
            # Retrieve physical-unit arrays from the 2D payload
            coords_raw = None
            truth_phys = payload_2d["truth_phys"]
            recon_phys = payload_2d["recon_phys"]
            field_names = payload_2d["field_names"]

            # For grid inference, use 2D coordinates
            try:
                grid_info = _infer_structured_grid(
                    payload_2d["coords_xy"],
                    num_x=cfg.get("Num_x", None),
                    num_y=cfg.get("Num_y", None),
                )
            except ValueError as e:
                print(f"[Warning: !] 2D grid inference failed; extra metrics skipped: {e}")
                grid_info = None
        else:
            metrics = result

    # ------------------------------------------------------------------
    # Extra structured-grid metrics (unified 2D / 3D)
    # ------------------------------------------------------------------
    extra_metrics = {}

    if need_extra and grid_info is not None:
        ndim = grid_info["ndim"]
        dz = grid_info["dz"]

        for c, name in enumerate(field_names):
            u = _reshape_flat_field_to_grid(truth_phys[:, c], grid_info)
            v = _reshape_flat_field_to_grid(recon_phys[:, c], grid_info)

            field_metrics: Dict[str, float] = {}
            analysis_payload: Dict[str, np.ndarray] = {
                "true_grid": u,
                "pred_grid": v,
                "abs_err_grid": np.abs(v - u),
                "x_unique": grid_info["x_unique"],
                "y_unique": grid_info["y_unique"],
            }
            if grid_info["z_unique"] is not None:
                analysis_payload["z_unique"] = grid_info["z_unique"]

            if "ssim" in args.extra_metrics:
                field_metrics["ssim"] = _ssim(
                    u, v,
                    data_range=float(u.max() - u.min())
                )

            if "grad" in args.extra_metrics:
                grad_metrics, grad_payload = _gradient_metrics(
                    u, v,
                    dx=grid_info["dx"],
                    dy=grid_info["dy"],
                    dz=dz,
                )
                field_metrics.update(grad_metrics)
                analysis_payload.update(grad_payload)

            if "spectrum" in args.extra_metrics:
                spec_metrics, spec_payload = _spectral_metrics(
                    u, v,
                    dx=grid_info["dx"],
                    dy=grid_info["dy"],
                    dz=dz,
                )
                field_metrics.update(spec_metrics)
                analysis_payload.update(spec_payload)

                spec_plot_path = out_dir / f"{prefix}_field_{name}_spectrum.png"
                _save_spectrum_plot(
                    spec_payload["k"],
                    spec_payload["spectrum_true"],
                    spec_payload["spectrum_pred"],
                    band_edges=spec_payload["band_edges"],
                    save_path=spec_plot_path,
                    title=f"{name} spectrum",
                )
                band_plot_path = out_dir / f"{prefix}_field_{name}_band_energy_ratio.png"
                _save_band_energy_plot(
                    spec_payload["band_names"],
                    spec_payload["band_ratio_pred_over_true"],
                    save_path=band_plot_path,
                    title=f"{name} band energy ratio",
                )

            extra_metrics[name] = field_metrics

            if args.save_analysis_npz:
                npz_path = out_dir / f"{prefix}_field_{name}_analysis.npz"
                np.savez_compressed(npz_path, **analysis_payload)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "demo_num": int(args.Demo_Num),
        "yaml_path": str(yaml_path),
        "model_root": str(model_root),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "snapshot_index": int(args.snapshot_index),
        "vis_cond_fields": [int(v) for v in vis_cond_fields],
        "vis_n_obs_list": [int(v) for v in vis_n_obs_list],
        "n_steps_generation": int(n_steps_generation),
        "is_3d": is_3d,
        "metrics": metrics,
        "metrics_by_mode": metrics_by_mode,
        "extra_metric_names": list(args.extra_metrics),
        "extra_metrics": extra_metrics,
    }

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[*] Evaluation finished.")
    print(f"[*] YAML      : {yaml_path}")
    print(f"[*] Checkpoint: {ckpt_path}")
    print(f"[*] Output dir : {out_dir}")
    print(f"[*] Metrics         : {json.dumps(metrics, indent=2)}")
    if len(extra_metrics) > 0:
        print(f"[*] Extra metrics   : {json.dumps(extra_metrics, indent=2)}")


if __name__ == "__main__":
    main()
