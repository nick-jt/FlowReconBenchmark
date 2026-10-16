
'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training

- Model backbone can be ConditionalPointMLPRBF, ConditionalPointPerceiver
'''
from paths import REPO_ROOT, DATA_DIR

import argparse
import json
import math
import os
import shutil
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from methods.dmfgen.dataset_shiftwing import ShiftWingDataset, collate_wing
from methods.dmfgen.helpers import MetricsLogger, TurbulentCombustionH5Dataset, append_extra_tokens, build_sparse_condition, build_sparse_condition_from_pool, create_recon_dir, validate_regular_grid_compatibility, visualize_reconstruction
from methods.dmfgen.measurement_ops import apply_measurement_operators
from protocol import VAL_FIXED_SEED, fixed_validation_draw, log_param_count, resume_best_val
from methods.dmfgen.Model import FNO, FNOFFM, ConditionalPointHybridLocalGlobalRBF, ConditionalPointMLPRBF, ConditionalPointPerceiver, PointCloudFFM
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args():

    p = argparse.ArgumentParser("Train a starter conditional point-cloud FFM on turbulent combustion HDF5 data.")

    p.add_argument("--config", type=str, 
                   default="configs/config_pointcloud_ffm.yaml", help="Path to YAML config")
    p.add_argument("--Demo-Num", type=int, 
                   default=0, help="Demo ID tag for saving directories")
    p.add_argument("--device-ids", type=int, nargs="+", default=[0])

    p.add_argument("--data", type=str, 
                   default=f"{DATA_DIR}/Merged_CH4COTU1P.h5")
    p.add_argument("--save-dir", type=str, 
                   default="outputs/runs/ffm_tc_pointcloud")
    p.add_argument("--field-names", type=str, nargs="+", default=["CH4", "CO", "T", "U_1", "p"])
    p.add_argument("--RELOAD", action="store_true",
                   help="If set, try to reload the latest matching checkpoint and continue training.")


    # ------------------------------
    # Backbone selection
    # ------------------------------
    p.add_argument(
        "--backbone", type=str, default="mlp_rbf", choices = ["mlp_rbf", "perceiver", "fno", "fno3d", "GL_rbf", "GL_rbf_ENH"], 
        help="Backbone type. point-cloud MLP+RBF, point-cloud Perceiver, or grid-based FNO baseline.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    # ------------------------------
    # These are hyperparameters for mlp_rbf backbone or part of GL_rbf
    # ------------------------------
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--field-embed-dim", type=int, default=64)
    p.add_argument("--rbf-sigma", type=float, default=0.05)
    p.add_argument("--USE-FOURIER-PE", "--USE_FOURIER_PE", dest="USE_FOURIER_PE", action="store_true",
                   help="If set, feed Fourier positional coordinate features to point_encoder.")
    p.add_argument("--fourier-pe-num-bands", type=int, default=32,
                   help="Number of frequency bands for Fourier positional coordinate encoding.")
    p.add_argument("--fourier-pe-max-freq", type=float, default=64.0,
                   help="Maximum frequency scale for Fourier positional coordinate encoding.")

    # ------------------------------
    # These are hyperparameters for Perceiver backbone or part of GL_rbf
    # ------------------------------
    p.add_argument("--latent-dim", type=int, default=256, 
                   help="Token / latent width for the Perceiver backbone.",)
    p.add_argument("--num-latents", type=int, default=128, 
                   help="Number of learned latent slots in the Perceiver.",)
    p.add_argument("--num-heads", type=int, default=8, 
                   help="Number of attention heads for Perceiver attention blocks.",)
    p.add_argument("--num-latent-blocks", type=int, default=4, 
                   help="Number of latent self-attention blocks.",)
    p.add_argument("--ff-mult", type=int, default=4, 
                   help="Expansion factor for Transformer feed-forward layers.",)
    p.add_argument("--attn-dropout", type=float, default=0.0, 
                   help="Dropout used inside attention layers.",)
    p.add_argument("--mlp-dropout", type=float, default=0.0, 
                   help="Dropout used inside token projection / FFN layers.",)
    p.add_argument("--decode-chunk-size", type=int, default=4096,
                   help="Chunk size for Perceiver output decoding. Useful for full-resolution reconstruction.",)
    p.add_argument("--share-query-proj", action="store_true",
        help="If set, use the same projection for Perceiver encoder query tokens and decoder query tokens.",)

    p.add_argument("--summary-type", type=str, default='cls',
        help="Only for GL_rbf; select either cls or mean",)

    # ----------------------------------------------------------
    # Hybrid local-global gather options
    # ----------------------------------------------------------
    p.add_argument(
        "--gather-mode", type=str, default="rbf", choices=["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"],
        help="Gather mode used by ConditionalPointHybridLocalGlobalRBF. 'rbf' preserves the current full gather as default.",
    )
    p.add_argument(
        "--gather-topk", type=int, default=32, 
        help="Number of nearest refined sensor tokens used in top-k gather modes.",
    )
    p.add_argument(
        "--spectral-weight", type=float, default=0.0,
        help="Weight on the binned spectral power loss (0 disables it).",
    )
    p.add_argument(
        "--spectral-block", type=int, default=32,
        help="Edge length of the contiguous grid block used for the spectral loss.",
    )
    p.add_argument(
        "--spectral-bins", type=int, default=12,
        help="Number of radial wavenumber shells in the spectral loss.",
    )
    p.add_argument(
        "--gather-multiscale", type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=False,
        help="Add a coarser second gather scale (zero-init residual branch).",
    )
    p.add_argument(
        "--gather-topk-coarse", type=int, default=64,
        help="Neighbors for the coarse gather scale when --gather-multiscale.",
    )
    p.add_argument(
        "--gather-coarse-sigma-scale", type=float, default=4.0,
        help="Multiplier on rbf_sigma for the coarse gather scale.",
    )
    p.add_argument(
        "--gather-query-chunk-size", type=int, default=None,
        help="Optional query chunk size for memory-friendly gathering. Applies to all gather modes.",
    )
    p.add_argument(
        "--learnable-rbf-sigma", action="store_true",
        help="If set, make the RBF sigma in the hybrid gather learnable.",
    )
    p.add_argument(
        "--neighbor-backend", type=str, default="torch", choices=["auto", "torch", "keops"],
        help="Neighbor / kernel backend for the hybrid gather. "
            "'auto' uses KeOps if available, otherwise falls back to pure PyTorch.",)
    p.add_argument(
        "--sensor-local-topk", type=int, default=8,
        help="Number of local sensor neighbors used by the sensor-side Point-Transformer refinement in gather_mode='topk_rbf_ptlocal'.",)
    p.add_argument(
        "--sensor-local-dropout", type=float, default=0.0,
        help="Dropout used inside the sensor-side local refinement block for gather_mode='topk_rbf_ptlocal'.",
    )
    p.add_argument("--sensor-coord-encoding", type=str, default=None,
                   choices=["raw", "fourier"],
                   help="Sensor coordinate encoding for GL_rbf/GL_rbf_ENH. "
                        "Use 'fourier' to give sensors the same coordinate features as queries.")
    p.add_argument("--latent-sensor-reinject", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, latents periodically re-attend to sparse sensor tokens.")
    p.add_argument("--latent-reinject-every", type=int, default=1,
                   help="Re-inject sensor information every N latent blocks when latent_sensor_reinject is enabled.")
    p.add_argument("--query-latent-readout", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, each query reads global context from latent memory before the final head.")
    p.add_argument("--query-readout-type", type=str, default=None,
                   choices=["point", "coord"],
                   help="'coord' uses Senseiver-style coordinate decoder tokens; "
                        "'point' uses the current flow-state point features.")
    p.add_argument("--query-readout-scale-init", type=float, default=None,
                   help="Initial scale for query-to-latent readout. "
                        "Use small positive values such as 1e-2 for GL_rbf_ENH.")
    p.add_argument("--enhanced-head-norm", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, apply LayerNorm to the fused [query, global, local] head input.")
    p.add_argument("--glres-scale-init", type=float, default=None,
                   help="Initial scale for topk_rbf_glres residual terms: sensor importance and coarse scaffold.")

    # ----------------------------------------------------------
    # These are hyperparameters for fno backbone
    # Num_x / Num_y must be supplied for the FNO baseline.
    # ----------------------------------------------------------
    p.add_argument( "--Num-x", dest="Num_x", type=int, default=None,
        help="Number of grid points along x for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument("--Num-y", dest="Num_y", type=int, default=None,
        help="Number of grid points along y for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument("--Num-z", dest="Num_z", type=int, default=None,
        help="Number of grid points along z. Required when backbone='fno3d'.",)
    p.add_argument( "--fno-modes-z", type=int, default=16,
        help="Retained Fourier modes along z for the 3-D FNO backbone (neuralop total-mode convention).",)
    p.add_argument("--fno-domain-padding", type=float, default=None,
        help="Optional upstream FNO domain padding fraction (None = upstream default).",)
    p.add_argument( "--fno-modes-x", type=int, default=32,
        help="Number of retained Fourier modes along x for the FNO baseline.",)
    p.add_argument( "--fno-modes-y", type=int, default=8,
        help="Number of retained Fourier modes along y for the FNO baseline.",)
    p.add_argument( "--fno-hidden-channels", type=int, default=64,
        help="Hidden channel width of the neuraloperator FNO baseline.",)
    p.add_argument( "--fno-n-layers", type=int, default=4,
        help="Number of Fourier layers in the FNO baseline.",)
    p.add_argument(
        "--condition-blur",
        action="store_true",
        help="If set, Gaussian-splat sparse FNO conditioning maps before concatenation.",
    )
    p.add_argument(
        "--condition-blur-kernel",
        type=int,
        default=5,
        help="Odd Gaussian kernel size used to splat sparse FNO conditioning maps.",
    )
    p.add_argument(
        "--condition-blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used to splat sparse FNO conditioning maps.",
    )

    # ------------------------------
    # These are hyperparameters for training process
    # ------------------------------
    p.add_argument("--n-query-points", type=int, default=4096)
    p.add_argument("--query-sampling", type=str, default="uniform", choices=["uniform", "obs_mix"])
    p.add_argument("--query-sample-near-ratio", type=float, default=0.25)
    p.add_argument("--query-sample-far-ratio", type=float, default=0.25)
    p.add_argument("--query-sample-sigma-ratio", type=float, default=0.05)
    p.add_argument("--prior", type=str, default="rff",
                   choices=["iid", "rff", "rff_powerlaw", "rff_kolmogorov"])
    p.add_argument("--prior-slope", type=float, default=5.0 / 3.0,
                   help="power-law source: E(k) ~ k^-slope (5/3 = Kolmogorov, 0 = band-limited white)")
    p.add_argument("--prior-k-min", type=float, default=1.0)
    p.add_argument("--prior-k-max", type=float, default=48.0)
    p.add_argument("--spectral-window", action="store_true",
                   help="Hann-taper the spectral-loss block (the block is non-periodic; "
                        "without this its top shells match edge leakage, not physics)")
    p.add_argument("--rff-features", type=int, default=256)
    p.add_argument("--rff-lengthscale", type=float, default=0.15)
    p.add_argument("--sigma-min", type=float, default=1e-4) # backward-compatible old args

    p.add_argument("--cond-field", type=int, default=2, help="Legacy single conditioned field.")
    p.add_argument("--n-obs-min", type=int, default=64, help="Legacy single-field minimum sensors.")
    p.add_argument("--n-obs-max", type=int, default=256, help="Legacy single-field maximum sensors.")

    # generalized args
    p.add_argument("--cond-fields", type=int, nargs="+", default=None,
                   help="Conditioned field ids, e.g. --cond-fields 0 2")
    p.add_argument("--n-obs-min-list", type=int, nargs="+", default=None,
                   help="Per-field minimum sensors. Length 1 broadcasts to all cond_fields.")
    p.add_argument("--n-obs-max-list", type=int, nargs="+", default=None,
                   help="Per-field maximum sensors. Length 1 broadcasts to all cond_fields.")

    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Visualization conditioned fields. Defaults to cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Visualization exact sensors per field. Defaults to n_obs_max_list.")
    
    # ODE solver used at generation time. For 1-RF, Euler is the main benchmark because the method is designed for coarse-step sampling.
    p.add_argument(
        "--ode-solver", type=str, default="euler",
            choices=["euler", "heun"], help="ODE solver for generation. Use Euler for the main 1-RF benchmark; Heun is optional.")
    # Reconstruction benchmark step counts. These are the NFEs to compare after moving to 1-RF.
    p.add_argument(
        "--benchmark-n-steps", type=int, nargs="+", default=[2, 4, 8, 16],
            help="Sampling step counts used for reconstruction benchmarking.")

    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--n-steps-generation", type=int, default=32)

    # Training efficiency
    p.add_argument("--use-amp", default=False, action=argparse.BooleanOptionalAction,
                   help="bf16 autocast for the forward/loss computation.")
    p.add_argument("--objective", type=str, default="flow", choices=["flow", "regression"],
                   help="flow: conditional rectified flow. regression: deterministic MSE "
                        "regression of the field (fno3d backbone only).")
    p.add_argument("--t-sampling", type=str, default="uniform",
                   choices=["uniform", "logit_normal"],
                   help="Flow-matching time distribution.")
    p.add_argument("--compile-model", default=False, action=argparse.BooleanOptionalAction,
                   help="torch.compile the velocity backbone (measured ~18%% "
                        "step-time and ~20%% memory reduction on H100).")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="EMA decay for model weights; 0 disables EMA. "
                        "Benchmarks/checkpoints use EMA weights when enabled.")

    # Realistic measurement operators (dict from YAML: noise_sigma_min/max,
    # occlusion_prob/kind/frac_min/frac_max, field_dropout_prob).
    p.add_argument("--measurement-ops", type=json.loads, default=None,
                   help='JSON dict of measurement operator settings, e.g. '
                        '\'{"noise_sigma_max": 0.1, "field_dropout_prob": 0.3}\'.')

    # Dataset selection
    p.add_argument("--sensor-pool", type=str, default=None, choices=[None, "surface"],
                   help="None: sensors anywhere on the point set (canonical). "
                        "'surface': restrict sensors to the H5's surface_indices pool "
                        "(cylinder surface-to-field task; YAML key sensor_pool).")
    p.add_argument("--dataset", type=str, default="jhu", choices=["jhu", "shiftwing"],
                   help="jhu: single-case snapshot H5. shiftwing: per-case wing "
                        "point clouds with a surface observation pool.")
    p.add_argument("--processed-root", type=str,
                   default=f"{DATA_DIR}/shift_wing/processed",
                   help="Processed SHIFT-WING directory (dataset=shiftwing).")

    return p.parse_args()


class EMAWeights:
    """Exponential moving average of model parameters (fp32 shadow)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.detach().float(), 1.0 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        # Checkpoints are loaded with map_location="cpu"; keep the shadow on
        # the device it was constructed on (the model's device) so lerp_
        # against live parameters does not mix devices after a resume.
        devices = {k: v.device for k, v in self.shadow.items()}
        loaded = {
            k: v.clone().to(devices.get(k, v.device))
            for k, v in state["shadow"].items()
        }
        # Keep shadow entries for params added after the checkpoint was
        # written (e.g. the zero-init multiscale gather branch).
        for k, v in self.shadow.items():
            loaded.setdefault(k, v)
        self.shadow = loaded

    def copy_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA weights into the model; returns the originals for restore."""
        original = {}
        state = model.state_dict()
        for k, shadow_v in self.shadow.items():
            original[k] = state[k].detach().clone()
            state[k].copy_(shadow_v.to(state[k].dtype))
        return original

    @staticmethod
    def restore(model: nn.Module, original: dict[str, torch.Tensor]) -> None:
        state = model.state_dict()
        for k, v in original.items():
            state[k].copy_(v)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_conditioning_args(args):
    # training
    if args.cond_fields is None:
        args.cond_fields = [args.cond_field]
    if args.n_obs_min_list is None:
        args.n_obs_min_list = [args.n_obs_min]
    if args.n_obs_max_list is None:
        args.n_obs_max_list = [args.n_obs_max]

    # visualization
    if args.vis_cond_fields is None:
        args.vis_cond_fields = list(args.cond_fields)
    if args.vis_n_obs_list is None:
        args.vis_n_obs_list = list(args.n_obs_max_list)

    return args

class IIDGaussianPrior(nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


class RFFGaussianPrior(nn.Module):
    """Scalable smooth Gaussian-field approximation via random Fourier features."""

    def __init__(self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)


def collate_snapshots(batch):
    out = {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "fields": torch.stack([b["fields"] for b in batch], dim=0),
        "time_index": torch.stack([b["time_index"] for b in batch], dim=0),
        "physical_time": torch.stack([b["physical_time"] for b in batch], dim=0),
    }
    if "valid_sensor_mask" in batch[0]:   # sensor_pool="surface"
        out["valid_sensor_mask"] = torch.stack([b["valid_sensor_mask"] for b in batch], dim=0)
    return out


def sample_query_subset(
    coords: torch.Tensor,
    fields: torch.Tensor,
    n_query: int | None,
    mode: str = "uniform",
    obs_coords: torch.Tensor | None = None,
    obs_mask: torch.Tensor | None = None,
    obs_counts: Sequence[int] | None = None,
    near_ratio: float = 0.25,
    far_ratio: float = 0.25,
    sigma_ratio: float = 0.05,
    obs_mix_chunk_size: int = 65536,
):
    if n_query is None or n_query >= coords.shape[1]:
        return coords, fields, None

    bsz, n_pts, coord_dim = coords.shape
    n_query = int(n_query)

    def take_weighted(weights: torch.Tensor, count: int) -> torch.Tensor:
        count = int(count)
        if count <= 0:
            return torch.empty(0, device=coords.device, dtype=torch.long)

        tiny = torch.finfo(coords.dtype).tiny
        weights = weights.to(dtype=coords.dtype).clamp_min(tiny)
        return torch.multinomial(weights, num_samples=count, replacement=False)

    def take_uniform(count: int) -> torch.Tensor:
        count = int(count)
        if count <= 0:
            return torch.empty(0, device=coords.device, dtype=torch.long)
        return torch.randperm(n_pts, device=coords.device)[:count]

    def finalize_indices(pieces: Sequence[torch.Tensor]) -> torch.Tensor:
        idx = torch.cat([p for p in pieces if p.numel() > 0], dim=0)
        # Avoid torch.unique here: CUDA unique has variable-size output and
        # triggers host synchronization. Rare duplicates are cheaper than a
        # per-batch sync in this loss-point sampler.
        return idx.sort().values

    all_idx = []
    for b in range(bsz):
        if mode == "obs_mix" and obs_coords is not None and obs_counts is not None:
            n_valid = int(obs_counts[b])
        else:
            n_valid = None

        if mode != "obs_mix" or n_valid is None or n_valid <= 0:
            idx = torch.randperm(n_pts, device=coords.device)[:n_query].sort().values
            all_idx.append(idx)
            continue

        # Compute min-distance from every grid point to the nearest valid
        # sensor in chunks to avoid materialising the full [N_pts × N_obs]
        # matrix, which is infeasible on 3-D grids (~1.95 M points).
        ocoords = obs_coords[b, :n_valid]                 # [n_valid, D]
        chunk = max(1, int(obs_mix_chunk_size))
        d_min_parts: list[torch.Tensor] = []
        for start in range(0, n_pts, chunk):
            end = min(start + chunk, n_pts)
            d_min_parts.append(
                torch.cdist(coords[b, start:end], ocoords, p=2.0).amin(dim=-1)
            )
        d_min = torch.cat(d_min_parts, dim=0)             # [n_pts]
        bbox_diag = (coords[b].amax(dim=0) - coords[b].amin(dim=0)).norm().clamp_min(1e-6)
        sigma = (sigma_ratio * bbox_diag).clamp_min(1e-6)

        near_count = min(n_query, max(0, int(round(n_query * near_ratio))))
        far_count = min(n_query - near_count, max(0, int(round(n_query * far_ratio))))
        uniform_count = n_query - near_count - far_count

        near_weights = torch.exp(-(d_min ** 2) / (2 * sigma ** 2 + 1e-12))
        far_weights = d_min.clamp_min(0.0)

        pieces = [
            take_weighted(near_weights, near_count),
            take_weighted(far_weights, far_count),
            take_uniform(uniform_count),
        ]

        idx = finalize_indices(pieces)
        all_idx.append(idx)

    idx = torch.stack(all_idx, dim=0)
    coord_idx = idx.unsqueeze(-1).expand(-1, -1, coord_dim)
    field_idx = idx.unsqueeze(-1).expand(-1, -1, fields.shape[-1])
    return torch.gather(coords, dim=1, index=coord_idx), torch.gather(fields, dim=1, index=field_idx), idx

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: int | None,
    query_sampling: str = "uniform",
    query_sample_near_ratio: float = 0.25,
    query_sample_far_ratio: float = 0.25,
    query_sample_sigma_ratio: float = 0.05,
    epoch: int = 0,
    measurement_cfg: dict | None = None,
    use_amp: bool = False,
    ema: Optional["EMAWeights"] = None,
    spectral_weight: float = 0.0,
    spectral_grid_shape: Sequence[int] | None = None,
    spectral_block: int = 32,
    spectral_bins: int = 12,
    spectral_window: bool = False,
) -> tuple[float, float, float]:
    """Run one training or evaluation epoch.

    Returns:
        (mean_loss, epoch_time_s, peak_gpu_mem_mb)
        epoch_time_s covers only the batch loop — logging and checkpointing
        overhead is excluded. peak_gpu_mem_mb is the CUDA peak since the last
        reset (call torch.cuda.reset_peak_memory_stats() before this function
        if you want a per-epoch peak).
    """
    training = optimizer is not None
    model.train(training)

    if training and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    total = 0.0
    count = 0

    mode_str = "Train" if training else "Eval"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False)

    t0 = time.perf_counter()
    for batch in pbar:
        coords_full = batch["coords"].to(device, non_blocking=True)
        fields_full = batch["fields"].to(device, non_blocking=True)

        if "obs_pool_coords" in batch:
            # Observations drawn from a dedicated pool (e.g. body surface),
            # decoupled from the generation point set.
            obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = (
                build_sparse_condition_from_pool(
                    pool_coords=batch["obs_pool_coords"].to(device, non_blocking=True),
                    pool_values=batch["obs_pool_values"].to(device, non_blocking=True),
                    pool_field_ids=batch["obs_pool_field_ids"].to(device, non_blocking=True),
                    n_obs_min=n_obs_min_list,
                    n_obs_max=n_obs_max_list,
                )
            )
            obs_counts = None
            if training and measurement_cfg:
                obs_values, obs_mask = apply_measurement_operators(
                    obs_coords=obs_coords,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_field_ids=obs_field_ids,
                    cfg=measurement_cfg,
                )
            # Flow-condition parameter tokens are appended after the
            # measurement operators: they are known exactly, not measured.
            obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = append_extra_tokens(
                obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids,
                extra_coords=batch["param_coords"].to(device, non_blocking=True),
                extra_values=batch["param_values"].to(device, non_blocking=True),
                extra_field_ids=batch["param_field_ids"].to(device, non_blocking=True),
            )
        else:
            # Build generalized sparse observations from the query point set.
            valid_mask = batch.get("valid_sensor_mask")   # sensor_pool="surface"
            if valid_mask is not None:
                valid_mask = valid_mask.to(device, non_blocking=True)
            obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids, obs_counts = build_sparse_condition(
                coords_full=coords_full,
                fields_full=fields_full,
                cond_fields=cond_fields,
                n_obs_min=n_obs_min_list,
                n_obs_max=n_obs_max_list,
                return_counts=True,
                valid_mask=valid_mask,
            )

            # Realistic measurement operators (noise / occlusion / channel dropout)
            # are applied during training so conditioning is amortized over them.
            if training and measurement_cfg:
                obs_values, obs_mask = apply_measurement_operators(
                    obs_coords=obs_coords,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_field_ids=obs_field_ids,
                    cfg=measurement_cfg,
                )

        # for models that must operate on the full regular grid like FNO,
        # point subsampling will be disabled.
        effective_n_query = None if getattr(model, "requires_full_grid", False) else n_query_points
        sampling_mode = query_sampling if training else "uniform"
        coords_q, fields_q, _ = sample_query_subset(
            coords=coords_full,
            fields=fields_full,
            n_query=effective_n_query,
            mode=sampling_mode,
            obs_coords=obs_coords,
            obs_mask=obs_mask,
            obs_counts=obs_counts,
            near_ratio=query_sample_near_ratio,
            far_ratio=query_sample_far_ratio,
            sigma_ratio=query_sample_sigma_ratio,
        )

        # Append a contiguous grid block for the binned spectral loss. Taking
        # it in grid-index space keeps it a genuine cube under the symmetry
        # augmentations, which move coordinates but not grid topology. One
        # forward pass serves both the pointwise and the spectral term.
        blk_shape = None
        if training and spectral_weight > 0.0 and spectral_grid_shape is not None:
            from methods.dmfgen.spectral_loss import block_indices
            b_idx, blk_shape = block_indices(
                spectral_grid_shape, spectral_block, coords_full.device)
            coords_q = torch.cat(
                [coords_q, coords_full[:, b_idx]], dim=1)
            fields_q = torch.cat(
                [fields_q, fields_full[:, b_idx]], dim=1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, loss_parts = model.training_loss(
                x1=fields_q,
                coords=coords_q,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                obs_indices=obs_indices,
                compute_metrics=False,
                spectral_block_shape=blk_shape,
                spectral_weight=spectral_weight if blk_shape is not None else 0.0,
                spectral_bins=spectral_bins,
                spectral_window=spectral_window,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        current_loss = float(loss.detach().cpu())
        total += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    epoch_time_s = time.perf_counter() - t0

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    return total / max(count, 1), epoch_time_s, peak_mem_mb


def run_reconstruction_benchmark(
    model: nn.Module,
    dataset: Dataset,
    epoch: int,
    device: torch.device,
    recon_dir: str,
    args: argparse.Namespace,
) -> None:
    """Save reconstruction plots/metrics for one epoch at the configured NFEs."""
    recon_dir_epoch = os.path.join(recon_dir, f"Epoch_{epoch}")
    os.makedirs(recon_dir_epoch, exist_ok=True)

    step_list = args.benchmark_n_steps if args.benchmark_n_steps else [args.n_steps_generation]
    for nfe in step_list:
        recon_metrics = visualize_reconstruction(
            model=model,
            dataset=dataset,
            epoch=epoch,
            device=device,
            save_dir=recon_dir_epoch,

            cond_fields=args.vis_cond_fields,
            n_obs=args.vis_n_obs_list,
            n_steps=nfe,
            ode_solver=args.ode_solver,
            snapshot_index=0,
            file_tag=f"{args.ode_solver}_nfe{nfe}",
            save_metrics_json=True,
            field_names=args.field_names,
        )

        metric_str = ", ".join([f"{k}:{v:.4e}" for k, v in recon_metrics.items()])
        print(f"[recon] epoch={epoch:04d} solver={args.ode_solver} n_steps={nfe} | {metric_str}")


def find_latest_run_dir(demo_dir: str, save_dir: str, demo_num: int) -> Path | None:
    save_root = Path(demo_dir) / Path(save_dir).parent
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    if not save_root.exists():
        return None

    candidates = [
        path for path in save_root.glob(f"{run_prefix}*")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def extract_run_timestamp(run_dir: Path, save_dir: str, demo_num: int) -> str:
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    run_name = run_dir.name
    if run_name.startswith(run_prefix):
        return run_name[len(run_prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_path(path: Path, suffix: str = "_bk") -> Path:
    candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}{suffix}{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def backup_existing_artifact(path: Path) -> None:
    if not path.exists():
        return

    target = backup_path(path)
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)

def main():

    args = parse_args()
    demo_dir = str(REPO_ROOT)   # config / save paths are repository-relative
    
    # YAML Loading and Backup
    config_path = os.path.join(demo_dir, args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.exists(config_path):
        print(f"\n[*] Starting:... I found config file at: {config_path}\n")
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        # Overwrite default args with YAML values
        if yaml_config is not None:
            for key, value in yaml_config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"Warning: YAML key '{key}' is not a recognized argument. Ignoring.")
        args = normalize_conditioning_args(args)
                    
        # Backup the YAML file
        backup_dir = os.path.join(demo_dir, "configs", "pointcloud_ffm")
        os.makedirs(backup_dir, exist_ok=True)
        backup_filename = f"config_pointcloud_ffm_DemoN{args.Demo_Num}_{timestamp}.yaml"
        shutil.copy(config_path, os.path.join(backup_dir, backup_filename))
        print(f"[*] Config backed up to: {os.path.join(backup_dir, backup_filename)}\n")
    else:
        print(f"\n[Warning: !] Config file not found at {config_path}. Using default parameters.\n")
        args.Demo_Num = 0  # Force Demo_Num to 0 as default
    
    # Setup the Dynamic Directories with Demo_Num
    set_seed(args.seed)

    start_epoch = 1
    best_val = float("inf")
    reload_ckpt = None
    run_timestamp = timestamp
    save_dir = Path(os.path.join(demo_dir, args.save_dir + f"_DemoN{args.Demo_Num}" + f"_{timestamp}"))

    if args.RELOAD:
        latest_run_dir = find_latest_run_dir(demo_dir=demo_dir, save_dir=args.save_dir, demo_num=args.Demo_Num)
        reload_path = None
        if latest_run_dir is not None:
            for ckpt_name in ("last.pt", "best.pt"):
                candidate = latest_run_dir / ckpt_name
                if candidate.exists():
                    reload_path = candidate
                    break

        if reload_path is not None:
            save_dir = latest_run_dir
            run_timestamp = extract_run_timestamp(latest_run_dir, args.save_dir, args.Demo_Num)
            reload_ckpt = torch.load(reload_path, map_location="cpu")
            start_epoch = int(reload_ckpt.get("epoch", 0)) + 1
            best_val = resume_best_val(reload_ckpt, latest_run_dir)

            backup_existing_artifact(reload_path)
            print(f"[*] RELOAD=True, resuming from: {reload_path}")
            print(f"[*] Resume will start from epoch {start_epoch}\n")
        else:
            print("[*] RELOAD=True, but no matching last.pt or best.pt was found. Training will start from scratch.\n")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Selection protocol stamps (protocol): checkpoints are chosen on the TUNE
    # half of the held-out frames with a fixed-seed sensor draw at vis_n_obs_list.
    args.val_split = "tune"
    args.val_fixed_seed = VAL_FIXED_SEED
    # Save the final parsed args to a JSON in the model folder just to be safe
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Setup CSV and Recon Dirs (both live inside the run's model directory)
    csv_base_dir = str(save_dir)
    recon_base_dir = str(save_dir)

    if args.RELOAD and reload_ckpt is not None:
        loss_dir = Path(csv_base_dir) / f"Loss_DemoN{args.Demo_Num}_{run_timestamp}"
        recon_dir_existing = Path(recon_base_dir) / "Recon" / f"demo_N{args.Demo_Num}_{run_timestamp}"

        backup_existing_artifact(loss_dir)
        backup_existing_artifact(recon_dir_existing)

    # Initialize helpers
    logger = MetricsLogger(base_dir=csv_base_dir, Demo_Num=args.Demo_Num, timestamp=run_timestamp)
    recon_dir = create_recon_dir(base_dir=recon_base_dir, Demo_Num=args.Demo_Num, timestamp=run_timestamp)
    
    print(f"[*] Model checkpoints will save to: {save_dir}")
    print(f"[*] Logging losses to: {logger.save_dir}")
    print(f"[*] Saving recon plots to: {recon_dir}\n")

    device_ids = args.device_ids
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    if args.dataset == "shiftwing":
        train_set = ShiftWingDataset(args.processed_root, split="train")
        val_set = ShiftWingDataset(args.processed_root, split="val")
        collate_fn = collate_wing
        args.field_names = train_set.field_names
        print(f"[*] SHIFT-WING dataset: {len(train_set)} train / {len(val_set)} val cases, "
              f"n_obs_field_types={train_set.n_obs_field_types}")
    else:
        train_set = TurbulentCombustionH5Dataset(
            args.data,
            split="train",
            train_ratio=args.train_ratio,
            field_names=args.field_names,
            seed=args.seed,
            time_stride=args.time_stride,
            stats_path=str(save_dir / "dataset_stats.pt"),
            sensor_pool=args.sensor_pool,
        )
        val_set = TurbulentCombustionH5Dataset(
            args.data,
            split="tune",
            train_ratio=args.train_ratio,
            field_names=args.field_names,
            seed=args.seed,
            time_stride=args.time_stride,
            stats_path=str(save_dir / "dataset_stats.pt"),
            sensor_pool=args.sensor_pool,
        )
        collate_fn = collate_snapshots
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    if args.num_workers > 0:
        # Keep workers alive across epochs to reduce epoch-boundary stalls.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **loader_kwargs,
    )

    # Grid shape used to cut the structured block for the spectral loss.
    # AUG_GRID_SHAPE already carries it for non-cubic datasets (FireBench);
    # otherwise infer a cube. Unstructured data (the wing) has no grid, so the
    # spectral term stays off there.
    spectral_grid_shape = None
    if args.spectral_weight > 0.0:
        env_grid = os.environ.get("AUG_GRID_SHAPE", "")
        n_pts = int(getattr(train_set, "num_points", 0) or 0)
        if env_grid:
            gs = tuple(int(v) for v in env_grid.split(","))
            if int(np.prod(gs)) == n_pts:
                spectral_grid_shape = gs
        elif n_pts:
            side = int(round(n_pts ** (1.0 / 3.0)))
            if side ** 3 == n_pts:
                spectral_grid_shape = (side, side, side)
        if spectral_grid_shape is None:
            print(f"[*] spectral loss requested but no usable grid for "
                  f"{n_pts} points; disabled")
            args.spectral_weight = 0.0
        else:
            print(f"[*] binned spectral loss ON: weight={args.spectral_weight}, "
                  f"grid={spectral_grid_shape}, block={args.spectral_block}^3, "
                  f"bins={args.spectral_bins}")

    if args.prior == "iid":
        prior = IIDGaussianPrior()
    elif args.prior in ("rff_powerlaw", "rff_kolmogorov"):
        from methods.dmfgen.spectral_prior import PowerLawRFFPrior
        prior = PowerLawRFFPrior(
            coord_dim=3, n_features=args.rff_features,
            slope=float(getattr(args, "prior_slope", 5.0 / 3.0)),
            k_min=float(getattr(args, "prior_k_min", 1.0)),
            k_max=float(getattr(args, "prior_k_max", 48.0)),
            seed=int(getattr(args, "seed", 0) or 0),
        )
        print(f"[*] spectrally-matched source prior: {prior}")
    else:
        prior = RFFGaussianPrior(
            coord_dim=3, n_features=args.rff_features,
            lengthscale=args.rff_lengthscale,
        )

    if args.backbone == "mlp_rbf":
        backbone = ConditionalPointMLPRBF(
            n_fields=train_set.num_fields,
            coord_dim=3,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            rbf_sigma=args.rbf_sigma,
            use_fourier_pe=args.USE_FOURIER_PE,
            fourier_pe_num_bands=args.fourier_pe_num_bands,
            fourier_pe_max_freq=args.fourier_pe_max_freq,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "perceiver":
        backbone = ConditionalPointPerceiver(
            n_fields=train_set.num_fields,
            coord_dim=3,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            field_embed_dim=args.field_embed_dim,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            decode_chunk_size=args.decode_chunk_size,
            share_query_proj=args.share_query_proj,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "GL_rbf_CQ":
        # Vendored upstream portable compact-query backbone; the adapter
        # merges configs/gl_rbf_cq_core.yaml first (mandatory), applies our
        # project-owned overrides, and returns a wrapper with our
        # training-loss contract (spectral loss, logit-normal t).
        raise NotImplementedError("the GL_rbf_CQ backbone ablation is not part of FlowReconBench (see PhyCoFlow_demo)")
        build_cq_model = None  # noqa: F841
        model, _cq_cfg = build_cq_model(args, train_set, device)
    elif args.backbone in ["GL_rbf", "GL_rbf_ENH"]:
        enhanced = args.backbone == "GL_rbf_ENH"

        # Enhanced defaults are resolved here so legacy GL_rbf configs/checkpoints
        # keep their original point-readout and zero-scale behavior.
        sensor_coord_encoding = args.sensor_coord_encoding
        if sensor_coord_encoding is None:
            sensor_coord_encoding = "fourier" if enhanced else "raw"

        latent_sensor_reinject = args.latent_sensor_reinject
        if latent_sensor_reinject is None:
            latent_sensor_reinject = enhanced

        query_latent_readout = args.query_latent_readout
        if query_latent_readout is None:
            query_latent_readout = enhanced

        enhanced_head_norm = args.enhanced_head_norm
        if enhanced_head_norm is None:
            enhanced_head_norm = enhanced

        query_readout_type = args.query_readout_type
        if query_readout_type is None:
            query_readout_type = "coord" if enhanced else "point"

        query_readout_scale_init = args.query_readout_scale_init
        if query_readout_scale_init is None:
            query_readout_scale_init = 1.0e-2 if enhanced else 0.0

        glres_scale_init = args.glres_scale_init
        if glres_scale_init is None:
            glres_scale_init = 1.0e-2 if enhanced else 0.0

        print(
            "[*] GL_rbf settings: "
            f"enhanced={enhanced}, "
            f"sensor_coord_encoding={sensor_coord_encoding}, "
            f"latent_sensor_reinject={latent_sensor_reinject}, "
            f"latent_reinject_every={args.latent_reinject_every}, "
            f"query_latent_readout={query_latent_readout}, "
            f"query_readout_type={query_readout_type}, "
            f"query_readout_scale_init={query_readout_scale_init}, "
            f"enhanced_head_norm={enhanced_head_norm}, "
            f"glres_scale_init={glres_scale_init}"
        )

        backbone = ConditionalPointHybridLocalGlobalRBF(
            n_fields=train_set.num_fields,
            coord_dim=3,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            rbf_sigma=args.rbf_sigma,
            summary_type=args.summary_type,

            gather_mode=args.gather_mode,
            gather_topk=args.gather_topk,
            gather_multiscale=args.gather_multiscale,
            gather_topk_coarse=args.gather_topk_coarse,
            gather_coarse_sigma_scale=args.gather_coarse_sigma_scale,
            gather_query_chunk_size=args.gather_query_chunk_size,
            learnable_rbf_sigma=args.learnable_rbf_sigma,
            neighbor_backend=args.neighbor_backend,

            sensor_local_topk=args.sensor_local_topk,
            sensor_local_dropout=args.sensor_local_dropout,
            use_fourier_pe=args.USE_FOURIER_PE,
            fourier_pe_num_bands=args.fourier_pe_num_bands,
            fourier_pe_max_freq=args.fourier_pe_max_freq,
            enhanced_backbone=enhanced,
            sensor_coord_encoding=sensor_coord_encoding,
            latent_sensor_reinject=latent_sensor_reinject,
            latent_reinject_every=args.latent_reinject_every,
            query_latent_readout=query_latent_readout,
            query_readout_type=query_readout_type,
            query_readout_scale_init=query_readout_scale_init,
            enhanced_head_norm=enhanced_head_norm,
            glres_scale_init=glres_scale_init,
            n_obs_field_types=getattr(train_set, "n_obs_field_types", None),
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "fno":
        # FNO requires an explicit regular-grid interpretation of the dataset.
        try:
            validate_regular_grid_compatibility(train_set, args.Num_x, args.Num_y)
            validate_regular_grid_compatibility(val_set, args.Num_x, args.Num_y)
        except ValueError as e:
            print(f"\n[Warning: !] {e}")
            print("[Warning: !] FNO baseline cannot start because the provided Num_x / Num_y "
                  "are missing or incompatible with the dataset.\n")
            raise SystemExit(1)

        backbone = FNO(
            n_fields=train_set.num_fields,
            Num_x=args.Num_x,
            Num_y=args.Num_y,
            n_modes_x=args.fno_modes_x,
            n_modes_y=args.fno_modes_y,
            hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_n_layers,
            condition_blur=args.condition_blur,
            condition_blur_kernel=args.condition_blur_kernel,
            condition_blur_sigma=args.condition_blur_sigma,
        )
        model = FNOFFM(backbone, prior, sigma_min=args.sigma_min).to(device)

        print(f"[*] Using grid-based FNO baseline with Num_x={args.Num_x}, Num_y={args.Num_y}")
        print("[*] Note: n_query_points is ignored for FNO because it requires the full grid.\n")
    elif args.backbone == "fno3d":
        # 3-D FNO architecture ablation. Faithful upstream neuraloperator core
        # (rfftn spectral conv), our sensor-conditioning API, full 125^3 grid.
        from methods.dmfgen.fno3d_backbone import (FNO3D, FNO3DFFM, FNO3DRegression,
                                               validate_regular_grid_compatibility_3d)
        try:
            gi = validate_regular_grid_compatibility_3d(
                train_set, args.Num_x, args.Num_y, args.Num_z)
            validate_regular_grid_compatibility_3d(
                val_set, args.Num_x, args.Num_y, args.Num_z)
        except ValueError as e:
            print(f"\n[Warning: !] {e}")
            print("[Warning: !] FNO3D baseline cannot start because Num_x/Num_y/Num_z "
                  "are missing or incompatible with the dataset.\n")
            raise SystemExit(1)

        backbone = FNO3D(
            n_fields=train_set.num_fields,
            Num_x=args.Num_x, Num_y=args.Num_y, Num_z=args.Num_z,
            n_modes_x=args.fno_modes_x,
            n_modes_y=args.fno_modes_y,
            n_modes_z=args.fno_modes_z,
            hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_n_layers,
            condition_blur=args.condition_blur,
            condition_blur_kernel=args.condition_blur_kernel,
            condition_blur_sigma=args.condition_blur_sigma,
            domain_padding=args.fno_domain_padding,
        )
        wrapper = FNO3DRegression if args.objective == "regression" else FNO3DFFM
        model = wrapper(backbone, prior, sigma_min=args.sigma_min).to(device)
        _nc = sum(p_.numel() for p_ in backbone.parameters() if p_.requires_grad)
        _nr = sum(p_.numel() * (2 if p_.is_complex() else 1)
                  for p_ in backbone.parameters() if p_.requires_grad)
        _kept = args.fno_modes_x * args.fno_modes_y * (args.fno_modes_z // 2 + 1)
        _tot = args.Num_x * args.Num_y * (args.Num_z // 2 + 1)
        print(f"[*] FNO3D grid=({args.Num_x},{args.Num_y},{args.Num_z}) "
              f"modes=({args.fno_modes_x},{args.fno_modes_y},{args.fno_modes_z}) "
              f"width={args.fno_hidden_channels} layers={args.fno_n_layers} "
              f"row_major={gi['row_major']}")
        print(f"[fno3d] params_numel={_nc} params_real_dof={_nr} "
              f"retained_modes={_kept} of {_tot} "
              f"({100.0 * _kept / _tot:.4f}% of the rfftn half-spectrum); "
              f"k_max={min(args.fno_modes_x, args.fno_modes_y, args.fno_modes_z) // 2} "
              f"of Nyquist {args.Num_x // 2}")
        if gi['diagnostics']:
            print(f"[fno3d] grid diagnostics: {gi['diagnostics']}")
        print("[*] Note: n_query_points is ignored for FNO3D because it requires the full grid.\n")
    else:
        raise ValueError(
            f'Error!!! Your backbone is not supported: {args.backbone}.'
            'Please select in ["mlp_rbf", "perceiver", "fno"]'
            )
    print(f'\nSelected Backbone: {args.backbone}\n')
    if args.objective == "regression" and not getattr(model, "deterministic", False):
        raise ValueError(f"objective=regression is implemented for backbone fno3d only, not {args.backbone}.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:  {total_params:,} total  ({trainable_params:,} trainable)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.t_sampling = args.t_sampling
    if args.t_sampling != "uniform":
        print(f"[*] Flow-matching time sampling: {args.t_sampling}")
    if args.compile_model:
        # Compile the bound forward rather than the module: the module object
        # (and therefore all checkpoint/EMA state-dict keys) is untouched,
        # so runs stay resume-compatible either way.
        model.model.forward = torch.compile(
            model.model.forward, mode="max-autotune-no-cudagraphs")
        print("[*] torch.compile enabled on the velocity backbone forward")
    ema = EMAWeights(model, args.ema_decay) if args.ema_decay > 0.0 else None
    if ema is not None:
        print(f"[*] EMA enabled with decay={args.ema_decay}")
    if args.use_amp:
        print("[*] bf16 autocast enabled for training/eval forward passes")
    if args.measurement_ops:
        print(f"[*] Measurement operators active during training: {args.measurement_ops}")

    if reload_ckpt is not None:
        missing, unexpected = model.load_state_dict(reload_ckpt["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
        if missing:
            # Only newly added zero-init modules (e.g. multiscale gather) may
            # be absent from older checkpoints.
            allowed = all("ms_coarse_out" in k for k in missing)
            if not allowed:
                raise RuntimeError(f"Missing checkpoint keys: {missing}")
            print(f"[*] RELOAD: initializing new modules not in checkpoint: {missing}")
        if "optimizer" in reload_ckpt:
            try:
                optimizer.load_state_dict(reload_ckpt["optimizer"])
            except ValueError as e:
                print(f"[*] RELOAD: optimizer state incompatible (new params?); "
                      f"starting optimizer fresh. ({e})")
        if ema is not None and "ema" in reload_ckpt:
            ema.load_state_dict(reload_ckpt["ema"])
        loaded_epoch = int(reload_ckpt.get("epoch", 0))
        print(f"[*] Reloaded model state from epoch {loaded_epoch}")

        if loaded_epoch > 0 and loaded_epoch % args.save_every == 0 and args.dataset == "jhu":
            print(f"[*] Saving reconstruction benchmark for reloaded epoch {loaded_epoch} before continuing.")
            run_reconstruction_benchmark(
                model=model,
                dataset=val_set,
                epoch=loaded_epoch,
                device=device,
                recon_dir=recon_dir,
                args=args,
            )

    log_param_count(model, f"dmfgen/{args.backbone}")
    _val_cond = {"cond_fields": args.cond_fields, "vis_n_obs_list": args.vis_n_obs_list,
                 "n_obs_min_list": args.n_obs_min_list, "n_obs_max_list": args.n_obs_max_list}
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_time, tr_mem = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cond_fields=args.cond_fields,
            n_obs_min_list=args.n_obs_min_list,
            n_obs_max_list=args.n_obs_max_list,
            n_query_points=args.n_query_points,
            query_sampling=args.query_sampling,
            spectral_weight=args.spectral_weight,
            spectral_grid_shape=spectral_grid_shape,
            spectral_block=args.spectral_block,
            spectral_bins=args.spectral_bins,
            spectral_window=bool(getattr(args, 'spectral_window', False)),
            query_sample_near_ratio=args.query_sample_near_ratio,
            query_sample_far_ratio=args.query_sample_far_ratio,
            query_sample_sigma_ratio=args.query_sample_sigma_ratio,
            epoch=epoch,
            measurement_cfg=args.measurement_ops,
            use_amp=args.use_amp,
            ema=ema,
        )
        scheduler.step()

        print(f"[train] epoch={epoch:04d} loss={tr_loss:.6e}  time={tr_time:.1f}s  peak_mem={tr_mem:.0f}MB")
        val_loss = None
        if epoch % args.eval_every == 0 or epoch == 1:
            with torch.no_grad(), fixed_validation_draw(_val_cond) as _val_n_obs:
                val_loss, _, _ = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    cond_fields=args.cond_fields,
                    n_obs_min_list=_val_n_obs,
                    n_obs_max_list=_val_n_obs,
                    n_query_points=args.n_query_points,
                    query_sampling=args.query_sampling,
                    spectral_weight=0.0,
                    query_sample_near_ratio=args.query_sample_near_ratio,
                    query_sample_far_ratio=args.query_sample_far_ratio,
                    query_sample_sigma_ratio=args.query_sample_sigma_ratio,
                    epoch=epoch,
                    use_amp=args.use_amp,
                )
            print(f"[valid] epoch={epoch:04d} loss={val_loss:.6e}")

            ckpt = {
                "model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "best_val": min(best_val, float(val_loss)),
                "mean": train_set.mean,
                "std": train_set.std,
                "field_names": train_set.field_names,
                "method": "1_rectified_flow",
                "backbone": args.backbone,
                "summary_type": args.summary_type,
                "ode_solver": args.ode_solver,
                "Num_x": args.Num_x,
                "Num_y": args.Num_y,
                "Num_z": args.Num_z,
                "fno_modes_x": args.fno_modes_x,
                "fno_modes_y": args.fno_modes_y,
                "fno_modes_z": args.fno_modes_z,
                "fno_hidden_channels": args.fno_hidden_channels,
                "fno_n_layers": args.fno_n_layers,
                "fno_domain_padding": args.fno_domain_padding,
                "condition_blur": args.condition_blur,
                "condition_blur_kernel": args.condition_blur_kernel,
                "condition_blur_sigma": args.condition_blur_sigma,
            }
            torch.save(ckpt, save_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, save_dir / "best.pt")
                print('Saving the best model...')
        
        if epoch % args.save_every == 0 and args.dataset == "jhu":
            # Benchmark the same validation snapshot at several NFEs,
            # using EMA weights when EMA is enabled. (The visualization
            # pipeline assumes the snapshot dataset; wing runs are
            # evaluated by the dedicated evaluation script.)
            original = ema.copy_to(model) if ema is not None else None
            run_reconstruction_benchmark(
                model=model,
                dataset=val_set,
                epoch=epoch,
                device=device,
                recon_dir=recon_dir,
                args=args,
            )
            if original is not None:
                EMAWeights.restore(model, original)

        logger.log_csv(epoch=epoch, train_loss=tr_loss, val_loss=val_loss,
                       epoch_time_s=tr_time, peak_gpu_mem_mb=tr_mem)
        _steps = epoch * len(train_loader)
        with open(save_dir / "cost_train.json", "w") as f:
            json.dump({"steps_per_epoch": len(train_loader), "epochs_completed": epoch,
                       "total_optimizer_steps": _steps, "batch_size": args.batch_size,
                       "samples_seen": _steps * args.batch_size}, f, indent=2)
        if epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs:
            logger.plot_history()

    print("Training complete.")
    print(f"Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    main()
