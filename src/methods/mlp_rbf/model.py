# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional


# ═════════ §1. Low-level MLP + gather utilities ═════════

# ------------------------------
# mlp_rbf backbone
# ------------------------------
def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, act=nn.GELU) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(dim, hidden_dim), act()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class FourierPositionalEncoding(nn.Module):
    """Sine-cosine frequency encoding for spatial coordinates."""

    def __init__(self, coord_dim: int, num_bands: int = 32, max_freq: float = 64.0):
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.num_bands = int(num_bands)
        self.out_dim = self.coord_dim * self.num_bands * 2
        freqs = torch.linspace(1.0, float(max_freq) / 2.0, self.num_bands)
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords[..., : self.coord_dim] * 2.0 - 1.0
        x = coords.unsqueeze(-1) * self.freqs * math.pi
        enc = torch.cat([x.sin(), x.cos()], dim=-1)
        return enc.reshape(*coords.shape[:-1], self.out_dim)

# ═════════ §4. Point-cloud backbones (for FFM) ═════════

class ConditionalPointMLPRBF(nn.Module):
    """
    Current baseline backbone:
      - per-query-point MLP encoder
      - sensor token encoder
      - RBF-weighted local sensor aggregation
      - one global pooled feature
      - pointwise velocity head

    This is your current model, kept under a clearer name so it can be
    compared directly against the Perceiver backbone.
    """
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        rbf_sigma: float = 0.05,
        use_fourier_pe: bool = False,
        fourier_pe_num_bands: int = 32,
        fourier_pe_max_freq: float = 64.0,
        n_obs_field_types: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        # See Senseiver.__init__ for why the SENSOR field-id vocabulary is
        # decoupled from n_fields (SHIFT-WING: 9 sensor types, 4 output
        # fields). Defaults to n_fields, so no existing run changes shape.
        self.n_obs_field_types = int(n_obs_field_types) if n_obs_field_types else int(n_fields)
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma
        self.use_fourier_pe = bool(use_fourier_pe)
        self.pos_enc = FourierPositionalEncoding(
            coord_dim,
            num_bands=fourier_pe_num_bands,
            max_freq=fourier_pe_max_freq,
        ) if use_fourier_pe else None
        self.coord_feat_dim = self.pos_enc.out_dim if self.pos_enc is not None else coord_dim

        self.field_embed = nn.Embedding(self.n_obs_field_types, field_embed_dim)

        self.point_encoder = make_mlp(self.coord_feat_dim + n_fields + 1, hidden_dim, hidden_dim, depth=3)
        self.obs_encoder = make_mlp(coord_dim + 1 + field_embed_dim, cond_dim, cond_dim, depth=3)
        self.global_encoder = make_mlp(hidden_dim, hidden_dim, hidden_dim, depth=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_fields),
        )

    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Embed the physical field identity for each sparse sensor.
        safe_field_ids = obs_field_ids.clamp_min(0)
        obs_field_feat = self.field_embed(safe_field_ids)
        obs_field_feat = obs_field_feat * obs_mask.unsqueeze(-1)

        # Encode sparse sensor tokens.
        obs_in = torch.cat([obs_coords, obs_values, obs_field_feat], dim=-1)
        obs_feat = self.obs_encoder(obs_in)
        obs_feat = obs_feat * obs_mask.unsqueeze(-1)

        # RBF weighting from each query point to each sparse sensor.
        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        weights = torch.softmax(-d2 / (2 * self.rbf_sigma ** 2 + 1e-12), dim=-1)
        return torch.einsum("bnm,bmd->bnd", weights, obs_feat)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_pts, _ = x_t.shape
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)

        coord_feat = self.pos_enc(coords) if self.pos_enc is not None else coords
        point_feat = self.point_encoder(torch.cat([coord_feat, x_t, t_feat], dim=-1))
        local_cond = self.aggregate_sparse_obs(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        global_feat = self.global_encoder(point_feat.mean(dim=1)).unsqueeze(1).expand(bsz, n_pts, -1)

        return self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))


class DeterministicMLPRBFRegressor(nn.Module):
    """Supervised sparse-sensor regressor using the existing FFM MLP-RBF backbone."""

    def __init__(self, backbone: ConditionalPointMLPRBF):
        super().__init__()
        self.backbone = backbone
        self.n_fields = int(backbone.n_fields)

    def forward(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_query, _ = query_coords.shape
        x_t = torch.zeros(
            bsz,
            n_query,
            self.n_fields,
            device=query_coords.device,
            dtype=query_coords.dtype,
        )
        t = torch.zeros(bsz, device=query_coords.device, dtype=query_coords.dtype)
        return self.backbone(
            t=t,
            x_t=x_t,
            coords=query_coords,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )
