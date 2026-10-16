# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from neuralop.models import FNO as NeuralOpFNO  # pip install neuraloperator
except ImportError:
    NeuralOpFNO = None  # resolved lazily; the FNO-based classes raise at instantiation


class FNOSupervisedGrid(nn.Module):
    """neuraloperator FNO for regular-grid supervised sparse-to-full regression."""

    def __init__(
        self,
        n_fields: int,
        Num_x: int,
        Num_y: int,
        n_modes_x: int = 32,
        n_modes_y: int = 8,
        hidden_channels: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        if NeuralOpFNO is None:
            raise ImportError("neuraloperator is required for the Geo-FNO baseline.")
        self.n_fields = int(n_fields)
        self.Num_x = int(Num_x)
        self.Num_y = int(Num_y)
        self.fno = NeuralOpFNO(
            n_modes=(int(n_modes_y), int(n_modes_x)),
            in_channels=2 * int(n_fields),
            out_channels=int(n_fields),
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
            positional_embedding="grid",
        )

    def forward(self, obs_value_grid: torch.Tensor, obs_mask_grid: torch.Tensor) -> torch.Tensor:
        return self.fno(torch.cat([obs_value_grid, obs_mask_grid], dim=1))


class FNOSupervisedIrregular(nn.Module):
    """FNO + sparse scatter/gather wrapper for irregular 2-D meshes."""

    def __init__(
        self,
        n_fields: int,
        latent_Nx: int = 64,
        latent_Ny: int = 64,
        n_modes_x: int = 24,
        n_modes_y: int = 24,
        hidden_channels: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        if NeuralOpFNO is None:
            raise ImportError("neuraloperator is required for the Geo-FNO baseline.")
        self.n_fields = int(n_fields)
        self.latent_Nx = int(latent_Nx)
        self.latent_Ny = int(latent_Ny)
        self.fno = NeuralOpFNO(
            n_modes=(int(n_modes_y), int(n_modes_x)),
            in_channels=2 * int(n_fields),
            out_channels=int(n_fields),
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
            positional_embedding="grid",
        )

    def _gather_from_grid(self, grid: torch.Tensor, coords_2d: torch.Tensor) -> torch.Tensor:
        sample_pts = (coords_2d * 2.0 - 1.0).unsqueeze(2)
        gathered = F.grid_sample(
            grid,
            sample_pts,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return gathered.squeeze(-1).permute(0, 2, 1)

    def forward(
        self,
        coords_2d: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = coords_2d.shape[0]
        n_fields = self.n_fields
        n_y, n_x = self.latent_Ny, self.latent_Nx
        device = coords_2d.device
        dtype = coords_2d.dtype

        val_grid = torch.zeros(batch_size, n_fields, n_y * n_x, device=device, dtype=dtype)
        mask_grid = torch.zeros(batch_size, n_fields, n_y * n_x, device=device, dtype=dtype)

        for b_idx in range(batch_size):
            valid = obs_mask[b_idx].bool()
            if not valid.any():
                continue
            idx = obs_indices[b_idx, valid].long()
            fld = obs_field_ids[b_idx, valid].long()
            val = obs_values[b_idx, valid, 0]
            obs_xy = coords_2d[b_idx, idx]
            ix = (obs_xy[:, 0] * (n_x - 1)).long().clamp(0, n_x - 1)
            iy = (obs_xy[:, 1] * (n_y - 1)).long().clamp(0, n_y - 1)
            flat = iy * n_x + ix
            val_grid[b_idx, fld, flat] = val
            mask_grid[b_idx, fld, flat] = 1.0

        val_grid = val_grid.reshape(batch_size, n_fields, n_y, n_x)
        mask_grid = mask_grid.reshape(batch_size, n_fields, n_y, n_x)
        pred_grid = self.fno(torch.cat([val_grid, mask_grid], dim=1))
        return self._gather_from_grid(pred_grid, coords_2d)
