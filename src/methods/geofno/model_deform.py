# Geo-FNO proper: Fourier neural operator with a learned deformation (Li et al. 2023,
# "Fourier Neural Operator with Learned Deformations for PDEs on General Geometries").
# Ported from the reference implementation (neuraloperator/Geo-FNO,
# elasticity/elas_geofno_v2.py: IPHI, SpectralConv2d, FNO2d). Differences from that file
# are listed in BENCHMARK_DIFF.md; the ones that change the function computed are marked
# [diff] below.
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class IPHI(nn.Module):
    """Learned deformation x -> xi from the physical domain to the latent torus."""

    def __init__(
        self,
        width: int = 32,
        center: Sequence[float] = (0.5, 0.5),
        aspect: Sequence[float] = (1.0, 1.0),
    ):
        super().__init__()
        self.width = int(width)
        self.fc0 = nn.Linear(4, self.width)
        self.fc_no_code = nn.Linear(3 * self.width, 4 * self.width)
        self.fc1 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc2 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc3 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc4 = nn.Linear(4 * self.width, 2)
        self.activation = torch.tanh
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32).reshape(1, 1, 2))
        # [diff] angle/radius are taken in isotropic physical units: the dataset
        # normalises x and y to [0, 1] separately, which turns the body into an ellipse.
        self.register_buffer("aspect", torch.tensor(aspect, dtype=torch.float32).reshape(1, 1, 2))
        self.register_buffer(
            "B",
            math.pi * torch.pow(2, torch.arange(0, self.width // 4, dtype=torch.float32)).reshape(1, 1, 1, self.width // 4),
        )
        # [diff] the deformation starts at the identity (reference: default init).
        nn.init.zeros_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch, N, 2)
        d = (x - self.center) * self.aspect
        angle = torch.atan2(d[:, :, 1], d[:, :, 0])
        radius = torch.norm(d, dim=-1, p=2)
        xd = torch.stack([x[:, :, 0], x[:, :, 1], angle, radius], dim=-1)

        # sin features from NeRF
        b = x.shape[0]
        xd_sin = torch.sin(self.B * xd.view(b, -1, 4, 1)).view(b, -1, 4 * (self.width // 4))
        xd_cos = torch.cos(self.B * xd.view(b, -1, 4, 1)).view(b, -1, 4 * (self.width // 4))
        xd = self.fc0(xd)
        xd = torch.cat([xd, xd_sin, xd_cos], dim=-1)
        xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = self.activation(xd)
        xd = self.fc2(xd)
        xd = self.activation(xd)
        xd = self.fc3(xd)
        xd = self.activation(xd)
        xd = self.fc4(xd)
        return x + x * xd


class SpectralConv2d(nn.Module):
    """Fourier layer on the latent torus. Either side can be a point cloud:
    points -> latent grid (xi_in), latent grid -> latent grid, latent grid -> points (xi_out).

    Every transform uses the Fourier-series convention u(xi) = sum_k u_k exp(2 pi i k.xi),
    so the point sums are quadratures of the same integrals the grid FFTs compute and a
    point cloud lying on the latent grid reproduces rfft2/irfft2 exactly."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, s1: int = 32, s2: int = 32):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)   # wavenumbers -modes1 .. modes1-1 along xi_1
        self.modes2 = int(modes2)   # wavenumbers 0 .. modes2-1 along xi_2 (Hermitian half)
        self.s1 = int(s1)
        self.s2 = int(s2)
        if self.s1 < 2 * self.modes1 or self.s2 // 2 + 1 < self.modes2:
            raise ValueError(f"latent grid {self.s1}x{self.s2} cannot hold modes ({self.modes1}, {self.modes2}).")

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        k1 = torch.cat([torch.arange(0, self.modes1), torch.arange(-self.modes1, 0)]).float()
        k2 = torch.arange(0, self.modes2).float()
        hermitian = torch.full((self.modes2,), 2.0)
        hermitian[0] = 1.0
        self.register_buffer("k1", k1, persistent=False)
        self.register_buffer("k2", k2, persistent=False)
        self.register_buffer("hermitian", hermitian, persistent=False)

    @staticmethod
    def compl_mul2d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def _phase(self, xi: torch.Tensor) -> torch.Tensor:
        # xi (batch, N, 2) -> 2 pi <k, xi>, (batch, N, 2*modes1*modes2)
        p1 = xi[..., 0:1] * self.k1
        p2 = xi[..., 1:2] * self.k2
        return 2 * math.pi * (p1.unsqueeze(-1) + p2.unsqueeze(-2)).flatten(-2)

    def forward(
        self,
        u: torch.Tensor,
        xi_in: Optional[torch.Tensor] = None,
        w_in: Optional[torch.Tensor] = None,
        xi_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batchsize = u.shape[0]

        # Fourier coefficients on the kept modes, (batch, channels, 2*modes1, modes2)
        if xi_in is None:
            s1, s2 = u.size(-2), u.size(-1)
            u_ft = torch.fft.rfft2(u, norm="forward")
            u_ft = torch.cat([u_ft[:, :, :self.modes1, :self.modes2], u_ft[:, :, -self.modes1:, :self.modes2]], dim=-2)
        else:
            # u (batch, channels, M) at points xi_in (batch, M, 2), quadrature weights w_in (batch, M)
            s1, s2 = self.s1, self.s2
            phase = self._phase(xi_in)
            uw = u * w_in.unsqueeze(1)
            u_ft = torch.complex(torch.matmul(uw, torch.cos(phase)), -torch.matmul(uw, torch.sin(phase)))
            u_ft = u_ft.view(batchsize, self.in_channels, 2 * self.modes1, self.modes2)

        # Multiply relevant Fourier modes
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, self.modes1:], self.weights2)

        # Return to physical space
        if xi_out is None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            return torch.fft.irfft2(out_ft, s=(s1, s2), norm="forward")

        # [diff] evaluates the same real trigonometric polynomial irfft2 does (the
        # reference mirrors the half spectrum with a one-index offset along k1).
        # xi_out (batch or 1, N, 2) -> (batch, channels, N)
        out_ft = (torch.cat([factor1, factor2], dim=-2) * self.hermitian).flatten(-2)
        phase = self._phase(xi_out).transpose(1, 2)
        return torch.matmul(out_ft.real, torch.cos(phase)) - torch.matmul(out_ft.imag, torch.sin(phase))


class GeoFNO2d(nn.Module):
    """FNO2d of the reference: point cloud -> latent grid -> ... -> point cloud."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        width: int,
        s1: int = 40,
        s2: int = 40,
        n_layers: int = 3,
    ):
        super().__init__()
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.width = int(width)
        self.s1 = int(s1)
        self.s2 = int(s2)
        self.n_layers = int(n_layers)

        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv_in = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.s1, self.s2)
        self.convs = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.s1, self.s2) for _ in range(self.n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.bs = nn.ModuleList([nn.Conv2d(2, self.width, 1) for _ in range(self.n_layers + 1)])
        self.conv_out = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.s1, self.s2)
        self.b_out = nn.Conv1d(2, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, device: torch.device) -> torch.Tensor:
        gridx = torch.arange(self.s1, dtype=torch.float32, device=device) / self.s1
        gridy = torch.arange(self.s2, dtype=torch.float32, device=device) / self.s2
        gridx = gridx.reshape(1, 1, self.s1, 1).expand(1, 1, self.s1, self.s2)
        gridy = gridy.reshape(1, 1, 1, self.s2).expand(1, 1, self.s1, self.s2)
        return torch.cat((gridx, gridy), dim=1)

    def forward(
        self,
        u: torch.Tensor,
        x_in: torch.Tensor,
        w_in: torch.Tensor,
        x_out: torch.Tensor,
        iphi: nn.Module,
    ) -> torch.Tensor:
        # u (batch, M, in_channels) input function values at x_in (batch, M, 2)
        # w_in (batch, M) quadrature weights of the input points (0 for padding)
        # x_out (batch or 1, N, 2) query mesh
        grid = self.get_grid(u.device)

        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc = self.conv_in(u, xi_in=iphi(x_in), w_in=w_in) + self.bs[0](grid)
        uc = F.gelu(uc)

        for conv, w, b in zip(self.convs, self.ws, self.bs[1:]):
            uc = conv(uc) + w(uc) + b(grid)
            uc = F.gelu(uc)

        u = self.conv_out(uc, xi_out=iphi(x_out)) + self.b_out(x_out.permute(0, 2, 1))

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u


class GeoFNOSupervisedMesh(nn.Module):
    """Geo-FNO for supervised sparse-to-full regression on the native mesh.

    The input function lives on the sensors (the reference supports different input and
    output point clouds): each sensor carries the same value + mask channels the grid
    variant rasterises, and the first Fourier layer transforms the sensor point cloud
    directly; the last one is evaluated at the query mesh points. No interpolation onto
    a grid happens on either side."""

    def __init__(
        self,
        n_fields: int,
        latent_Nx: int = 128,
        latent_Ny: int = 64,
        n_modes_x: int = 32,
        n_modes_y: int = 16,
        hidden_channels: int = 64,
        n_layers: int = 3,
        iphi_width: int = 32,
        center: Sequence[float] = (0.5, 0.5),
        aspect: Sequence[float] = (1.0, 1.0),
        domain_padding: float = 0.1,
    ):
        super().__init__()
        self.n_fields = int(n_fields)
        # [diff] the unit square is embedded in [pad, 1 - pad]^2 of the latent torus so
        # the non-periodic inflow/outflow never wrap onto each other.
        self.domain_padding = float(domain_padding)
        self.iphi = IPHI(width=int(iphi_width), center=self._embed(torch.tensor(center, dtype=torch.float32)).tolist(), aspect=aspect)
        self.fno = GeoFNO2d(
            in_channels=2 * self.n_fields,
            out_channels=self.n_fields,
            modes1=int(n_modes_x),
            modes2=int(n_modes_y),
            width=int(hidden_channels),
            s1=int(latent_Nx),
            s2=int(latent_Ny),
            n_layers=int(n_layers),
        )

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.domain_padding + (1.0 - 2.0 * self.domain_padding) * x

    def forward(
        self,
        coords_2d: torch.Tensor,
        obs_coords_2d: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        # coords_2d (batch or 1, N, 2) in [0, 1]; a single mesh is shared by the batch.
        valid = (obs_mask > 0).unsqueeze(-1)
        onehot = F.one_hot(obs_field_ids.clamp_min(0), self.n_fields).to(obs_values.dtype) * valid
        features = torch.cat([onehot * obs_values, onehot], dim=-1)
        # [diff] mean over each field's sensors instead of a raw sum: the sensor count
        # varies from sample to sample here, and the mean is the quadrature of the same
        # integral at every count.
        weights = (onehot / onehot.sum(dim=1, keepdim=True).clamp_min(1.0)).sum(dim=-1)
        return self.fno(features, self._embed(obs_coords_2d), weights, self._embed(coords_2d), self.iphi)
