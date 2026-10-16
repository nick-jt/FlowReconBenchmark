"""Spectrally-matched Gaussian-process source for rectified flow on fields.

The default RFF source draws angular frequencies from an isotropic Gaussian
with scale 1/lengthscale, i.e. a squared-exponential kernel concentrated at
one scale. At the lengthscale used for the turbulence runs (0.15 of the
domain) essentially all of the source's energy sits near wavenumber one,
while the data carries structure out to the Nyquist wavenumber. The flow then
has to synthesise every small scale from nothing, which is hardest exactly
where the observations say least -- an unobserved channel, whose sample is
otherwise close to a draw from this very smooth prior.

`PowerLawRFFPrior` keeps the random-feature construction (so the source is
still one consistent function evaluable at any point set, which is what makes
training and sampling discretisation-independent) but draws the radial
frequency from a Kolmogorov-like law instead of a Gaussian. For an isotropic
field in 3D with shell-averaged energy spectrum E(k) ~ k^(-slope), the radial
density of the random-feature frequencies is p(k) ~ k^(-slope), sampled on
[k_min, k_max] by inverse CDF with directions uniform on the sphere.

Setting slope=5/3 gives an inertial-range source; slope=0 recovers a
white-noise-like source band-limited to k_max, which is the usual choice in
image diffusion and is provided for ablation.
"""

import math

import torch
import torch.nn as nn


def _sample_powerlaw_radii(n: int, k_min: float, k_max: float, slope: float,
                           generator=None) -> torch.Tensor:
    """Draw radii with density p(k) proportional to k^(-slope) on [k_min, k_max]."""
    u = torch.rand(n, generator=generator)
    if abs(1.0 - slope) < 1e-6:                      # p(k) ~ 1/k -> log-uniform
        return k_min * (k_max / k_min) ** u
    a = 1.0 - slope
    return (k_min ** a + u * (k_max ** a - k_min ** a)) ** (1.0 / a)


class PowerLawRFFPrior(nn.Module):
    """Stationary GP source with a power-law energy spectrum."""

    def __init__(self, coord_dim: int = 3, n_features: int = 1024,
                 slope: float = 5.0 / 3.0, k_min: float = 1.0,
                 k_max: float = 32.0, seed: int = 0):
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.n_features = int(n_features)
        self.slope = float(slope)
        self.k_min = float(k_min)
        self.k_max = float(k_max)

        g = torch.Generator().manual_seed(int(seed))
        radii = _sample_powerlaw_radii(self.n_features, self.k_min, self.k_max,
                                       self.slope, generator=g)
        direction = torch.randn(self.coord_dim, self.n_features, generator=g)
        direction = direction / direction.norm(dim=0, keepdim=True).clamp_min(1e-12)
        # coords are normalised to [0, 1], so a wavenumber of k cycles across the
        # domain corresponds to an angular frequency of 2*pi*k.
        self.register_buffer("omega", direction * (2.0 * math.pi * radii)[None, :])
        self.register_buffer("phase", 2 * math.pi * torch.rand(self.n_features,
                                                               generator=g))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords[..., : self.coord_dim] @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int,
                chunk: int = 262_144) -> torch.Tensor:
        # A richer spectrum needs more random features, and the feature matrix
        # is [B, N, F] -- at full-field sampling resolution that would dominate
        # peak memory, so the draw is chunked over points. The weights are
        # drawn once per call so the result is still one coherent GP sample.
        bsz, n_pts, _ = coords.shape
        w = torch.randn(bsz, n_channels, self.n_features, device=coords.device,
                        dtype=coords.dtype)
        out = torch.empty(bsz, n_pts, n_channels, device=coords.device,
                          dtype=coords.dtype)
        for s in range(0, n_pts, chunk):
            e = min(s + chunk, n_pts)
            out[:, s:e] = torch.einsum("bnf,bcf->bnc",
                                       self._features(coords[:, s:e]), w)
        return out

    def extra_repr(self) -> str:
        return (f"n_features={self.n_features}, slope={self.slope:.3f}, "
                f"k_min={self.k_min}, k_max={self.k_max}")
