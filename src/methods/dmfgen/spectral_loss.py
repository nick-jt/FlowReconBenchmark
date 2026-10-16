"""Binned spectral power loss for 3D field reconstruction.

A pointwise regression loss is dominated by the energy-containing scales: in
3D turbulence the inertial and dissipation ranges carry orders of magnitude
less power, so matching them buys almost nothing in MSE and the model is free
to under-resolve them. That is exactly the deficit we measure on JHU, where the
reconstructed inertial band holds only 0.61-0.74 of the true energy.

The fix that works is comparing *binned* power in radial wavenumber shells on a
log scale, so every decade of scale contributes comparably. Comparing raw
Fourier coefficients instead does not help -- it reweights the same pointwise
information and stays dominated by the large scales.

Two properties make this cheap for us:

  * The loss is evaluated on a contiguous block taken in GRID-INDEX space, so
    it is unaffected by the symmetry augmentations. Rotating or translating the
    point cloud changes coordinates, not the grid topology, and a radially
    binned spectrum is rotation invariant by construction.
  * Flow matching gives a differentiable endpoint for free. With a straight
    path x_t = (1-t)x_0 + t x_1 and predicted velocity v, the implied clean
    field is x_t + (1-t)v, so no sampling loop is needed inside training.

The estimate is exact at t -> 1 and noisiest at t -> 0, where the (1-t) factor
amplifies velocity error; `t_weighting` down-weights the noisy end.
"""

from typing import Optional, Sequence, Tuple

import torch


def block_indices(
    grid_shape: Sequence[int],
    block: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Flat indices of a random contiguous block**3 sub-cube of a C-ordered grid."""
    gx, gy, gz = (int(s) for s in grid_shape)
    b = int(block)
    if b > min(gx, gy, gz):
        raise ValueError(f"block {b} exceeds grid {grid_shape}")

    def start(limit: int) -> int:
        if limit == b:
            return 0
        return int(torch.randint(0, limit - b + 1, (1,), device="cpu",
                                 generator=generator).item())

    i0, j0, k0 = start(gx), start(gy), start(gz)
    ii = torch.arange(i0, i0 + b, device=device)
    jj = torch.arange(j0, j0 + b, device=device)
    kk = torch.arange(k0, k0 + b, device=device)
    flat = ((ii[:, None, None] * gy + jj[None, :, None]) * gz + kk[None, None, :])
    return flat.reshape(-1), (b, b, b)


def _shell_bins(block: int, n_bins: int, device: torch.device) -> Tuple[torch.Tensor, int]:
    """Assign each rfft mode to a radial shell; returns bin ids and bin count."""
    kx = torch.fft.fftfreq(block, device=device) * block
    ky = torch.fft.fftfreq(block, device=device) * block
    kz = torch.fft.rfftfreq(block, device=device) * block
    kmag = torch.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2
                      + kz[None, None, :] ** 2)
    # Truncate at the Nyquist radius: corner modes beyond it are only partially
    # sampled and produce spurious ratios (this bit us in the spectral report).
    k_nyq = block / 2.0
    edges = torch.linspace(0.0, k_nyq, n_bins + 1, device=device)
    bin_id = torch.bucketize(kmag.reshape(-1), edges[1:-1].contiguous())
    bin_id = bin_id.masked_fill(kmag.reshape(-1) > k_nyq, n_bins)  # dumped, then dropped
    return bin_id, n_bins


_HANN_CACHE: dict = {}


def _hann_window(bx: int, by: int, bz: int, device, dtype) -> torch.Tensor:
    """Separable Hann window, variance-preserving, cached per shape/device."""
    key = (bx, by, bz, str(device), str(dtype))
    w = _HANN_CACHE.get(key)
    if w is None:
        wx = torch.hann_window(bx, periodic=False, device=device, dtype=dtype)
        wy = torch.hann_window(by, periodic=False, device=device, dtype=dtype)
        wz = torch.hann_window(bz, periodic=False, device=device, dtype=dtype)
        w = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        w = w / w.pow(2).mean().sqrt()
        _HANN_CACHE[key] = w
    return w


def binned_power(
    field: torch.Tensor,
    block_shape: Sequence[int],
    n_bins: int = 12,
    window: bool = False,
) -> torch.Tensor:
    """Radially binned power spectrum. field: [B, N, C] with N = prod(block_shape).

    The block is a sub-volume of a larger field and is therefore not periodic.
    Without `window`, the FFT spreads the edge discontinuity across every
    shell: on our data the top shells of a 32^3 block read ~100x their true
    power, so those bins match leakage rather than physics and the loss gives
    almost no high-wavenumber signal. `window` applies a variance-preserving
    Hann taper first, which restores the true decay.

    Returns [B, C, n_bins] of mean power per shell, in float32.
    """
    b, n, c = field.shape
    bx, by, bz = (int(s) for s in block_shape)
    x = field.reshape(b, bx, by, bz, c).permute(0, 4, 1, 2, 3).float()
    if window:
        x = x * _hann_window(bx, by, bz, x.device, x.dtype)[None, None]
    spec = torch.fft.rfftn(x, dim=(2, 3, 4))
    power = spec.real ** 2 + spec.imag ** 2                     # [B, C, bx, by, bz//2+1]
    power = power.reshape(b, c, -1)

    bin_id, nb = _shell_bins(bx, n_bins, field.device)
    keep = bin_id < nb
    bid = bin_id[keep]
    pw = power[..., keep]

    out = torch.zeros(b, c, nb, device=field.device, dtype=torch.float32)
    cnt = torch.zeros(nb, device=field.device, dtype=torch.float32)
    out.index_add_(2, bid, pw)
    cnt.index_add_(0, bid, torch.ones_like(bid, dtype=torch.float32))
    return out / cnt.clamp_min(1.0)


def binned_spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    block_shape: Sequence[int],
    n_bins: int = 12,
    eps: float = 1e-8,
    sample_weight: Optional[torch.Tensor] = None,
    clamp: float = 4.0,
    window: bool = False,
) -> torch.Tensor:
    """Mean squared log-power discrepancy over radial shells.

    `sample_weight` ([B]) down-weights samples whose endpoint estimate is
    unreliable; passing t is the natural choice, since the (1-t) factor in
    x_t + (1-t)v amplifies velocity error as t -> 0. `clamp` bounds the
    per-shell log discrepancy so an early-training prediction with almost no
    power in the dissipation range cannot produce an unbounded gradient.
    """
    p = binned_power(pred, block_shape, n_bins, window=window)
    q = binned_power(target, block_shape, n_bins, window=window)
    d = (torch.log(p + eps) - torch.log(q + eps)).clamp(-clamp, clamp)
    per_sample = (d ** 2).mean(dim=(1, 2))                      # [B]
    if sample_weight is not None:
        w = sample_weight.to(per_sample.dtype).clamp_min(0.0)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-8)
    return per_sample.mean()
