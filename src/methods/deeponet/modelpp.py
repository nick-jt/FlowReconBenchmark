"""DeepONet++ -- structured-branch redesign of the DeepONet baseline
(ICLR cross-cube JHU sparse-reconstruction arm).

Motivation (pre-registered finding, FLEET_AUDIT_2026-08-29)
-----------------------------------------------------------
The upstream-faithful set-encoder DeepONet pools every sensor feature through a
single GLOBAL mean (deeponet_baseline.py:139): the pooled vector is 736 scalars
carrying no information about WHERE any sensor sits or WHAT it read locally.
The result was near-constant predictions (Ux 0.593 / Uz 0.652 rel_l2 at n=50).
That vanilla row stays in the table as the faithful reference; this module is
the labelled "DeepONet++ (structured branch)" variant.  It is still
recognisably a DeepONet: one branch producing n_fields*p coefficients, one
shared trunk basis of rank p, combined by an inner product plus a scalar bias.
No sensor ever attends to a query.

Branch redesign (primary + one, per Nick's spec)
------------------------------------------------
PRIMARY -- multiresolution binned pooling.  Per-sensor features
phi(x_s, v_s, e_{f_s}) are split into three chunks and pooled as
occupancy-weighted means over
    1^3 (global, 256 feats)  +  4^3 (64 bins, 24 feats)  +  8^3 (512 bins,
    6 feats)
spatial bins of the sensor cloud; each bin also appends its occupancy fraction.
The concatenated pooled vector is ~5.8k scalars that carry WHERE information
at two spatial scales on top of the global summary.

SECONDARY -- Fourier moments.  Per field f and integer mode k in {0..3}^3
(64 modes), the sensor-value-weighted moments
    M_{f,k} = sum_s m_s 1[fid_s=f] v_s exp(-2*pi*i k.x_s) / count_f
(real+imag, 4 fields x 64 x 2 = 512 scalars).  Chosen over an attention-pooling
head because the moments are PARAMETER-FREE, translation-structured (a shifted
sensor cloud phase-rotates the moments), and complement the bins exactly where
bins are weak: smooth large-scale structure that bin quantisation aliases.  An
attention head would add parameters, optimisation risk, and a second
set-encoder to debug, for information the bins already localise.

Each pooled block enters a per-block linear projection to a shared width C;
the four projections are summed (equivalent to one linear layer on the
concatenated pooled vector), ReLU'd, and a final LINEAR layer emits the
n_fields*p coefficients -- the same "1 hidden + linear out" rho shape as the
vanilla branch (fnn.py:66 convention: branch output linear).

Trunk redesign
--------------
Random Fourier features (Tancik et al. 2020): gamma(x) = [cos(2*pi*B x),
sin(2*pi*B x)] with B ~ N(0, sigma^2), sigma=8, 128 frequencies, plus the raw
affine coords -- then an upstream-style FNN with ACTIVATED output
(deeponet_strategy.py:35 convention kept).  RFF over SIREN because SIREN's
benefit hinges on its bespoke init and would interact untestably with the
upstream activated-output convention; an RFF-MLP is drop-in, robust, and
attacks the same spectral bias.  B is drawn once from a fixed-seed generator
and registered as a persistent buffer, so checkpoints reload bit-identically.

Conventions kept from upstream deepxde (unchanged from deeponet_baseline.py)
---------------------------------------------------------------------------
Glorot-normal weights / zero biases, ReLU, branch output linear, trunk output
activated, scalar per-output bias b init 0, split_branch multi-output, Adam
lr 1e-3 constant.

Interface: identical to DeepONetSetBranch (encode / trunk_forward / combine /
forward / n_fields / p / bottleneck_scalars / n_params), so
eval_deeponet_iclr.py drives it unchanged once its builder is swapped.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _fnn_layers(sizes: List[int]) -> nn.ModuleList:
    """Upstream deepxde FNN layer stack: Linear + Glorot-normal W + zero b."""
    layers = nn.ModuleList()
    for i in range(1, len(sizes)):
        lin = nn.Linear(sizes[i - 1], sizes[i])
        nn.init.xavier_normal_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
    return layers


def _glorot_linear(n_in: int, n_out: int) -> nn.Linear:
    lin = nn.Linear(n_in, n_out)
    nn.init.xavier_normal_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class DeepONetPP(nn.Module):
    """DeepONet with a spatially structured (multiresolution binned + Fourier
    moment) branch and an RFF-MLP trunk.

    forward(query_coords [B,Q,3], obs_coords [B,M,3], obs_values [B,M,1],
            obs_mask [B,M], obs_field_ids [B,M]) -> [B,Q,n_fields]
    """

    def __init__(
        self,
        n_fields: int = 4,
        coord_dim: int = 3,
        p: int = 768,
        field_embed_dim: int = 32,
        phi_width: int = 384,
        d_global: int = 256,
        bin_resolutions: tuple = (4, 8),
        bin_feat_dims: tuple = (24, 6),
        fourier_kmax: int = 4,          # modes 0..kmax-1 per axis
        merge_width: int = 544,
        rff_freqs: int = 128,
        rff_sigma: float = 8.0,
        trunk_width: int = 768,
        trunk_layers: int = 3,          # 2 hidden + output; output ACTIVATED
        rff_seed: int = 1234,
    ) -> None:
        super().__init__()
        assert len(bin_resolutions) == len(bin_feat_dims)
        self.n_fields = int(n_fields)
        self.coord_dim = int(coord_dim)
        self.p = int(p)
        self.bin_resolutions = tuple(int(r) for r in bin_resolutions)
        self.bin_feat_dims = tuple(int(d) for d in bin_feat_dims)
        self.d_global = int(d_global)
        self.fourier_kmax = int(fourier_kmax)
        self.merge_width = int(merge_width)

        # per-sensor tag: this protocol gives every channel an INDEPENDENT
        # mask, so each sensor carries one scalar + an embedding of its field.
        self.field_embed = nn.Embedding(self.n_fields, int(field_embed_dim))
        nn.init.normal_(self.field_embed.weight, mean=0.0, std=1.0)

        in_dim = self.coord_dim + 1 + int(field_embed_dim)
        d_head = self.d_global + sum(self.bin_feat_dims)
        self.phi = _fnn_layers([in_dim, phi_width, phi_width, d_head])

        # per-block linear projections into the shared merge width C
        self.proj_global = _glorot_linear(self.d_global, self.merge_width)
        self.proj_bins = nn.ModuleList([
            _glorot_linear(r ** 3 * (d + 1), self.merge_width)   # +1 occupancy
            for r, d in zip(self.bin_resolutions, self.bin_feat_dims)])
        n_modes = self.fourier_kmax ** 3
        self.proj_fourier = _glorot_linear(self.n_fields * n_modes * 2,
                                           self.merge_width)
        # rho: [sum of projections] -> ReLU -> LINEAR coefficients
        self.rho_out = _glorot_linear(self.merge_width, self.n_fields * self.p)

        # integer Fourier modes k in {0..kmax-1}^3, buffer (not trained)
        ks = torch.arange(self.fourier_kmax, dtype=torch.float32)
        kk = torch.stack(torch.meshgrid(ks, ks, ks, indexing="ij"),
                         dim=-1).reshape(-1, 3)                      # [n_modes,3]
        self.register_buffer("fourier_k", kk, persistent=False)

        # RFF trunk: fixed-seed Gaussian frequency matrix, persistent buffer
        g = torch.Generator().manual_seed(int(rff_seed))
        B = torch.randn(int(rff_freqs), self.coord_dim, generator=g) * float(rff_sigma)
        self.register_buffer("rff_B", B, persistent=True)
        trunk_in = 2 * int(rff_freqs) + self.coord_dim   # sin, cos, raw affine
        self.trunk = _fnn_layers([trunk_in] + [int(trunk_width)] *
                                 (int(trunk_layers) - 1) + [self.p])

        # upstream deeponet.py:121-123 -- one trainable scalar bias per output
        self.b = nn.Parameter(torch.zeros(self.n_fields))
        self.act = nn.ReLU()

    # -- branch --------------------------------------------------------------
    def _binned_pool(self, feats, coords, mask, res, offset, d):
        """Occupancy-weighted mean of feats[..., offset:offset+d] over res^3
        spatial bins; returns [B, res^3*(d+1)] (means + occupancy fraction)."""
        Bsz, M, _ = feats.shape
        nb = res ** 3
        ix = (coords * res).long().clamp_(0, res - 1)                # [B,M,3]
        bin_id = (ix[..., 0] * res + ix[..., 1]) * res + ix[..., 2]  # [B,M]
        flat = (torch.arange(Bsz, device=feats.device).unsqueeze(1) * nb
                + bin_id).reshape(-1)                                # [B*M]
        w = mask.to(feats.dtype).reshape(-1, 1)                      # [B*M,1]
        f = feats[..., offset:offset + d].reshape(Bsz * M, d) * w
        sums = feats.new_zeros(Bsz * nb, d).index_add_(0, flat, f)
        cnts = feats.new_zeros(Bsz * nb, 1).index_add_(0, flat, w)
        means = sums / cnts.clamp_min(1.0)
        total = mask.to(feats.dtype).sum(dim=1).clamp_min(1.0)       # [B]
        occ = cnts.view(Bsz, nb) / total.unsqueeze(1)                # fraction
        return torch.cat([means.view(Bsz, nb * d), occ], dim=-1)

    def _fourier_moments(self, coords, values, mask, fid):
        """Sensor-value-weighted Fourier modes of the sensor cloud, per field.
        Returns [B, n_fields * n_modes * 2]."""
        phase = 2.0 * torch.pi * (coords @ self.fourier_k.t())       # [B,M,K]
        cosp, sinp = torch.cos(phase), torch.sin(phase)
        out = []
        m = mask.to(coords.dtype)
        for f in range(self.n_fields):
            w = (m * (fid == f).to(coords.dtype)).unsqueeze(-1)      # [B,M,1]
            wv = w * values                                          # [B,M,1]
            cnt = w.sum(dim=1).clamp_min(1.0)                        # [B,1]
            re = (wv * cosp).sum(dim=1) / cnt                        # [B,K]
            im = (wv * sinp).sum(dim=1) / cnt
            out.extend([re, im])
        return torch.cat(out, dim=-1)

    def encode(self, obs_coords, obs_values, obs_mask, obs_field_ids):
        fid = obs_field_ids.clamp_min(0).long()
        emb = self.field_embed(fid)
        x = torch.cat([2.0 * obs_coords - 1.0, obs_values, emb], dim=-1)
        for lin in self.phi:
            x = self.act(lin(x))                                     # [B,M,d_head]

        m = obs_mask.to(x.dtype)
        gsum = (x[..., :self.d_global] * m.unsqueeze(-1)).sum(dim=1)
        gmean = gsum / m.sum(dim=1, keepdim=True).clamp_min(1.0)
        h = self.proj_global(gmean)

        offset = self.d_global
        for lin, res, d in zip(self.proj_bins, self.bin_resolutions,
                               self.bin_feat_dims):
            h = h + lin(self._binned_pool(x, obs_coords, obs_mask, res,
                                          offset, d))
            offset += d
        h = h + self.proj_fourier(self._fourier_moments(
            obs_coords, obs_values, obs_mask, fid))

        h = self.act(h)
        c = self.rho_out(h)                                          # LINEAR out
        return c.view(c.shape[0], self.n_fields, self.p)

    # -- trunk ---------------------------------------------------------------
    def trunk_forward(self, query_coords):
        proj = 2.0 * torch.pi * (query_coords @ self.rff_B.t())
        x = torch.cat([torch.cos(proj), torch.sin(proj),
                       2.0 * query_coords - 1.0], dim=-1)
        for lin in self.trunk[:-1]:
            x = self.act(lin(x))
        return self.act(self.trunk[-1](x))     # trunk OUTPUT activated (upstream)

    def combine(self, coeff, basis):
        return torch.einsum("bkp,bqp->bqk", coeff, basis) + self.b.view(1, 1, -1)

    def forward(self, query_coords, obs_coords, obs_values, obs_mask, obs_field_ids):
        coeff = self.encode(obs_coords, obs_values, obs_mask, obs_field_ids)
        basis = self.trunk_forward(query_coords)
        return self.combine(coeff, basis)

    # -- accounting ----------------------------------------------------------
    def bottleneck_scalars(self) -> int:
        return self.n_fields * self.p

    def pooled_scalars(self) -> int:
        """Size of the structured pooled representation (vs vanilla's 736)."""
        n_modes = self.fourier_kmax ** 3
        return (self.d_global
                + sum(r ** 3 * (d + 1) for r, d in zip(self.bin_resolutions,
                                                       self.bin_feat_dims))
                + self.n_fields * n_modes * 2)

    def n_params(self) -> int:
        return sum(q.numel() for q in self.parameters() if q.requires_grad)


def build_deeponetpp(arch: dict) -> DeepONetPP:
    return DeepONetPP(
        n_fields=int(arch.get("n_fields", 4)),
        coord_dim=int(arch.get("coord_dim", 3)),
        p=int(arch["p"]),
        field_embed_dim=int(arch.get("field_embed_dim", 32)),
        phi_width=int(arch.get("phi_width", 384)),
        d_global=int(arch.get("d_global", 256)),
        bin_resolutions=tuple(arch.get("bin_resolutions", [4, 8])),
        bin_feat_dims=tuple(arch.get("bin_feat_dims", [24, 6])),
        fourier_kmax=int(arch.get("fourier_kmax", 4)),
        merge_width=int(arch["merge_width"]),
        rff_freqs=int(arch.get("rff_freqs", 128)),
        rff_sigma=float(arch.get("rff_sigma", 8.0)),
        trunk_width=int(arch.get("trunk_width", 768)),
        trunk_layers=int(arch.get("trunk_layers", 3)),
        rff_seed=int(arch.get("rff_seed", 1234)),
    )


if __name__ == "__main__":
    TARGET = 6_506_253
    for p, C in [(384, 688), (640, 588), (768, 544)]:
        m = DeepONetPP(p=p, merge_width=C)
        n = m.n_params()
        print(f"p={p:4d} C={C:3d}: params={n} ({100*(n-TARGET)/TARGET:+.3f}% of "
              f"{TARGET}) pooled={m.pooled_scalars()} bottleneck="
              f"{m.bottleneck_scalars()} rank_cap={min(p, C, 768)}")
