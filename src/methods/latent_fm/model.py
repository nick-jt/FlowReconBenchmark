# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from typing import Sequence
from typing import Tuple
from methods.common import helpers_baseline as BASELINE_HELPERS


# ═════════ §7. Latent FM (stage 1 AE + stage 2) ═════════

# -----------------------------------------------------------------------------
# Stage 1: Convolutional Autoencoder
# -----------------------------------------------------------------------------

def _gn_groups(channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g

class _ResBlock2d(nn.Module):
    """GN-SiLU-Conv3x3 -> GN-SiLU-Conv3x3 residual block at fixed channel count.

    state_dict layout (matches checkpoint):
        block.0 : GroupNorm(channels)
        block.1 : SiLU
        block.2 : Conv2d(channels, channels, 3, padding=1)
        block.3 : GroupNorm(channels)
        block.4 : SiLU
        block.5 : Conv2d(channels, channels, 3, padding=1)
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        g = _gn_groups(channels, groups)
        self.block = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class ConvAE(nn.Module):
    """UNet-style convolutional autoencoder used as the Stage-1 latent space.

    state_dict layout (from best.pt):
        encoder.0                : Conv2d(n_fields, base_ch, 4, 2, 1)   # downsample #1
        encoder.{1..num_res}     : _ResBlock2d(base_ch)
        encoder.{k}              : Conv2d(ch_i, ch_{i+1}, 4, 2, 1)      # downsample
        ... repeated n_levels times ...
        decoder (mirror): ResBlocks, ConvTranspose2d, ..., final ConvT -> n_fields.
    Channel schedule: c_i = min(base_ch * 2**i, latent_ch) for i in 0..n_levels.
    """

    def __init__(self, n_fields: int, base_ch: int = 64, latent_ch: int = 128,
                 n_levels: int = 3, Num_y: int = 100, Num_x: int = 403,
                 deform_coord_dim: int = 0, deform_hidden: int = 128,
                 deform_depth: int = 3, num_res_blocks: int = 2):
        super().__init__()
        self.n_fields = n_fields
        self.base_ch = base_ch
        self.latent_ch = latent_ch
        self.n_levels = n_levels
        self.Num_y = Num_y
        self.Num_x = Num_x
        self.num_res_blocks = num_res_blocks

        # Padding so H, W are divisible by 2**n_levels.
        factor = 2 ** n_levels
        self.H_pad = int(math.ceil(Num_y / factor) * factor)
        self.W_pad = int(math.ceil(Num_x / factor) * factor)

        # Channel schedule.
        chs = [min(base_ch * (2 ** i), latent_ch) for i in range(n_levels + 1)]
        # chs[0] is input-side channel after first conv; chs[-1] == latent_ch.

        # --- Encoder ---
        enc = []
        # encoder.0 : first downsample conv from n_fields to chs[0]
        enc.append(nn.Conv2d(n_fields, chs[0], kernel_size=4, stride=2, padding=1))
        # After first downsample we have chs[0] channels; place num_res_blocks at chs[0].
        for _ in range(num_res_blocks):
            enc.append(_ResBlock2d(chs[0]))
        # Then for each subsequent level: downsample conv then num_res_blocks.
        for i in range(1, n_levels):
            enc.append(nn.Conv2d(chs[i - 1], chs[i], kernel_size=4, stride=2,
                                 padding=1))
            for _ in range(num_res_blocks):
                enc.append(_ResBlock2d(chs[i]))
        self.encoder = nn.Sequential(*enc)

        # --- Decoder (mirrors encoder) ---
        dec = []
        # Start at deepest level (chs[n_levels-1]) with num_res_blocks.
        for _ in range(num_res_blocks):
            dec.append(_ResBlock2d(chs[n_levels - 1]))
        # For each level going up: ConvTranspose then num_res_blocks at lower ch.
        for i in range(n_levels - 1, 0, -1):
            dec.append(nn.ConvTranspose2d(chs[i], chs[i - 1], kernel_size=4,
                                          stride=2, padding=1))
            for _ in range(num_res_blocks):
                dec.append(_ResBlock2d(chs[i - 1]))
        # Final ConvTranspose back to n_fields.
        dec.append(nn.ConvTranspose2d(chs[0], n_fields, kernel_size=4, stride=2,
                                      padding=1))
        self.decoder = nn.Sequential(*dec)

        # --- Optional coordinate deformer (for irregular meshes) ---
        if deform_coord_dim and deform_coord_dim > 0:
            layers = []
            d_in = deform_coord_dim
            for _ in range(deform_depth - 1):
                layers += [nn.Linear(d_in, deform_hidden), nn.SiLU()]
                d_in = deform_hidden
            final = nn.Linear(d_in, 2)
            # Zero-init final layer so `deform(coords) = sigmoid(coords[...,:2]
            # would collapse to ~0.5 everywhere at init. Instead the deformer
            # is used as a residual perturbation (see deform()), with final
            # layer zero-initialized so it starts as the identity map on the
            # first two coord components. This guarantees x_grid is non-trivial
            # from epoch 0, giving a real reconstruction signal to learn from.
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
            self.deformer = nn.Sequential(*layers)
        else:
            self.deformer = None

    # --------- Padding helpers ---------
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        pad_h = self.H_pad - H
        pad_w = self.W_pad - W
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :self.Num_y, :self.Num_x]

    # --------- Core ops ---------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def deform(self, coords: torch.Tensor) -> torch.Tensor:
        assert self.deformer is not None, "ConvAE was built without a deformer"
        # Residual deformer with clamp to [0,1]. splat_to_grid and
        # gather_from_grid both expect normalized [0,1]^2 coords per their
        # docstrings; without bounding, an unconstrained nn.Linear output
        # puts bilinear weights outside [0,1], making scatter-add accumulate
        # wildly large values (loss ~1e11 at epoch 0). Dataset coords are
        # already in [0,1]^3 (z=0 padding), so using the first two channels
        # as the base map and letting the MLP learn a small perturbation is
        # both well-conditioned and matches the reference LatentFM design
        # (which had no deformer at all on regular grids).
        delta = self.deformer(coords)
        base = coords[..., :2]
        return (base + delta).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class _ResBlock3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_gn_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_gn_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvAE3D(nn.Module):
    """3D convolutional autoencoder for volumetric LatentFM inputs."""

    def __init__(self, n_fields: int, base_ch: int = 32, latent_ch: int = 96,
                 n_levels: int = 3, Num_z: int = 125, Num_y: int = 125,
                 Num_x: int = 125, num_res_blocks: int = 1):
        super().__init__()
        self.n_fields = n_fields
        self.base_ch = base_ch
        self.latent_ch = latent_ch
        self.n_levels = n_levels
        self.Num_z = Num_z
        self.Num_y = Num_y
        self.Num_x = Num_x
        self.num_res_blocks = num_res_blocks
        self.deformer = None

        factor = 2 ** n_levels
        self.D_pad = int(math.ceil(Num_z / factor) * factor)
        self.H_pad = int(math.ceil(Num_y / factor) * factor)
        self.W_pad = int(math.ceil(Num_x / factor) * factor)

        chs = [min(base_ch * (2 ** i), latent_ch) for i in range(n_levels + 1)]

        enc = [nn.Conv3d(n_fields, chs[0], kernel_size=4, stride=2, padding=1)]
        for _ in range(num_res_blocks):
            enc.append(_ResBlock3d(chs[0]))
        for i in range(1, n_levels):
            enc.append(nn.Conv3d(chs[i - 1], chs[i], kernel_size=4, stride=2, padding=1))
            for _ in range(num_res_blocks):
                enc.append(_ResBlock3d(chs[i]))
        self.encoder = nn.Sequential(*enc)

        dec = []
        for _ in range(num_res_blocks):
            dec.append(_ResBlock3d(chs[n_levels - 1]))
        for i in range(n_levels - 1, 0, -1):
            dec.append(nn.ConvTranspose3d(chs[i], chs[i - 1], kernel_size=4, stride=2, padding=1))
            for _ in range(num_res_blocks):
                dec.append(_ResBlock3d(chs[i - 1]))
        dec.append(nn.ConvTranspose3d(chs[0], n_fields, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*dec)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, D, H, W = x.shape
        pad = (0, self.W_pad - W, 0, self.H_pad - H, 0, self.D_pad - D)
        if all(v == 0 for v in pad):
            return x
        return F.pad(x, pad)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._pad(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

# -----------------------------------------------------------------------------
# Stage 2: Latent Flow Matching UNet
# -----------------------------------------------------------------------------

class _TimestepEmbedding(nn.Module):
    """Sinusoidal(t) -> Linear -> SiLU -> Linear  (emb_dim -> emb_dim)."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.to(t.dtype if t.is_floating_point() else torch.float32)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))

class _AdaGNResBlock(nn.Module):
    """Residual block with AdaGN-style FiLM conditioning on the time embedding.

    state_dict layout (matches checkpoint):
        norm1     : GroupNorm(channels)
        conv1     : Conv2d(channels, channels, 3, padding=1)
        norm2     : GroupNorm(channels)
        conv2     : Conv2d(channels, channels, 3, padding=1)
        emb_proj  : Linear(emb_dim, 2*channels)  # scale, shift
    """

    def __init__(self, channels: int, emb_dim: int, groups: int = 8):
        super().__init__()
        g = _gn_groups(channels, groups)
        self.channels = channels
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return x + h

class PointNetSensorEncoder(nn.Module):
    """PointNet-style encoder for sparse sensor tokens.

    Adapted from Kashefi et al. (arXiv:2601.03030) "Flow Matching and Diffusion
    Models via PointNet for Generating Fluid Fields on Irregular Geometries":
    shared Conv1d MLPs on per-point tokens followed by a masked global
    max-pool, producing a permutation-invariant sample-level feature.

    To keep the velocity UNet's channel layout identical to the masked-image
    conditioner, the global feature is projected through two heads and tiled
    across the latent (h, w) grid to fill the same (cond_feat, cond_mask)
    slots that LatentFMUNet already consumes.
    """

    def __init__(self, n_fields: int, coord_dim: int = 2,
                 hidden_mult: float = 1.0):
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.out_feat_ch = n_fields
        self.out_mask_ch = n_fields + 1

        # Per-token input: [coords (D), value (1), field one-hot (F), mask (1)]
        in_ch = coord_dim + 1 + n_fields + 1
        m = hidden_mult
        c1, c2, c3 = int(64 * m), int(128 * m), int(1024 * m)

        def _gn(ch):
            return nn.GroupNorm(_gn_groups(ch), ch)

        self.conv1 = nn.Conv1d(in_ch, c1, 1)
        self.n1 = _gn(c1)
        self.conv2 = nn.Conv1d(c1, c1, 1)
        self.n2 = _gn(c1)
        self.conv3 = nn.Conv1d(c1, c2, 1)
        self.n3 = _gn(c2)
        self.conv4 = nn.Conv1d(c2, c3, 1)
        self.n4 = _gn(c3)

        self.head_feat = nn.Sequential(
            nn.Linear(c3, c2), nn.SiLU(), nn.Linear(c2, self.out_feat_ch),
        )
        self.head_mask = nn.Sequential(
            nn.Linear(c3, c2), nn.SiLU(), nn.Linear(c2, self.out_mask_ch),
        )

    def forward(self, obs_coords_2d: torch.Tensor,
                obs_values: torch.Tensor,
                obs_mask: torch.Tensor,
                obs_field_ids: torch.Tensor,
                h: int, w: int):
        B, M = obs_mask.shape
        dtype = obs_values.dtype
        fld = obs_field_ids.clamp(min=0).long()
        fld_oh = F.one_hot(fld, num_classes=self.n_fields).to(dtype)
        fld_oh = fld_oh * obs_mask.unsqueeze(-1)

        tok = torch.cat(
            [obs_coords_2d.to(dtype), obs_values, fld_oh,
             obs_mask.unsqueeze(-1)],
            dim=-1,
        )  # [B, M, in_ch]
        x = tok.transpose(1, 2)  # [B, in_ch, M]

        x = F.relu(self.n1(self.conv1(x)))
        x = F.relu(self.n2(self.conv2(x)))
        x = F.relu(self.n3(self.conv3(x)))
        x = F.relu(self.n4(self.conv4(x)))  # [B, c3, M]

        # Masked global max-pool (ignore padded sensor slots).
        mask_b1m = obs_mask.unsqueeze(1) > 0.5
        neg_inf = torch.finfo(x.dtype).min
        x = x.masked_fill(~mask_b1m, neg_inf)
        g = x.amax(dim=-1)  # [B, c3]
        # Safety: if a sample has zero valid sensors, max returns -inf.
        g = torch.where(torch.isneginf(g), torch.zeros_like(g), g)

        cond_feat = self.head_feat(g).view(B, -1, 1, 1).expand(-1, -1, h, w).contiguous()
        cond_mask = self.head_mask(g).view(B, -1, 1, 1).expand(-1, -1, h, w).contiguous()
        return cond_feat, cond_mask


class LatentFMUNet(nn.Module):
    """Velocity network operating in the AE's latent grid.

    Sparse-field-as-image conditioning: the masked sparse field and per-field
    mask are stacked as extra input channels at the denoiser's resolution
    (rather than encoded through the frozen AE). Input channels of in_conv:

        latent_ch (z_t) + n_fields (downsampled masked sparse field)
                        + n_fields + 1 (per-field mask + any-sensor aggregate)
    """

    def __init__(self, latent_ch: int, n_fields: int, base_ch: int = 256,
                 ch_mult: Sequence[int] = (1, 2), num_res_blocks: int = 2,
                 num_heads: int = 8, attn_at_level: Optional[int] = None):
        super().__init__()
        self.latent_ch = latent_ch
        self.n_fields = n_fields
        self.base_ch = base_ch
        self.ch_mult = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_heads = num_heads
        # Attention is placed at the bottleneck by default.
        self.attn_at_level = attn_at_level

        emb_dim = base_ch
        self.time_embed = _TimestepEmbedding(emb_dim)

        in_ch = latent_ch + n_fields + (n_fields + 1)
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        channels = [base_ch * m for m in self.ch_mult]
        n_lv = len(channels)

        # --- Down path ---
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        prev_ch = base_ch
        for i, ch in enumerate(channels):
            # If ch differs from prev_ch, the first resblock would need a
            # channel change; for the shipped configs channels[0]==base_ch so
            # the first level is identity-in-channel.
            assert ch == prev_ch or i > 0
            blocks = nn.ModuleList(
                [_AdaGNResBlock(ch, emb_dim) for _ in range(num_res_blocks)])
            self.down_blocks.append(blocks)
            if i < n_lv - 1:
                # Downsample conv lifts to next level's channel count.
                self.down_convs.append(
                    nn.Conv2d(ch, channels[i + 1], kernel_size=4,
                              stride=2, padding=1))
                prev_ch = channels[i + 1]
            else:
                prev_ch = ch

        # --- Middle ---
        mid_ch = channels[-1]
        self.mid_res1 = _AdaGNResBlock(mid_ch, emb_dim)
        self.mid_attn = nn.MultiheadAttention(mid_ch, num_heads,
                                              batch_first=True)
        self.mid_attn_norm = nn.GroupNorm(_gn_groups(mid_ch), mid_ch)
        self.mid_res2 = _AdaGNResBlock(mid_ch, emb_dim)

        # --- Up path (mirror) ---
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(n_lv):
            # Level index from deepest (0) to shallowest (n_lv-1).
            lv = n_lv - 1 - i  # actual resolution level this up block finishes at
            skip_ch = channels[lv]
            if i == 0:
                # First up block: mid_ch + skip_ch (both == channels[-1])
                in_blk = mid_ch + skip_ch
            else:
                # After previous up_conv we are at channels[lv]; concat skip.
                in_blk = channels[lv] + skip_ch
            blocks = nn.ModuleList(
                [_AdaGNResBlock(in_blk, emb_dim) for _ in range(num_res_blocks)])
            self.up_blocks.append(blocks)
            if i < n_lv - 1:
                # Up-sample back to next shallower channel count.
                self.up_convs.append(
                    nn.ConvTranspose2d(in_blk, channels[lv - 1],
                                       kernel_size=4, stride=2, padding=1))

        # --- Output head ---
        final_ch = None
        # After last up_blocks iteration we are at in_blk of that iteration.
        if n_lv == 1:
            final_ch = mid_ch + channels[0]
        else:
            final_ch = channels[0] + channels[0]  # last level: up_conv_out + skip
        self.out_conv = nn.Sequential(
            nn.GroupNorm(_gn_groups(final_ch), final_ch),
            nn.SiLU(),
            nn.Conv2d(final_ch, latent_ch, kernel_size=3, padding=1),
        )

    def forward(self, t: torch.Tensor, z_t: torch.Tensor,
                z_obs: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        t      : [B] in [0, 1]
        z_t    : [B, latent_ch, h, w]
        z_obs  : [B, latent_ch, h, w]
        z_mask : [B, n_fields + 1, h, w]  (per-field + any-sensor)
        """
        emb = self.time_embed(t)
        h = self.in_conv(torch.cat([z_t, z_obs, z_mask], dim=1))

        skips = []
        n_lv = len(self.down_blocks)
        for i, blocks in enumerate(self.down_blocks):
            for blk in blocks:
                h = blk(h, emb)
            skips.append(h)
            if i < n_lv - 1:
                h = self.down_convs[i](h)

        # --- Middle with self-attention ---
        h = self.mid_res1(h, emb)
        B, C, H, W = h.shape
        h_norm = self.mid_attn_norm(h)
        tokens = h_norm.flatten(2).transpose(1, 2)  # [B, H*W, C]
        attn_out, _ = self.mid_attn(tokens, tokens, tokens, need_weights=False)
        h = h + attn_out.transpose(1, 2).reshape(B, C, H, W)
        h = self.mid_res2(h, emb)

        # --- Up path ---
        # Resize `h` to match each skip's spatial extent before concat. With
        # Conv2d(k=4, s=2, p=1) down and ConvTranspose2d(k=4, s=2, p=1) up,
        # odd input dims floor on the way down (e.g., 13 -> 6) and double
        # back (6 -> 12), so we need a 1-cell correction when the original
        # latent dim was odd. Nearest-neighbor interpolate is weight-free and
        # keeps checkpoint compatibility.
        for i, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, emb)
            if i < n_lv - 1:
                h = self.up_convs[i](h)

        return self.out_conv(h)


class _AdaGNResBlock3D(nn.Module):
    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn_groups(channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_gn_groups(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.emb(emb).chunk(2, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv1(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class LatentFMUNet3D(nn.Module):
    """Compact 3D velocity network operating on AE latent volumes."""

    def __init__(self, latent_ch: int, n_fields: int, base_ch: int = 128,
                 ch_mult: Sequence[int] = (1, 2), num_res_blocks: int = 1,
                 num_heads: int = 8):
        super().__init__()
        self.latent_ch = latent_ch
        self.n_fields = n_fields
        self.base_ch = base_ch
        self.ch_mult = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_heads = num_heads

        emb_dim = base_ch
        self.time_embed = _TimestepEmbedding(emb_dim)
        in_ch = latent_ch + n_fields + (n_fields + 1)
        self.in_conv = nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1)

        channels = [base_ch * m for m in self.ch_mult]
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        prev_ch = base_ch
        for i, ch in enumerate(channels):
            if ch != prev_ch:
                self.down_convs.append(nn.Conv3d(prev_ch, ch, kernel_size=1))
            else:
                self.down_convs.append(nn.Identity())
            self.down_blocks.append(nn.ModuleList([_AdaGNResBlock3D(ch, emb_dim) for _ in range(num_res_blocks)]))
            prev_ch = ch
            if i < len(channels) - 1:
                self.down_convs.append(nn.Conv3d(ch, channels[i + 1], kernel_size=4, stride=2, padding=1))
                prev_ch = channels[i + 1]

        mid_ch = channels[-1]
        self.mid_res1 = _AdaGNResBlock3D(mid_ch, emb_dim)
        self.mid_res2 = _AdaGNResBlock3D(mid_ch, emb_dim)

        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(len(channels)):
            lv = len(channels) - 1 - i
            skip_ch = channels[lv]
            in_blk = (channels[lv] if i > 0 else mid_ch) + skip_ch
            self.up_blocks.append(nn.ModuleList([_AdaGNResBlock3D(in_blk, emb_dim) for _ in range(num_res_blocks)]))
            if i < len(channels) - 1:
                self.up_convs.append(nn.ConvTranspose3d(in_blk, channels[lv - 1], kernel_size=4, stride=2, padding=1))

        final_ch = (channels[0] + channels[0]) if len(channels) > 1 else (mid_ch + channels[0])
        self.out_conv = nn.Sequential(
            nn.GroupNorm(_gn_groups(final_ch), final_ch),
            nn.SiLU(),
            nn.Conv3d(final_ch, latent_ch, kernel_size=3, padding=1),
        )

    def forward(self, t: torch.Tensor, z_t: torch.Tensor,
                z_obs: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t)
        h = self.in_conv(torch.cat([z_t, z_obs, z_mask], dim=1))
        skips = []
        conv_idx = 0
        for i, blocks in enumerate(self.down_blocks):
            h = self.down_convs[conv_idx](h)
            conv_idx += 1
            for blk in blocks:
                h = blk(h, emb)
            skips.append(h)
            if i < len(self.down_blocks) - 1:
                h = self.down_convs[conv_idx](h)
                conv_idx += 1

        h = self.mid_res2(self.mid_res1(h, emb), emb)

        for i, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-3:] != skip.shape[-3:]:
                h = F.interpolate(h, size=skip.shape[-3:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, emb)
            if i < len(self.up_convs):
                h = self.up_convs[i](h)
        return self.out_conv(h)

class LatentFlowMatching(nn.Module):
    """Rectified-Flow training & Euler-ODE sampling in the AE latent space."""

    def __init__(self, ae: ConvAE, velocity_net: LatentFMUNet,
                 Num_x: int, Num_y: int,
                 cond_mode: str = "image",
                 pointnet_encoder: Optional["PointNetSensorEncoder"] = None):
        super().__init__()
        self.ae = ae
        self.velocity_net = velocity_net
        self.Num_x = Num_x
        self.Num_y = Num_y
        self.n_fields = ae.n_fields
        self.latent_ch = ae.latent_ch
        self.n_levels = ae.n_levels
        self.spatial_dim = 3 if isinstance(ae, ConvAE3D) else 2

        if cond_mode not in ("image", "interp", "pointnet", "image_norm"):
            raise ValueError(
                f"cond_mode must be 'image', 'interp', 'pointnet' or 'image_norm', got {cond_mode!r}")
        if self.spatial_dim == 3 and cond_mode not in ("image", "image_norm"):
            raise ValueError("3D LatentFlowMatching currently supports cond_mode='image' only.")
        self.cond_mode = cond_mode
        if cond_mode == "pointnet":
            if pointnet_encoder is None:
                raise ValueError("cond_mode='pointnet' requires pointnet_encoder.")
            self.pointnet = pointnet_encoder
        else:
            self.pointnet = None
        # Fidelity repair 2 (latent rescaling): the rectified flow's source is N(0, I)
        # but AE latents are not unit scale. The adapter sets this to 1/std(z) (global
        # or per channel) when `latent_scale_mode` is configured; it stays 1 otherwise,
        # in which case the original code path runs unchanged.
        self.register_buffer("latent_scale", torch.ones(1))

    def _latent_scale_active(self) -> bool:
        return bool((self.latent_scale != 1).any())

    # --- Condition downsampling: sparse-field-as-image conditioning ---
    def _pad_to_ae(self, x: torch.Tensor) -> torch.Tensor:
        """Right/bottom zero-pad to (ae.H_pad, ae.W_pad)."""
        if self.spatial_dim == 3:
            return F.pad(
                x,
                (0, self.ae.W_pad - x.shape[-1],
                 0, self.ae.H_pad - x.shape[-2],
                 0, self.ae.D_pad - x.shape[-3]),
            )
        return F.pad(x, (0, self.ae.W_pad - x.shape[-1],
                         0, self.ae.H_pad - x.shape[-2]))

    def _downsample_mask(self, mask_grid: torch.Tensor) -> torch.Tensor:
        """mask_grid: [B, n_fields, H, W]  (any H, W <= AE's H_pad/W_pad).
        Returns [B, n_fields + 1, h, w] at latent resolution: per-field
        max-pool (preserves "any sensor in this cell") + any-sensor aggregate.
        """
        factor = 2 ** self.n_levels
        m = self._pad_to_ae(mask_grid)
        pool = F.max_pool3d if self.spatial_dim == 3 else F.max_pool2d
        m_ds = pool(m, kernel_size=factor, stride=factor)
        any_m = m_ds.amax(dim=1, keepdim=True)
        return torch.cat([m_ds, any_m], dim=1)

    def _latent_hw(self) -> Tuple[int, int]:
        factor = 2 ** self.n_levels
        if self.spatial_dim == 3:
            return self.ae.D_pad // factor, self.ae.H_pad // factor, self.ae.W_pad // factor
        return self.ae.H_pad // factor, self.ae.W_pad // factor

    def _encode_condition(self, cond_inputs: dict):
        """Dispatch conditioning on self.cond_mode.

        cond_inputs must contain:
          - 'image' / 'interp' mode: obs_value_grid [B, F, Ny, Nx],
                                     obs_mask_grid  [B, F, Ny, Nx]
          - 'pointnet' mode:         obs_coords_2d [B, M, 2], obs_values [B, M, 1],
                                     obs_mask [B, M], obs_field_ids [B, M]

        Returns (cond_feat, cond_mask) matching LatentFMUNet's channel slots.
        """
        if self.cond_mode == "image_norm":
            # Fidelity repair 1: masked MEAN per latent cell (not the average of a
            # zero-filled grid, which attenuates the signal by ~100-600x at 0.1-1%
            # sensor density) and a saturating density count/(1+count) in place of
            # the binary occupancy mask; channel count is unchanged.
            gv = cond_inputs["obs_value_grid"]
            gm = cond_inputs["obs_mask_grid"]
            factor = 2 ** self.n_levels
            pool = F.avg_pool3d if self.spatial_dim == 3 else F.avg_pool2d

            v = self._pad_to_ae(gv * gm)
            m = self._pad_to_ae(gm)
            num = pool(v, kernel_size=factor, stride=factor)
            den = pool(m, kernel_size=factor, stride=factor)
            cond_feat = torch.where(den > 0, num / den.clamp(min=1e-8),
                                    torch.zeros_like(num))

            cnt = den * float(factor ** self.spatial_dim)
            m_ds = cnt / (1.0 + cnt)
            cond_mask = torch.cat([m_ds, m_ds.amax(dim=1, keepdim=True)], dim=1)
            return cond_feat, cond_mask

        if self.cond_mode in ("image", "interp"):
            obs_value_grid = cond_inputs["obs_value_grid"]
            obs_mask_grid = cond_inputs["obs_mask_grid"]
            factor = 2 ** self.n_levels
            if self.cond_mode == "image":
                # masked sparse field (zero at unobserved cells)
                cond_grid = obs_value_grid * obs_mask_grid
            else:
                # nearest-neighbor Voronoi fill: dense interpolated field
                cond_grid = BASELINE_HELPERS.nearest_fill_grid(obs_value_grid, obs_mask_grid)
            cond_grid = self._pad_to_ae(cond_grid)
            pool = F.avg_pool3d if self.spatial_dim == 3 else F.avg_pool2d
            cond_feat = pool(cond_grid, kernel_size=factor, stride=factor)
            # Mask channels always carry the *actual* sensor positions so the
            # network can distinguish real measurements from interp guesses.
            cond_mask = self._downsample_mask(obs_mask_grid)
            return cond_feat, cond_mask

        # pointnet
        h, w = self._latent_hw()
        return self.pointnet(
            cond_inputs["obs_coords_2d"],
            cond_inputs["obs_values"],
            cond_inputs["obs_mask"],
            cond_inputs["obs_field_ids"],
            h, w,
        )

    def training_loss(self, fields_grid: torch.Tensor, cond_inputs: dict):
        """Rectified-flow MSE loss with configurable sparse conditioning."""
        with torch.no_grad():
            x1 = self.ae.encode(fields_grid)
            if self._latent_scale_active():
                x1 = x1 * self.latent_scale
        cond_feat, cond_mask = self._encode_condition(cond_inputs)

        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
        t_shape = (B,) + (1,) * (x1.ndim - 1)
        t_ = t.view(t_shape)
        x_t = (1 - t_) * x0 + t_ * x1
        target = x1 - x0

        pred = self.velocity_net(t, x_t, cond_feat, cond_mask)
        loss = F.mse_loss(pred, target)
        info = {"loss": float(loss.detach().cpu())}
        return loss, info

    @torch.no_grad()
    def sample(self, cond_inputs: dict,
               n_steps: int = 8, ode_solver: str = "euler") -> torch.Tensor:
        """ODE integration from t=0 (noise) to t=1 (data) in latent space,
        then AE-decode back to the field grid. Supports 'euler' and 'heun'."""
        if ode_solver not in ("euler", "heun"):
            raise ValueError(
                f"Unsupported ode_solver={ode_solver!r}. Use 'euler' or 'heun'."
            )
        cond_feat, cond_mask = self._encode_condition(cond_inputs)
        B = cond_feat.shape[0]
        shape = (B, self.latent_ch, *cond_feat.shape[2:])
        x = torch.randn(shape, device=cond_feat.device, dtype=cond_feat.dtype)

        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((B,), k * dt, device=x.device, dtype=x.dtype)
            v = self.velocity_net(t, x, cond_feat, cond_mask)
            if ode_solver == "euler":
                x = x + dt * v
            else:  # heun: 2nd-order predictor-corrector
                t_next = torch.full((B,), (k + 1) * dt,
                                    device=x.device, dtype=x.dtype)
                x_pred = x + dt * v
                v_next = self.velocity_net(t_next, x_pred, cond_feat, cond_mask)
                x = x + 0.5 * dt * (v + v_next)

        if self._latent_scale_active():
            x = x / self.latent_scale
        out = self.ae.decode(x)
        return out
