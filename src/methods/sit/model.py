# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════════════════════
# SiT (Scalable Interpolant Transformers) — Model Components          

# ─────────────────────────────────────────────────────────────────────────────
# [SiT] adaLN-Zero modulation helper
# ─────────────────────────────────────────────────────────────────────────────
def _sit_modulate(x, shift, scale):
    """[SiT] adaLN-Zero: y = x * (1 + scale) + shift (scalar broadcast)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Sinusoidal timestep embedding
# ─────────────────────────────────────────────────────────────────────────────
class SiTTimestepEmbedder(nn.Module):
    """[SiT] Embeds scalar diffusion/flow timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Core transformer block (adaLN-Zero conditioned)
# ─────────────────────────────────────────────────────────────────────────────
class SiTBlock(nn.Module):
    """[SiT] A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        from timm.models.vision_transformer import Attention, Mlp
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # qk_norm=True adds per-head LayerNorm on Q and K to bound attention
        # logit magnitude. Without it, Q·Kᵀ norms drift upward over tens of
        # epochs until softmax saturates to one-hot and gradients explode —
        # the signature failure mode of plain ViT/DiT attention on long runs.
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                              qk_norm=True, norm_layer=nn.LayerNorm,
                              **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(_sit_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_sit_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Final output head for patch tokenizer (unpatchify-ready)
# ─────────────────────────────────────────────────────────────────────────────
class SiTFinalLayer(nn.Module):
    """[SiT] The final layer of SiT (patch tokenizer path)."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _sit_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Point-token tokenizer (native irregular-mesh path)
# ─────────────────────────────────────────────────────────────────────────────
class SiTPointTokenEmbedder(nn.Module):
    """[SiT] Per-node input embedder for irregular meshes.

    Replaces PatchEmbed when `tokenizer='pointnet'`. Each mesh node becomes
    one transformer token: its features are the concatenation of
      - the noisy field values at that node        (in_channels)
      - optional sparse-sensor conditioning        (cond_channels)
      - Fourier-feature positional encoding of (x,y) coords
    followed by a small MLP to hidden_size. The DiT blocks never see raw
    coords or a grid — only hidden-size tokens.
    """

    def __init__(self, in_channels: int, cond_channels: int, coord_dim: int,
                 hidden_size: int, fourier_num_freqs: int = 64,
                 fourier_scale: float = 10.0):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.coord_dim = coord_dim
        self.fourier_num_freqs = fourier_num_freqs
        # Fixed random Fourier basis (non-trainable) — standard NeRF-style
        # positional encoding. Gaussian(0, scale²) frequencies.
        B = torch.randn(coord_dim, fourier_num_freqs) * fourier_scale
        self.register_buffer("fourier_B", B, persistent=True)

        token_in = in_channels + cond_channels + 2 * fourier_num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(token_in, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x_nodes, coords, cond_nodes=None):
        # x_nodes: [B, N, in_channels]
        # coords:  [B, N, coord_dim]
        # cond_nodes: [B, N, cond_channels] or None
        proj = coords @ self.fourier_B                   # [B, N, F]
        pos = torch.cat([proj.sin(), proj.cos()], dim=-1)
        parts = [x_nodes]
        if self.cond_channels > 0:
            assert cond_nodes is not None, "cond_channels > 0 requires cond_nodes"
            parts.append(cond_nodes)
        parts.append(pos)
        return self.mlp(torch.cat(parts, dim=-1))


class SiTPointFinalLayer(nn.Module):
    """[SiT] Per-token output head (analog of SiTFinalLayer for point tokens)."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _sit_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] 2D / 1D sin-cos positional embedding helpers (non-square grids)
# ─────────────────────────────────────────────────────────────────────────────
def _sit_get_2d_sincos_pos_embed_nonsquare(embed_dim, gh, gw):
    """[SiT] 2D sin-cos positional embedding for non-square grids (gh x gw patches)."""
    assert embed_dim % 2 == 0
    grid_h = np.arange(gh, dtype=np.float32)
    grid_w = np.arange(gw, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # w goes first in meshgrid
    emb_h = _sit_get_1d_sincos_pos_embed(embed_dim // 2, grid_h.flatten())
    emb_w = _sit_get_1d_sincos_pos_embed(embed_dim // 2, grid_w.flatten())
    return np.concatenate([emb_h, emb_w], axis=1)  # (gh*gw, embed_dim)


def _sit_get_1d_sincos_pos_embed(embed_dim, pos):
    """[SiT] 1D sin-cos positional embedding."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Main model: SiT adapted for physical fields (no class labels, no VAE)
# ─────────────────────────────────────────────────────────────────────────────
class SiTPhysics(nn.Module):
    """
    [SiT] SiT adapted for physical field reconstruction from sparse observations.

    Core architecture (patch tokenization, adaLN-Zero DiT blocks, sinusoidal
    timestep embedding, flow-matching transport framework) is unchanged from
    the original SiT. The only difference is a widened patch-embed input:
    the noisy field tensor is channel-concatenated with the sparse-field-
    as-image conditioning (masked sparse field + per-field/any-sensor mask)
    before patchification, mirroring the LatentFM 'image' conditioner. The
    output head still produces only the `in_channels` field channels.

    Changes from the original SiT:
      - No class label embedding (y_embedder removed)
      - in_channels / out_channels set to n_fields (not latent VAE channels)
      - learn_sigma disabled
      - Supports non-square grids via separate H/W patch counts
      - Optional `cond_channels` extra channels concatenated at the patch
        embed for sparse-sensor conditioning (default 0 = unconditional).
    """

    def __init__(
        self,
        # Grid-mode params (used only when tokenizer='patch')
        input_size_h: int = 0,
        input_size_w: int = 0,
        patch_size: int = 2,
        # Common
        in_channels: int = 5,
        cond_channels: int = 0,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        # Tokenizer
        tokenizer: str = "patch",
        coord_dim: int = 2,
        fourier_num_freqs: int = 64,
        fourier_scale: float = 10.0,
    ):
        super().__init__()

        if tokenizer not in ("patch", "pointnet"):
            raise ValueError(f"tokenizer must be 'patch' or 'pointnet', got {tokenizer!r}")

        self.tokenizer = tokenizer
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.cond_channels = cond_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w

        if tokenizer == "patch":
            # Patch embedding with room for optional sparse conditioning channels.
            # The output head still produces only `in_channels` channels; only the
            # patch embedding sees the widened input.
            from timm.models.vision_transformer import PatchEmbed
            self.x_embedder = PatchEmbed(
                img_size=(input_size_h, input_size_w),
                patch_size=patch_size,
                in_chans=in_channels + cond_channels,
                embed_dim=hidden_size,
                bias=True,
            )
            num_patches = (input_size_h // patch_size) * (input_size_w // patch_size)
            # Fixed sin-cos positional embedding for non-square grids.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        else:  # pointnet
            self.x_embedder = SiTPointTokenEmbedder(
                in_channels=in_channels,
                cond_channels=cond_channels,
                coord_dim=coord_dim,
                hidden_size=hidden_size,
                fourier_num_freqs=fourier_num_freqs,
                fourier_scale=fourier_scale,
            )
            self.pos_embed = None  # Fourier features are inside the embedder.

        self.t_embedder = SiTTimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        if tokenizer == "patch":
            self.final_layer = SiTFinalLayer(hidden_size, patch_size, self.out_channels)
        else:
            self.final_layer = SiTPointFinalLayer(hidden_size, self.out_channels)

        self._initialize_weights(input_size_h, input_size_w, hidden_size, patch_size)

    def _initialize_weights(self, H, W, hidden_size, patch_size):
        """Initialize weights; uses 2D sin-cos pos embed for non-square grids."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if self.tokenizer == "patch":
            # 2D sin-cos positional embedding for non-square patch grids.
            gh = H // patch_size
            gw = W // patch_size
            pos_embed = _sit_get_2d_sincos_pos_embed_nonsquare(hidden_size, gh, gw)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # Patch embed like nn.Linear.
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Timestep embedding MLP.
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers.
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)  — supports non-square grids.
        """
        c = self.out_channels
        p = self.patch_size
        gh = self.input_size_h // p
        gw = self.input_size_w // p

        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, gh * p, gw * p)
        return imgs

    def forward(self, x, t,
                obs_value_grid=None, obs_mask_grid=None,
                coords=None, obs_value_nodes=None, obs_mask_nodes=None,
                **kwargs):
        """
        Forward pass of SiT for physical fields. Dispatches on self.tokenizer.

        tokenizer='patch' (grid DiT):
          x:              [B, C, H, W] noisy spatial fields
          obs_value_grid: [B, n_fields, H, W] sparse observed values
          obs_mask_grid:  [B, n_fields, H, W] per-field observation mask

        tokenizer='pointnet' (irregular-mesh DiT):
          x:               [B, N, C] noisy fields at mesh nodes
          coords:          [B, N, coord_dim] normalized mesh-node coordinates
          obs_value_nodes: [B, N, n_fields] sparse observed values scattered back
                           to mesh nodes (zero where no sensor)
          obs_mask_nodes:  [B, N, n_fields] per-field observation mask at nodes
        """
        c = self.t_embedder(t)

        if self.tokenizer == "patch":
            if self.cond_channels > 0:
                if obs_value_grid is None or obs_mask_grid is None:
                    cond_feat = torch.zeros(
                        x.shape[0], self.in_channels, x.shape[2], x.shape[3],
                        device=x.device, dtype=x.dtype,
                    )
                    cond_mask = torch.zeros(
                        x.shape[0], self.in_channels + 1, x.shape[2], x.shape[3],
                        device=x.device, dtype=x.dtype,
                    )
                else:
                    # Caller supplies obs_value_grid already in the form they
                    # want the network to see: masked sparse field (zeros at
                    # unobserved) for cond_mode='image', or Voronoi-filled
                    # dense field for cond_mode='interp'. The mask channels
                    # always carry actual sensor locations.
                    cond_feat = obs_value_grid
                    any_mask = obs_mask_grid.amax(dim=1, keepdim=True)
                    cond_mask = torch.cat([obs_mask_grid, any_mask], dim=1)
                x = torch.cat([x, cond_feat, cond_mask], dim=1)

            x = self.x_embedder(x) + self.pos_embed   # (N, T, D)
            for block in self.blocks:
                x = block(x, c)
            x = self.final_layer(x, c)                 # (N, T, p²·C)
            x = self.unpatchify(x)                     # (N, C, H, W)
            return x

        # pointnet tokenizer
        assert coords is not None, "pointnet tokenizer requires `coords` kwarg."
        cond_nodes = None
        if self.cond_channels > 0:
            if obs_value_nodes is None or obs_mask_nodes is None:
                B, N, _ = x.shape
                cond_feat = torch.zeros(B, N, self.in_channels,
                                        device=x.device, dtype=x.dtype)
                cond_mask_full = torch.zeros(B, N, self.in_channels + 1,
                                             device=x.device, dtype=x.dtype)
            else:
                # Caller supplies obs_value_nodes already shaped for the chosen
                # cond_mode (masked vs. Voronoi-filled).
                cond_feat = obs_value_nodes
                any_mask = obs_mask_nodes.amax(dim=-1, keepdim=True)
                cond_mask_full = torch.cat([obs_mask_nodes, any_mask], dim=-1)
            cond_nodes = torch.cat([cond_feat, cond_mask_full], dim=-1)

        tokens = self.x_embedder(x, coords[..., :self.x_embedder.coord_dim], cond_nodes)
        for block in self.blocks:
            tokens = block(tokens, c)
        out = self.final_layer(tokens, c)              # [B, N, C]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Lightweight EMA used by the SiT baseline
# ─────────────────────────────────────────────────────────────────────────────
class SiTEMA:
    """[SiT] Exponential Moving Average of model parameters (lightweight version)."""

    def __init__(self, parameters, decay=0.9999):
        self.decay = decay
        self.shadow = [p.clone().detach() for p in parameters]
        self.backup = []

    def update(self, parameters):
        for s, p in zip(self.shadow, parameters):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def store(self, parameters):
        self.backup = [p.clone().detach() for p in parameters]

    def restore(self, parameters):
        for p, b in zip(parameters, self.backup):
            p.data.copy_(b)
        self.backup = []

    def copy_to(self, parameters):
        for p, s in zip(parameters, self.shadow):
            p.data.copy_(s)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        # torch.load(..., map_location="cpu") brings shadow back as CPU
        # tensors; move them to the device of the shadow set at __init__
        # so EMA.update matches the live (GPU) parameters.
        if self.shadow:
            device = self.shadow[0].device
            self.shadow = [s.to(device) for s in state_dict["shadow"]]
        else:
            self.shadow = list(state_dict["shadow"])
