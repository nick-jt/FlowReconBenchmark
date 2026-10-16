# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import math
import torch
import torch.nn as nn
from methods.senseiver import local_arms
from typing import Optional


# =============================================================================
# Senseiver (Santos et al., Nat Mach Intell 2023).
#
# Upstream reference: third_party/Senseiver (github.com/OrchardLANL/Senseiver @ e443eb0).
#
# `upstream_layout=True` reproduces upstream's module graph exactly:
#   * positional.py:18-22  -- per-axis top frequency taken from the grid shape.
#   * model.py:33-43       -- EVERY cross-attention is
#                             Sequential(Residual(CrossAttention), Residual(mlp)).
#   * model.py:24-30       -- mlp is width preserving: LayerNorm, Linear(d,d),
#                             GELU, Linear(d,d).
#   * model.py:194-197,215-216 -- the encoder ties weights: one `layer_1` plus a
#                             single `layer_n` reused for every remaining depth.
#   * model.py:74-83       -- dropout sits on the residual branch.
#   * model.py:242,276     -- the decoder read-out is a bare Linear.
#
# `upstream_layout=False` is the pre-audit layout and exists ONLY so that the
# frozen pre-audit run directories under Save_TrainedModel/.../baseline_senseiver
# stay loadable.  New runs must set it True.
# =============================================================================


class SenseiverFourierPositionalEncoding(nn.Module):
    """Sine-cosine frequency encoding for spatial coordinates.

    Upstream (positional.py:18-22) defaults `max_frequencies` to the *spatial
    shape of the grid*, so each axis gets ``linspace(1, N_axis/2, num_bands)``.
    On a 125^3 grid that is a top band of 62.5, not 32.  `max_freq` therefore
    accepts either a scalar (all axes share one ladder) or a per-axis sequence.
    """

    def __init__(self, coord_dim: int = 2, num_bands: int = 32, max_freq: float = 64.0):
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.num_bands = int(num_bands)
        self.out_dim = self.coord_dim * self.num_bands * 2
        if isinstance(max_freq, (int, float)):
            max_freqs = [float(max_freq)] * self.coord_dim
        else:
            max_freqs = [float(v) for v in max_freq]
            if len(max_freqs) != self.coord_dim:
                raise ValueError(
                    f"max_freq must be a scalar or have {self.coord_dim} entries, "
                    f"got {len(max_freqs)}."
                )
        self.max_freqs = max_freqs
        # [coord_dim, num_bands]; broadcasts against coords[..., :, None].
        freqs = torch.stack(
            [torch.linspace(1.0, mf / 2.0, self.num_bands) for mf in max_freqs], dim=0
        )
        # Non-persistent: `freqs` is fully determined by (coord_dim, num_bands,
        # max_freq), so it must not sit in the state dict -- the per-axis [D, B]
        # form here would otherwise collide with the scalar [B] form stored by
        # pre-audit checkpoints.
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords[..., : self.coord_dim] * 2.0 - 1.0
        x = coords.unsqueeze(-1) * self.freqs * math.pi
        enc = torch.cat([x.sin(), x.cos()], dim=-1)
        return enc.reshape(*coords.shape[:-1], self.out_dim)


def _senseiver_mlp(num_channels: int, ff_mult: int = 1) -> nn.Sequential:
    """Upstream model.py:24-30.  ff_mult=1 is the upstream (width-preserving) form."""
    hidden = int(round(num_channels * ff_mult))
    return nn.Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, hidden),
        nn.GELU(),
        nn.Linear(hidden, num_channels),
    )


class SenseiverSelfAttentionBlock(nn.Module):
    """Upstream self_attention_layer (model.py:46-55)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        upstream_layout: bool = False,
    ):
        super().__init__()
        self.upstream_layout = bool(upstream_layout)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        if self.upstream_layout:
            # Residual-branch dropout, model.py:78,83.
            self.drop_attn = nn.Dropout(dropout)
            self.drop_mlp = nn.Dropout(dropout)
            self.mlp = _senseiver_mlp(dim, ff_mult)
        else:
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * ff_mult, dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        if self.upstream_layout:
            x = x + self.drop_attn(h)
            x = x + self.drop_mlp(self.mlp(x))
            return x
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SenseiverCrossAttentionLayer(nn.Module):
    """Upstream cross_attention_layer (model.py:33-43).

    Sequential(Residual(CrossAttention), Residual(mlp)) -- the trailing
    feed-forward is part of *every* upstream cross-attention, encoder and
    decoder alike.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        ff_mult: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.attn = nn.MultiheadAttention(
            q_dim, num_heads, kdim=kv_dim, vdim=kv_dim, dropout=dropout, batch_first=True
        )
        self.drop_attn = nn.Dropout(dropout)
        self.mlp = _senseiver_mlp(q_dim, ff_mult)
        self.drop_mlp = nn.Dropout(dropout)

    def forward(
        self,
        q_in: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = self.norm_kv(kv)
        h, _ = self.attn(self.norm_q(q_in), k, k, key_padding_mask=key_padding_mask)
        x = q_in + self.drop_attn(h)
        x = x + self.drop_mlp(self.mlp(x))
        return x


class SenseiverEncoderBlock(nn.Module):
    """Upstream Encoder.create_layer (model.py:176-192)."""

    def __init__(
        self,
        latent_dim: int,
        kv_dim: int,
        num_cross_heads: int,
        num_self_heads: int,
        num_self_attn_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.0,
        upstream_layout: bool = False,
    ):
        super().__init__()
        self.upstream_layout = bool(upstream_layout)
        if self.upstream_layout:
            self.cross_layer = SenseiverCrossAttentionLayer(
                q_dim=latent_dim,
                kv_dim=kv_dim,
                num_heads=num_cross_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
        else:
            self.norm_q = nn.LayerNorm(latent_dim)
            self.norm_kv = nn.LayerNorm(kv_dim)
            self.cross_attn = nn.MultiheadAttention(
                latent_dim,
                num_cross_heads,
                kdim=kv_dim,
                vdim=kv_dim,
                dropout=dropout,
                batch_first=True,
            )
        self.self_attn_layers = nn.ModuleList(
            [
                SenseiverSelfAttentionBlock(
                    latent_dim, num_self_heads, ff_mult, dropout, upstream_layout
                )
                for _ in range(num_self_attn_layers)
            ]
        )

    def forward(
        self,
        latent: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.upstream_layout:
            latent = self.cross_layer(latent, kv, key_padding_mask=key_padding_mask)
        else:
            q = self.norm_q(latent)
            k = self.norm_kv(kv)
            h, _ = self.cross_attn(q, k, k, key_padding_mask=key_padding_mask)
            latent = latent + h
        for layer in self.self_attn_layers:
            latent = layer(latent)
        return latent


class Senseiver(nn.Module):
    """Perceiver-IO sparse-sensor-to-full-field deterministic baseline."""

    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 2,
        num_latents: int = 128,
        latent_dim: int = 256,
        num_encoder_layers: int = 3,
        num_self_attn_per_block: int = 3,
        num_cross_attn_heads: int = 4,
        num_self_attn_heads: int = 4,
        dec_num_cross_attn_heads: int = 4,
        field_embed_dim: int = 32,
        space_bands: int = 32,
        max_freq: float = 64.0,
        ff_mult: int = 4,
        dropout: float = 0.0,
        upstream_layout: bool = False,
        enc_preproc_ch: Optional[int] = None,
        dec_latent_dim: Optional[int] = None,
        dec_preproc_ch: Optional[int] = None,
        share_encoder_layers: Optional[bool] = None,
        n_obs_field_types: Optional[int] = None,
    ):
        super().__init__()
        self.n_fields = int(n_fields)
        # Number of distinct SENSOR field ids, which is not necessarily the
        # number of generated fields. On SHIFT-WING a sensor may be one of 9
        # types (0-3 volumetric, 4-6 wall shear, 7-8 Mach/alpha parameter
        # tokens) while only 4 fields are reconstructed; embedding id 6 into a
        # 4-row table is an out-of-range index, so this must be widened or the
        # model cannot even represent the observations. Defaults to n_fields,
        # so every existing checkpoint keeps its exact embedding shape.
        self.n_obs_field_types = int(n_obs_field_types) if n_obs_field_types else int(n_fields)
        self.coord_dim = int(coord_dim)
        self.upstream_layout = bool(upstream_layout)
        self.num_encoder_layers = int(num_encoder_layers)
        if share_encoder_layers is None:
            share_encoder_layers = self.upstream_layout
        self.share_encoder_layers = bool(share_encoder_layers)

        self.pos_enc = SenseiverFourierPositionalEncoding(coord_dim, space_bands, max_freq)
        pos_dim = self.pos_enc.out_dim
        self.field_embed = nn.Embedding(self.n_obs_field_types, field_embed_dim)

        # Upstream keys/values are the `enc_preproc_ch`-wide projection of
        # (sensor value + positional encoding); s_parser.py:45 defaults it to 64
        # independently of the latent width.  Default here to latent_dim so the
        # pre-audit layout is unchanged.
        kv_dim = int(enc_preproc_ch) if enc_preproc_ch else int(latent_dim)
        self.kv_dim = kv_dim
        # + 1 sensor value, + field embedding (our per-channel-mask adaptation).
        self.encoder_preproc = nn.Linear(1 + field_embed_dim + pos_dim, kv_dim)
        self.latent = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        def _make_block() -> SenseiverEncoderBlock:
            return SenseiverEncoderBlock(
                latent_dim=latent_dim,
                kv_dim=kv_dim,
                num_cross_heads=num_cross_attn_heads,
                num_self_heads=num_self_attn_heads,
                num_self_attn_layers=num_self_attn_per_block,
                ff_mult=ff_mult,
                dropout=dropout,
                upstream_layout=self.upstream_layout,
            )

        if self.share_encoder_layers:
            # model.py:194-197 / 215-216: layer_1 then a single reused layer_n.
            self.encoder_blocks = nn.ModuleList(
                [_make_block()] + ([_make_block()] if self.num_encoder_layers > 1 else [])
            )
        else:
            self.encoder_blocks = nn.ModuleList(
                [_make_block() for _ in range(self.num_encoder_layers)]
            )

        dec_latent_dim = int(dec_latent_dim) if dec_latent_dim else int(latent_dim)
        self.dec_latent_dim = dec_latent_dim
        self.decoder_query_token = nn.Parameter(torch.randn(1, dec_latent_dim) * 0.02)

        if self.upstream_layout:
            # model.py:235-255.  With dec_preproc_ch unset (upstream default)
            # the query stream is simply [positional encoding | output token].
            q_dim = pos_dim + dec_latent_dim if not dec_preproc_ch else int(dec_preproc_ch)
            self.decoder_preproc = (
                nn.Linear(pos_dim + dec_latent_dim, int(dec_preproc_ch))
                if dec_preproc_ch
                else None
            )
            self.decoder_cross_layer = SenseiverCrossAttentionLayer(
                q_dim=q_dim,
                kv_dim=latent_dim,
                num_heads=dec_num_cross_attn_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            # model.py:242 -- bare Linear read-out, no trailing LayerNorm.
            self.decoder_postproc = nn.Linear(q_dim, n_fields)
        else:
            self.decoder_preproc = nn.Linear(pos_dim + latent_dim, latent_dim)
            self.decoder_norm_q = nn.LayerNorm(latent_dim)
            self.decoder_norm_kv = nn.LayerNorm(latent_dim)
            self.decoder_cross_attn = nn.MultiheadAttention(
                latent_dim,
                dec_num_cross_attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder_postproc = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, n_fields),
            )
        local_arms.attach(self)   # improvement arms; no-op unless SEN_LOCAL_* is set

    def forward(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        base = self._forward_base(query_coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        return local_arms.residual(self, base, query_coords, obs_coords, obs_values,
                                   obs_mask, obs_field_ids)

    def _forward_base(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_query, _ = query_coords.shape

        sensor_pos = self.pos_enc(obs_coords)
        sensor_field = self.field_embed(obs_field_ids.long().clamp(min=0))
        sensor_input = torch.cat([obs_values, sensor_field, sensor_pos], dim=-1)
        sensor_input = self.encoder_preproc(sensor_input)

        key_pad_mask = ~obs_mask.bool()
        latent = self.latent.unsqueeze(0).expand(batch_size, -1, -1)
        if self.share_encoder_layers:
            latent = self.encoder_blocks[0](latent, sensor_input, key_padding_mask=key_pad_mask)
            for _ in range(self.num_encoder_layers - 1):
                latent = self.encoder_blocks[1](
                    latent, sensor_input, key_padding_mask=key_pad_mask
                )
        else:
            for block in self.encoder_blocks:
                latent = block(latent, sensor_input, key_padding_mask=key_pad_mask)

        query_pos = self.pos_enc(query_coords)
        dq = self.decoder_query_token.expand(batch_size, n_query, -1)
        query_input = torch.cat([query_pos, dq], dim=-1)

        if self.upstream_layout:
            if self.decoder_preproc is not None:
                query_input = self.decoder_preproc(query_input)
            out = self.decoder_cross_layer(query_input, latent)
            return self.decoder_postproc(out)

        query_input = self.decoder_preproc(query_input)
        q = self.decoder_norm_q(query_input)
        kv = self.decoder_norm_kv(latent)
        h, _ = self.decoder_cross_attn(q, kv, kv)
        return self.decoder_postproc(query_input + h)
