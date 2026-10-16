"""Senseiver improvement arms with a local residual path (PLAN_IMPROVE_2026-08-30
sec. 1b). Both are ENHANCED variants, not upstream Senseiver, and both are inert
unless enabled through the environment, in which case the model is exactly the
unpatched Senseiver at step 0 (gate parameters start at 0):

    y(q) = senseiver_decoder(q) + gate[c] * residual_c(q)

"Senseiver+local"  (SEN_LOCAL_IDW=1):  residual = k-NN inverse-distance interpolation
    of the same channel's sensors (mirrors classical/anchors_jhu.kd_predict mode='idw',
    non-periodic distances; torch.cdist + topk on the GPU, chunked over queries; the
    interpolant is computed under no_grad so gradients reach only the gate/decoder).
    SEN_IDW_K (8), SEN_IDW_CHUNK (2048), SEN_IDW_CDIST_MODE, SEN_IDW_EXACT_EPS (1e-3).
"Senseiver+xattn"  (SEN_LOCAL_XATTN=1): residual = single-head cross-attention from the
    query to its K nearest valid sensors (learned weights and read-out).
    SEN_XATTN_K (8), SEN_XATTN_DIM (64), SEN_XATTN_CHUNK (2048).
The two arms are mutually exclusive. Checkpoints trained with an arm require the same
environment at load time (the gate / attention parameters are part of the state dict).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOCAL = os.environ.get("SEN_LOCAL_IDW", "0") == "1"
_IDW_K = int(os.environ.get("SEN_IDW_K", "8"))
_IDW_CHUNK = int(os.environ.get("SEN_IDW_CHUNK", "2048"))
# cdist mode. fp32 matmul-expansion cdist loses ~5e-4 absolute accuracy near d=0
# (cancellation); the direct kernel is accurate to ~1e-7 but 23x slower in the training
# loop. Queries and sensors both sit on the 125^3 grid, so true distances are either
# exactly 0 or >= 1/124 ~ 8.06e-3: an exact-hit threshold of 1e-3 separates the two with
# >10x margin over the mm error. SEN_IDW_CDIST_MODE=donot_use_mm_for_euclid_dist restores
# the exact kernel.
_IDW_CDIST_MODE = os.environ.get("SEN_IDW_CDIST_MODE", "use_mm_for_euclid_dist")
_IDW_EXACT_EPS = float(os.environ.get("SEN_IDW_EXACT_EPS", "1e-3"))

_XATTN = os.environ.get("SEN_LOCAL_XATTN", "0") == "1"
_X_K = int(os.environ.get("SEN_XATTN_K", "8"))
_X_DIM = int(os.environ.get("SEN_XATTN_DIM", "64"))
_X_CHUNK = int(os.environ.get("SEN_XATTN_CHUNK", "2048"))

if _XATTN and _LOCAL:
    raise SystemExit("[senxattn] SEN_LOCAL_XATTN and SEN_LOCAL_IDW are mutually exclusive arms")


@torch.no_grad()
def _idw_interp(query_coords, obs_coords, obs_values, obs_mask, obs_field_ids,
                n_fields, k, chunk):
    """Per-observed-channel k-NN inverse-distance interpolation on GPU.

    Numerics mirror classical/anchors_jhu.kd_predict(mode='idw'):
    w = 1/max(d, 1e-12), exact hit (d < _IDW_EXACT_EPS, see mode note above)
    takes the nearest value, non-periodic Euclidean distances.  Unobserved channels stay 0 (== the
    train-split mean in z-score units, same convention as kd_predict).
    Returns [B, Q, n_fields].
    """
    B, Q, D = query_coords.shape
    device, dtype = query_coords.device, query_coords.dtype
    out = torch.zeros(B, Q, n_fields, device=device, dtype=dtype)
    valid = obs_mask.bool()
    fids = obs_field_ids.long()
    vals = obs_values[..., 0]

    for f in range(n_fields):
        m_f = (fids == f) & valid                       # [B, M]
        counts = m_f.sum(dim=1)                         # [B]
        if int(counts.min()) == 0:
            if int(counts.max()) == 0:
                continue  # channel unobserved everywhere: no residual
            # mixed presence cannot happen with fixed cond_fields; be safe
            raise RuntimeError(
                f"[senlocal] field {f} observed in only part of the batch")
        m_max = int(counts.max())
        # valid-first ordering per row -> padded per-field sensor tensors
        order = torch.argsort((~m_f).to(torch.uint8), dim=1, stable=True)
        sel = order[:, :m_max]                          # [B, m_max]
        coords_f = torch.gather(
            obs_coords, 1, sel.unsqueeze(-1).expand(-1, -1, D)).clone()
        vals_f = torch.gather(vals, 1, sel)
        pad = torch.arange(m_max, device=device).unsqueeze(0) >= counts.unsqueeze(1)
        # push padded sensors far outside the unit box so topk never picks
        # them (real neighbours exist: counts >= k is checked below)
        coords_f[pad] = 2.0e3
        kk = min(int(k), int(counts.min()))
        for s in range(0, Q, chunk):
            q = query_coords[:, s:s + chunk]            # [B, q, D]
            d = torch.cdist(q, coords_f, compute_mode=_IDW_CDIST_MODE)
            d_k, i_k = torch.topk(d, kk, dim=2, largest=False)
            v_k = torch.gather(
                vals_f.unsqueeze(1).expand(-1, d_k.shape[1], -1), 2, i_k)
            exact = d_k[..., 0] < _IDW_EXACT_EPS
            w = 1.0 / d_k.clamp_min(1e-12)
            est = (w * v_k).sum(-1) / w.sum(-1)
            est = torch.where(exact, v_k[..., 0], est)
            out[:, s:s + chunk, f] = est
    return out


class _LocalXAttn(nn.Module):
    """Single-head cross-attention from a query point to its K neighbours."""

    def __init__(self, pos_dim: int, field_embed_dim: int, n_fields: int,
                 dim: int):
        super().__init__()
        kv_in = 1 + field_embed_dim + pos_dim  # value | field emb | rel pos
        self.q_proj = nn.Linear(pos_dim, dim)
        self.k_proj = nn.Linear(kv_in, dim)
        self.v_proj = nn.Linear(kv_in, dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                  nn.GELU(), nn.Linear(dim, n_fields))
        self.scale = dim ** -0.5

    def forward(self, q_pos, nb_feat):
        # q_pos [B,q,P]; nb_feat [B,q,K,kv_in]
        q = self.q_proj(q_pos).unsqueeze(2)          # [B,q,1,D]
        k = self.k_proj(nb_feat)                     # [B,q,K,D]
        v = self.v_proj(nb_feat)
        w = F.softmax((q * k).sum(-1) * self.scale, dim=-1)  # [B,q,K]
        h = (w.unsqueeze(-1) * v).sum(2)             # [B,q,D]
        return self.head(h)                          # [B,q,n_fields]


def attach(model: nn.Module) -> None:
    """Called at the end of Senseiver.__init__: adds the arm's parameters if enabled."""
    if _LOCAL:
        # plain scalar (not sigmoid): init 0 makes the model exactly the
        # unpatched Senseiver at step 0, and the learned value is directly
        # the blend weight.  +n_fields parameters.
        model.local_gate = nn.Parameter(torch.zeros(model.n_fields))
        model.local_idw_k = _IDW_K
        print(f"[senlocal] IDW residual path ACTIVE: k={_IDW_K} "
              f"gate_params={model.n_fields} chunk={_IDW_CHUNK}", flush=True)
    if _XATTN:
        model.local_xattn = _LocalXAttn(model.pos_enc.out_dim,
                                        model.field_embed.embedding_dim,
                                        model.n_fields, _X_DIM)
        model.local_gate = nn.Parameter(torch.zeros(model.n_fields))
        model.local_xattn_k = _X_K
        n = sum(p.numel() for p in model.local_xattn.parameters())
        print(f"[senxattn] local cross-attn ACTIVE: K={_X_K} dim={_X_DIM} "
              f"extra_params={n + model.n_fields}", flush=True)


def residual(model: nn.Module, base, query_coords, obs_coords, obs_values, obs_mask,
             obs_field_ids):
    """Called by Senseiver.forward on its output; identity when no arm is attached."""
    gate = getattr(model, "local_gate", None)
    if gate is None:
        return base
    xa = getattr(model, "local_xattn", None)
    if xa is None:
        idw = _idw_interp(query_coords, obs_coords, obs_values, obs_mask,
                          obs_field_ids, model.n_fields, model.local_idw_k,
                          _IDW_CHUNK)
        return base + gate.view(1, 1, -1) * idw
    B, Q, D = query_coords.shape
    K = model.local_xattn_k
    res = torch.empty_like(base)
    # push invalid (padded) sensors far outside the unit box: never in top-K
    # as long as every batch item has >= K valid sensors (min draw 1953).
    oc = torch.where(obs_mask.bool().unsqueeze(-1), obs_coords,
                     torch.full_like(obs_coords, 2.0e3))
    fid = obs_field_ids.long().clamp(min=0)
    for s in range(0, Q, _X_CHUNK):
        q = query_coords[:, s:s + _X_CHUNK]                     # [B,q,D]
        with torch.no_grad():
            d = torch.cdist(q, oc)                              # [B,q,M]
            _, i_k = torch.topk(d, K, dim=2, largest=False)     # [B,q,K]
        nb_c = torch.gather(
            obs_coords.unsqueeze(1).expand(-1, q.shape[1], -1, -1), 2,
            i_k.unsqueeze(-1).expand(-1, -1, -1, D))            # [B,q,K,D]
        nb_v = torch.gather(
            obs_values[..., 0].unsqueeze(1).expand(-1, q.shape[1], -1), 2,
            i_k).unsqueeze(-1)                                  # [B,q,K,1]
        nb_f = torch.gather(
            fid.unsqueeze(1).expand(-1, q.shape[1], -1), 2, i_k)
        rel = model.pos_enc(
            (q.unsqueeze(2) - nb_c).reshape(B, -1, D)
        ).reshape(B, q.shape[1], K, -1)
        nb_feat = torch.cat([nb_v, model.field_embed(nb_f), rel], dim=-1)
        res[:, s:s + _X_CHUNK] = xa(model.pos_enc(q), nb_feat)
    return base + model.local_gate.view(1, 1, -1) * res
