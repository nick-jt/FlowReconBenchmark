"""Surface-pool conditioning for the point-native baselines on SHIFT-WING.

WHY THIS FILE EXISTS
--------------------
Every baseline adapter in ``model_baseline.py`` builds its conditioning with
``helpers_baseline.build_sparse_condition``, which samples sensors from
RANDOM VOLUME POINTS of the field being reconstructed.  On SHIFT-WING that is
not the inverse problem: the task is surface-to-volume reconstruction, and our
model (``train_pointcloud_ffm.py``) only ever sees

  * the 65,536-point body-surface observation pool, 4 observable channels
    (Cp, tau_x, tau_y, tau_z  ->  field ids 3, 4, 5, 6), and
  * two exact scalar parameter tokens (Mach, alpha -> field ids 7, 8).

A baseline conditioned on interior samples would be handed information our
model never receives; it would be solving a strictly easier problem and the
comparison would invert.  This module gives the baselines the SAME draw.

WHAT IS REUSED VERBATIM
-----------------------
``build_wing_condition`` calls ``helpers.build_sparse_condition_from_pool``
and ``helpers.append_extra_tokens`` -- the very two functions our model's
trainer calls at ``train_pointcloud_ffm.py:580`` and ``:599`` -- with the same
per-channel budgets.  Nothing is reimplemented, so there is no second
implementation to drift.

Deliberate difference from our model's training loop, recorded here because it
is the one thing that is NOT identical: our run additionally pushes the pool
draw through ``apply_measurement_operators`` (sensor noise sigma in [0, 0.1],
20% ball occlusion, 30% channel dropout) before appending the parameter
tokens.  That is a property of OUR method (amortising the posterior over
measurement operators), not of the benchmark protocol, and the baseline
machinery has no equivalent.  The baselines therefore train on strictly
CLEANER conditioning than ours.  If that biases anything it biases in the
baselines' favour, so it cannot invert the comparison; it is stated in every
wing baseline config.

GATING
------
Everything here is inert unless the config sets

    shared:
      conditioning:
        source: surface_pool

so no other dataset's behaviour changes.  See ``surface_pool_enabled``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from methods.dmfgen.dataset_shiftwing import N_OBS_FIELD_TYPES, PARAM_FIELD_IDS, SURF_VALUE_FIELD_IDS
# The canonical helpers, NOT helpers_baseline: these are the functions our
# model's own trainer uses, so the baseline draw is the same routine.
from methods.dmfgen.helpers import append_extra_tokens, build_sparse_condition_from_pool

__all__ = [
    "surface_pool_enabled",
    "wing_cond_channels",
    "build_wing_condition",
    "wing_fill_nodes",
    "N_OBS_FIELD_TYPES",
    "PARAM_FIELD_IDS",
    "SURF_VALUE_FIELD_IDS",
]


def surface_pool_enabled(cfg: dict) -> bool:
    """True iff this config asks for surface-pool conditioning.

    The single gate for every additive branch added to ``model_baseline.py``.
    """
    try:
        source = cfg["shared"]["conditioning"].get("source")
    except (KeyError, TypeError, AttributeError):
        return False
    return str(source or "").lower() == "surface_pool"


def wing_cond_channels(dataset) -> int:
    """Number of distinct sensor FIELD TYPES the conditioning must represent.

    Not ``dataset.num_fields``.  The volume has 4 generated channels, but a
    wing sensor can carry any of 9 field ids (0..3 volumetric, 4..6 wall
    shear, 7..8 parameter tokens).  Any embedding or per-field channel stack
    indexed by ``obs_field_ids`` must be this wide or it indexes out of range.
    """
    return int(getattr(dataset, "n_obs_field_types", N_OBS_FIELD_TYPES))


def build_wing_condition(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    n_obs_min: Sequence[int],
    n_obs_max: Sequence[int],
):
    """Surface-pool observation tuple for one training batch.

    Mirrors ``train_pointcloud_ffm.py`` exactly: draw m_c ~ U{min_c, max_c}
    locations independently per pool channel, then append the two exact
    parameter tokens last (always valid, indices 0, never used for clamping).

    Returns the standard padded 5-tuple
        (obs_coords [B, M, 3], obs_values [B, M, 1], obs_mask [B, M],
         obs_indices [B, M] (POOL indices -- see the warning below),
         obs_field_ids [B, M]).

    WARNING on ``obs_indices``: for the volume draw these index the QUERY
    point set, which is what ``helpers_baseline.scatter_sensors_to_nodes``
    relies on.  Here they index the SURFACE POOL, a different point set
    entirely, so scattering sensors onto mesh nodes by index is meaningless
    on this dataset.  Node-token models must use the geometric fill
    (``wing_fill_nodes``) instead.
    """
    oc, ov, om, oi, ofid = build_sparse_condition_from_pool(
        pool_coords=batch["obs_pool_coords"].to(device, non_blocking=True),
        pool_values=batch["obs_pool_values"].to(device, non_blocking=True),
        pool_field_ids=batch["obs_pool_field_ids"].to(device, non_blocking=True),
        n_obs_min=list(n_obs_min),
        n_obs_max=list(n_obs_max),
    )
    return append_extra_tokens(
        oc, ov, om, oi, ofid,
        extra_coords=batch["param_coords"].to(device, non_blocking=True),
        extra_values=batch["param_values"].to(device, non_blocking=True),
        extra_field_ids=batch["param_field_ids"].to(device, non_blocking=True),
    )


def wing_fill_nodes(
    node_coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    n_cond_ch: int = N_OBS_FIELD_TYPES,
    sigma: float = 0.05,
    chunk: int = 8192,
    global_field_ids: Sequence[int] = tuple(PARAM_FIELD_IDS),
):
    """Node-token conditioning for the surface-pool problem.

    Same contract as ``helpers_baseline.nearest_sensor_fill_nodes`` -- each
    node gets, per field id, its nearest valid sensor's value plus a soft
    support weight exp(-d^2 / 2 sigma^2) -- with two wing-specific changes,
    kept in a separate function so the shared volume path is untouched:

      1. the channel stack is ``n_cond_ch`` wide (9 field types), not
         ``n_fields`` (4).  Indexing a 4-wide stack with field id 6 is an
         out-of-bounds write; this is why the shared helper cannot be reused
         as-is here.

      2. the parameter tokens (``global_field_ids``, default Mach/alpha) are
         GLOBAL, not local.  They sit at the single coordinate (0.5, 0.5, 0.5)
         and are known exactly everywhere, so they are broadcast to every node
         with support 1.0.  Treating them as ordinary point sensors would pass
         them through the same exp(-d^2/2 sigma^2) decay as a pressure tap and
         zero them out for all but the handful of nodes near the domain
         centre -- i.e. it would silently throw the flow conditions away.

    Returns (value [B, N, n_cond_ch], support [B, N, n_cond_ch]).
    """
    B, N, _ = node_coords.shape
    device, dtype = node_coords.device, node_coords.dtype
    n_cond_ch = int(n_cond_ch)
    value = torch.zeros(B, N, n_cond_ch, device=device, dtype=dtype)
    support = torch.zeros(B, N, n_cond_ch, device=device, dtype=dtype)
    inv2s2 = 1.0 / (2.0 * float(sigma) ** 2)
    globals_ = set(int(f) for f in global_field_ids)

    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        oc = obs_coords[b, valid]
        ov = obs_values[b, valid, 0]
        of = obs_field_ids[b, valid].long()
        for f in of.unique().tolist():
            f = int(f)
            if not (0 <= f < n_cond_ch):
                raise IndexError(
                    f"sensor field id {f} does not fit a {n_cond_ch}-channel "
                    "conditioning stack; pass n_cond_ch=n_obs_field_types.")
            sel = of == f
            if f in globals_:
                # Exactly-known global scalar: same value at every node.
                value[b, :, f] = ov[sel][0]
                support[b, :, f] = 1.0
                continue
            oc_f = oc[sel]
            ov_f = ov[sel]
            for s in range(0, N, chunk):
                d = torch.cdist(node_coords[b, s:s + chunk], oc_f)
                dmin, imin = d.min(dim=1)
                value[b, s:s + chunk, f] = ov_f[imin]
                support[b, s:s + chunk, f] = torch.exp(-dmin.pow(2) * inv2s2)
    return value, support


def describe_condition(obs_mask: torch.Tensor, obs_field_ids: torch.Tensor) -> str:
    """One-line sensor census, printed once per run as a protocol fingerprint."""
    valid = obs_mask.bool()
    ids = obs_field_ids[valid].long()
    counts = {int(f): int((ids == f).sum()) for f in ids.unique().tolist()}
    n_param = sum(counts.get(int(f), 0) for f in PARAM_FIELD_IDS)
    return (f"sensors={int(valid.sum())} (surface={int(valid.sum()) - n_param}, "
            f"param_tokens={n_param}) per_field_id={counts}")
