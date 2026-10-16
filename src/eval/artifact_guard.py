"""Write-time identity guard for evaluation artifacts.

Four times on 2026-09-10 a knob that changed WHAT was measured -- sensor
density, measurement operator, ensemble size -- was missing from WHERE the
result was written, and one run silently replaced another's JSON (DELTA_STATUS
sec.20, 21, 23). Each was fixed at its own naming site. This module is the fix
that does not depend on every name being right.

Every payload already records its own identity: protocol stamp, model, sensor
count, K, NFE, conditioning source and channels. Before a JSON is written over
an existing file, the identity stored INSIDE the existing file is compared with
the new payload's:

  same identity       -> overwrite (a re-run, or a resume refreshing a roll-up)
  different identity  -> the existing file is left untouched, the new payload
                         goes to a sidecar named after its own identity, and
                         the conflict is recorded so the caller exits non-zero

Neither artifact is ever destroyed. That matters because a collision is most
likely to be discovered at the END of a long job, where refusing to write at
all would throw the compute away.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# n_frames / ckpt / seed / split belong here as much as density does: a 3-frame
# smoke run aimed at a hardcoded canonical name agrees with the 50-frame
# canonical result on every other field, and would otherwise be allowed to
# replace it. Same bug, one level down, in the tool meant to end it.
# s3gm_sampler is stamped only by S3GM rows scored with the upstream sampler.
IDENTITY_KEYS = ("protocol", "model", "n_obs", "K", "nfe", "cond_source",
                 "cond_fields", "n_frames", "ckpt", "seed", "split", "s3gm_sampler")

# Artifacts written before a field existed carry no value for it. For this
# evaluator every artifact that predates the protocol stamp is a CLEAN run --
# measurement operators did not exist yet -- so an absent stamp in an EXISTING
# file is read as the clean protocol. That is what lets an operator run collide
# with, and be stopped by, a pre-stamp clean file instead of slipping past it.
LEGACY_DEFAULTS = {"protocol": "kolm2d_matched_v1"}

CONFLICTS: list[dict] = []


def _default_log(msg: str) -> None:
    print(msg, flush=True)        # flushed: SLURM logs are read while running


def _norm(key, value):
    if value is None:
        return None
    if key in ("n_obs", "cond_fields"):
        vals = value if isinstance(value, (list, tuple)) else [value]
        return tuple(sorted({int(v) for v in vals}))
    if key in ("K", "nfe", "n_frames", "seed"):
        return int(value)
    return value


def identity(payload: dict, legacy: bool = False) -> dict:
    """The fields that say what was measured. legacy=True fills defaults for
    fields an older artifact could not have carried."""
    out = {}
    for k in IDENTITY_KEYS:
        if k in payload:
            out[k] = _norm(k, payload[k])
        elif legacy and k in LEGACY_DEFAULTS:
            out[k] = LEGACY_DEFAULTS[k]
    return out


def _tag(ident: dict) -> str:
    blob = json.dumps(ident, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def safe_write_json(path, payload: dict, indent: int = 1, log=_default_log) -> Path:
    """Write payload to path unless that would replace a different measurement.
    Returns the path actually written (the sidecar on a conflict)."""
    path = Path(path)
    new_id = identity(payload)
    if path.exists() and path.stat().st_size > 0:
        try:
            old = json.loads(path.read_text())
        except (OSError, ValueError):
            old = None                      # unreadable: nothing to protect
        if isinstance(old, dict):
            old_id = identity(old, legacy=True)
            shared = [k for k in IDENTITY_KEYS if k in old_id and k in new_id]
            diff = {k: (old_id[k], new_id[k]) for k in shared
                    if old_id[k] != new_id[k]}
            if diff:
                side = path.with_name(
                    f"{path.stem}.CONFLICT_{_tag(new_id)}{path.suffix}")
                side.write_text(json.dumps(payload, indent=indent))
                CONFLICTS.append({
                    "intended": str(path), "written": str(side),
                    "differs": {k: [str(a), str(b)] for k, (a, b) in diff.items()},
                })
                what = ", ".join(f"{k}: existing={a} new={b}"
                                 for k, (a, b) in diff.items())
                log(f"[guard] REFUSED to overwrite {path.name} ({what}); "
                    f"existing file kept, new result written to {side.name}")
                return side
    path.write_text(json.dumps(payload, indent=indent))
    return path


def exit_if_conflicts(log=_default_log) -> None:
    """Call once at the end of a run: any refused overwrite makes the job fail
    loudly, so a collision cannot pass as a clean success."""
    if not CONFLICTS:
        return
    log(f"[guard] {len(CONFLICTS)} write(s) refused -- a different measurement "
        "already held each target name. Nothing was overwritten:")
    for c in CONFLICTS:
        log(f"  {Path(c['intended']).name} -> kept; new result in "
            f"{Path(c['written']).name}  differs on {c['differs']}")
    raise SystemExit(3)
