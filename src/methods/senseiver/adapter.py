# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from methods.common.helpers_baseline import build_sparse_condition
from methods.common.helpers_wing_baseline import build_wing_condition
from methods.common.helpers_wing_baseline import describe_condition as describe_wing_condition
from methods.common.helpers_wing_baseline import surface_pool_enabled
from methods.common.helpers_wing_baseline import wing_cond_channels
from methods.common.adapters import BaseBaselineAdapter, BaselineBundle, _checkpoint_model_state
from methods.common.config import _require_no_surface_pool, resolve_stage_config
from methods.common.deterministic import visualize_reconstruction_deterministic
from methods.senseiver.model import Senseiver

def run_epoch_senseiver(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
    model = bundle.model
    optimizer = bundle.optimizer if training else None
    shared_cond = bundle.config["shared"]["conditioning"]
    stage_cfg = resolve_stage_config(bundle.config)
    n_query_points = int(stage_cfg["training"].get("n_query_points", 4096))
    # Default 1.0 preserves the pre-audit behaviour for every existing config;
    # the ICLR config decides this explicitly.
    _gc = stage_cfg["training"].get("grad_clip", 1.0)
    grad_clip = None if _gc is None else float(_gc)
    # Measure the pre-clip global gradient norm so the paper can state that no
    # clipping was applied AND report the observed range.  max_norm=inf makes
    # clip_grad_norm_ a pure measurement.
    grad_norm_log_every = int(stage_cfg["training"].get("grad_norm_log_every", 10))
    _gnorms: list[float] = []

    _surface_pool = surface_pool_enabled(bundle.config)

    model.train(training)
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"Senseiver Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    for batch in pbar:
        coords = batch["coords"].to(bundle.device)
        fields = batch["fields"].to(bundle.device)
        batch_size, n_pts, _ = coords.shape
        valid_mask = batch.get("valid_sensor_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(bundle.device)

        if _surface_pool:
            # SHIFT-WING: sensors come from the body-surface pool plus the two
            # exact parameter tokens, never from interior volume points.
            obs_coords, obs_values, obs_mask, _, obs_field_ids = build_wing_condition(
                batch, bundle.device,
                n_obs_min=shared_cond["n_obs_min_list"],
                n_obs_max=shared_cond["n_obs_max_list"],
            )
        else:
            obs_coords, obs_values, obs_mask, _, obs_field_ids = build_sparse_condition(
                coords_full=coords,
                fields_full=fields,
                cond_fields=shared_cond["cond_fields"],
                n_obs_min=shared_cond["n_obs_min_list"],
                n_obs_max=shared_cond["n_obs_max_list"],
                valid_mask=valid_mask,
            )
        if training and epoch == 1 and count == 0:
            print(f"[cond] senseiver source="
                  f"{'surface_pool' if _surface_pool else 'volume'} "
                  f"{describe_wing_condition(obs_mask[:1], obs_field_ids[:1])}",
                  flush=True)

        if 0 < n_query_points < n_pts:
            idx = torch.randperm(n_pts, device=bundle.device)[:n_query_points]
            query_coords = coords[:, idx]
            query_fields = fields[:, idx]
        else:
            query_coords = coords
            query_fields = fields

        pred = model(query_coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        # Upstream network_light.py:68 uses reduction='sum'.  Adam is exactly
        # invariant to a constant rescaling of the loss, so 'mean' here is
        # equivalent -- and it is what makes upstream's lr=1e-4 transfer
        # unchanged.  The reduction is NOT the deviation; the clip below is.
        loss = F.mse_loss(pred, query_fields)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Upstream has NO gradient clipping.  Measured on this config
            # (diag_senseiver_gradnorm.py, 80 steps): the pre-clip global norm
            # has median 3.51, min 1.27, max 28.6 -- so max_norm=1.0 binds on
            # 100% of steps, rescaling by a step-varying 0.035..0.787.  Adam is
            # invariant to a CONSTANT rescale but not a step-varying one, so
            # this is normalised-gradient Adam, a different optimiser from the
            # published one.  `grad_clip: null` restores upstream exactly.
            if grad_norm_log_every > 0 and (count // max(batch_size, 1)) % grad_norm_log_every == 0:
                _gnorms.append(float(nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf"))))
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss * batch_size
        count += batch_size
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    if training and _gnorms:
        import numpy as _np
        _a = _np.asarray(_gnorms)
        _clipped = float((_a > grad_clip).mean()) if grad_clip is not None else 0.0
        print(
            f"[gradnorm] epoch={epoch:04d} n={_a.size} grad_clip={grad_clip}"
            f" min={_a.min():.4f} median={_np.median(_a):.4f}"
            f" p95={_np.percentile(_a, 95):.4f} max={_a.max():.4f}"
            f" clipped_frac={_clipped:.4f}",
            flush=True,
        )
        try:
            _gp = bundle.run_dir / "grad_norm_history.json"
            _hist = json.loads(_gp.read_text()) if _gp.exists() else []
            _hist.append({
                "epoch": int(epoch), "n": int(_a.size),
                "grad_clip": grad_clip,
                "min": float(_a.min()), "median": float(_np.median(_a)),
                "p95": float(_np.percentile(_a, 95)), "max": float(_a.max()),
                "clipped_frac": _clipped,
            })
            _gp.write_text(json.dumps(_hist, indent=2))
        except Exception as _exc:
            print(f"[gradnorm] could not write history: {_exc}", flush=True)

    return total_loss / max(count, 1)


class SenseiverAdapter(BaseBaselineAdapter):
    name = "senseiver"

    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        stage_cfg = resolve_stage_config(cfg)
        arch = stage_cfg["architecture"]
        training = stage_cfg["training"]

        coord_dim = int(arch["coord_dim"])
        # Upstream positional.py:18-22 takes the top frequency PER AXIS from the
        # grid shape (125 -> a 62.5 top band), rather than a single scalar.
        # `max_freq: null` in the YAML selects that upstream behaviour.
        if arch.get("max_freq") is None:
            data_cfg = cfg["shared"]["data"]
            axes = [data_cfg.get("num_x"), data_cfg.get("num_y"), data_cfg.get("num_z")]
            axes = [int(a) for a in axes[:coord_dim] if a]
            if len(axes) != coord_dim:
                raise ValueError(
                    "max_freq: null needs num_x/num_y/num_z for every coordinate axis."
                )
            max_freq = axes
        elif isinstance(arch["max_freq"], (list, tuple)):
            max_freq = [float(v) for v in arch["max_freq"]]
        else:
            max_freq = float(arch["max_freq"])

        model = Senseiver(
            n_fields=train_set.num_fields,
            coord_dim=coord_dim,
            num_latents=int(arch["num_latents"]),
            latent_dim=int(arch["latent_dim"]),
            num_encoder_layers=int(arch["num_encoder_layers"]),
            num_self_attn_per_block=int(arch["num_self_attn_per_block"]),
            num_cross_attn_heads=int(arch["num_cross_attn_heads"]),
            num_self_attn_heads=int(arch["num_self_attn_heads"]),
            dec_num_cross_attn_heads=int(arch["dec_num_cross_attn_heads"]),
            field_embed_dim=int(arch["field_embed_dim"]),
            space_bands=int(arch["space_bands"]),
            max_freq=max_freq,
            ff_mult=int(arch["ff_mult"]),
            dropout=float(arch["dropout"]),
            # Defaults below reproduce the pre-audit layout so that the frozen
            # pre-audit senseiver run dirs keep loading; the ICLR config sets
            # upstream_layout: true.
            upstream_layout=bool(arch.get("upstream_layout", False)),
            enc_preproc_ch=arch.get("enc_preproc_ch"),
            dec_latent_dim=arch.get("dec_latent_dim"),
            dec_preproc_ch=arch.get("dec_preproc_ch"),
            share_encoder_layers=arch.get("share_encoder_layers"),
            # None everywhere except SHIFT-WING, where the sensor field-id
            # vocabulary (9) is wider than the output field count (4).
            n_obs_field_types=(wing_cond_channels(train_set)
                               if surface_pool_enabled(cfg) else None),
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_bottleneck = int(arch["num_latents"]) * int(arch["latent_dim"])
        print(f"[senseiver] trainable_params={n_params}", flush=True)
        print(f"[senseiver] latent_bottleneck_scalars={n_bottleneck}", flush=True)
        print(f"[senseiver] upstream_layout={bool(arch.get('upstream_layout', False))} "
              f"max_freq={max_freq}", flush=True)

        # Upstream network_light.py:78 uses plain Adam at a constant LR with no
        # weight decay and no schedule.  `optimizer: upstream` restores that.
        if str(training.get("optimizer", "adamw")).lower() == "upstream":
            optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
            scheduler = None
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(training["epochs"]),
            )
        return BaselineBundle(
            baseline_model=self.name,
            training_stage=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=None,
            device=device,
            run_dir=run_dir,
            config=cfg,
            dataset_train=train_set,
            dataset_val=val_set,
        )

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        state = dict(_checkpoint_model_state(checkpoint, "Senseiver"))
        # Derived, non-persistent buffer; pre-audit checkpoints stored the
        # scalar [num_bands] form.  Drop it and keep the config-derived value.
        state.pop("pos_enc.freqs", None)
        bundle.model.load_state_dict(state)
        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.scheduler is not None and checkpoint.get("scheduler") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler"])

    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
        return run_epoch_senseiver(bundle, loader, training, epoch)

    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        stage_cfg = resolve_stage_config(bundle.config)
        arch = stage_cfg["architecture"]
        return {
            "baseline_model": bundle.baseline_model,
            "training_stage": bundle.training_stage,
            "model": bundle.model.state_dict(),
            "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
            "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "mean": bundle.dataset_train.mean,
            "std": bundle.dataset_train.std,
            "field_names": bundle.dataset_train.field_names,
            "method": "Senseiver",
            "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
            "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
            "n_fields": int(bundle.dataset_train.num_fields),
            "coord_dim": int(arch["coord_dim"]),
            "num_latents": int(arch["num_latents"]),
            "latent_dim": int(arch["latent_dim"]),
            "dataset": bundle.config["shared"]["data"]["dataset_name"],
        }

    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None, save_obs_consistency_plots: bool = False) -> dict[str, float]:
        _require_no_surface_pool(bundle.config, "senseiver")
        return visualize_reconstruction_deterministic(
            bundle=bundle,
            dataset=dataset,
            save_dir=save_dir,
            epoch=epoch,
            snapshot_index=snapshot_index,
            file_tag="senseiver",
            save_obs_consistency_plots=save_obs_consistency_plots,
        )
