# Extracted verbatim from PhyCoFlow_demo/0_demo_TurbulentCombustion/src/model_baseline.py
# (definitions are byte-identical; only this import header is generated).
from __future__ import annotations

import abc
import contextlib
import csv
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Tuple


class TrainingHistoryLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.csv_path = run_dir / "loss_history.csv"
        self.json_path = run_dir / "loss_history.json"
        self.plot_path = run_dir / "loss_history.png"
        self.rows: list[dict[str, Any]] = []
        self._cumul_train_time_s: float = 0.0
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                "epoch_time_s", "peak_gpu_mem_mb", "cumul_train_time_s",
            ])

    def append(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        epoch_time_s: Optional[float] = None,
        peak_gpu_mem_mb: Optional[float] = None,
    ) -> None:
        """Append one epoch record.

        epoch_time_s and peak_gpu_mem_mb should be measured *before* calling
        this method so that file-I/O latency is not counted in the timings.
        cumul_train_time_s is accumulated internally from epoch_time_s values.
        """
        if epoch_time_s is not None:
            self._cumul_train_time_s += epoch_time_s
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "epoch_time_s": None if epoch_time_s is None else round(float(epoch_time_s), 3),
            "peak_gpu_mem_mb": None if peak_gpu_mem_mb is None else round(float(peak_gpu_mem_mb), 1),
            "cumul_train_time_s": round(self._cumul_train_time_s, 3) if epoch_time_s is not None else None,
        }
        self.rows.append(row)
        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                row["epoch"],
                row["train_loss"],
                "" if row["val_loss"] is None else row["val_loss"],
                "" if row["epoch_time_s"] is None else row["epoch_time_s"],
                "" if row["peak_gpu_mem_mb"] is None else row["peak_gpu_mem_mb"],
                "" if row["cumul_train_time_s"] is None else row["cumul_train_time_s"],
            ])
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)
        self._save_plot()

    def _save_plot(self) -> None:
        epochs = [row["epoch"] for row in self.rows]
        train_points = [
            (row["epoch"], row["train_loss"])
            for row in self.rows
            if row["train_loss"] is not None and float(row["train_loss"]) > 0.0
        ]
        val_points = [
            (row["epoch"], row["val_loss"])
            for row in self.rows
            if row["val_loss"] is not None and float(row["val_loss"]) > 0.0
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        if train_points:
            ax.plot(
                [epoch for epoch, _ in train_points],
                [loss for _, loss in train_points],
                label="Train Loss",
                marker="o",
                color="blue",
                markersize=4,
            )
        if val_points:
            ax.plot(
                [epoch for epoch, _ in val_points],
                [loss for _, loss in val_points],
                label="Validation Loss",
                marker="s",
                color="orange",
                markersize=5,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss History")
        if train_points or val_points:
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        if epochs:
            ax.set_xlim(left=1, right=max(epochs))
        if train_points or val_points:
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)


@dataclass
class BaselineBundle:
    baseline_model: str
    training_stage: int
    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Optional[Any]
    ema: Optional[Any]
    device: torch.device
    run_dir: Path
    config: dict
    dataset_train: Optional[Any] = None
    dataset_val: Optional[Any] = None
    components: dict[str, Any] = field(default_factory=dict)


class LatentEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> dict:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        for name, value in state_dict.items():
            if name in self.shadow:
                self.shadow[name].copy_(value)


class BaseBaselineAdapter(abc.ABC):
    name: str
    uses_custom_training_loop: bool = False

    @abstractmethod
    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> Tuple[float, float, float]:
        raise NotImplementedError

    @abstractmethod
    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        raise NotImplementedError

    @abstractmethod
    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None, save_obs_consistency_plots: bool = False) -> dict[str, float]:
        raise NotImplementedError

    @contextlib.contextmanager
    def evaluation_weights(self, bundle: BaselineBundle) -> Iterator[None]:
        yield

    def run_custom_training(
        self,
        cfg: dict,
        device: torch.device,
        run_dir: Path,
        reload_ckpt: Optional[dict],
    ) -> dict:
        raise NotImplementedError(f"{self.name} does not define a custom training loop.")


def _checkpoint_model_state(checkpoint: dict, model_name: str) -> dict:
    if "model" not in checkpoint:
        raise KeyError(f"{model_name} checkpoint is missing required 'model' state_dict.")
    model_state = checkpoint["model"]
    if not isinstance(model_state, dict):
        raise TypeError(
            f"{model_name} checkpoint['model'] must be a state_dict, got {type(model_state)!r}."
        )
    model_state = dict(model_state)
    model_state.pop("_metadata", None)
    return model_state
