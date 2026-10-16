"""Append-only loss tracking with a figure that refreshes at every checkpoint.

Every baseline trainer in this project reported progress as scalars in a log
file and nothing else. That is how four CoNFiLD arms ran ~25 GPU-hours with a
plausible-looking loss while the decoder had collapsed to a latent-independent
mean field, and how a SiT sampler producing sheared striping still returned a
mid-table relative L2. A curve on disk that updates as the run proceeds is the
cheapest defence against both.

Usage:

    tracker = LossTracker(out_dir, name="stage1")
    ...
    tracker.log(step=step, train=float(loss))            # cheap, every step
    ...
    tracker.log(step=step, val=rel, ratio=dep)           # at checkpoints
    tracker.plot()                                       # refresh the figure

The CSV is the source of truth and is appended to, so a resumed or crashed run
keeps its history and the figure can always be rebuilt offline.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List


class LossTracker:
    def __init__(self, out_dir, name: str = "loss", plot_every: int | None = None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.csv_path = self.dir / f"{name}_history.csv"
        self.png_path = self.dir / f"{name}_history.png"
        self.plot_every = plot_every
        self._rows: List[Dict[str, float]] = []
        self._keys: List[str] = []
        self._n_since_plot = 0
        if self.csv_path.exists():                 # resume keeps prior history
            try:
                with open(self.csv_path, newline="") as fh:
                    for r in csv.DictReader(fh):
                        self._rows.append({k: float(v) for k, v in r.items()
                                           if v not in ("", None)})
                for r in self._rows:
                    for k in r:
                        if k not in self._keys:
                            self._keys.append(k)
            except Exception:
                self._rows, self._keys = [], []

    def log(self, step: int, **metrics) -> None:
        row = {"step": float(step)}
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                continue
        for k in row:
            if k not in self._keys:
                self._keys.append(k)
        self._rows.append(row)
        self._append_csv(row)
        self._n_since_plot += 1
        if self.plot_every and self._n_since_plot >= self.plot_every:
            self.plot()

    def _append_csv(self, row: Dict[str, float]) -> None:
        # Rewrite the header when a new metric appears mid-run, so the file
        # stays readable rather than silently losing columns.
        write_header = not self.csv_path.exists()
        existing = None
        if not write_header:
            with open(self.csv_path, newline="") as fh:
                existing = next(csv.reader(fh), None)
            if existing != self._keys:
                with open(self.csv_path, newline="") as fh:
                    old = list(csv.DictReader(fh))
                with open(self.csv_path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=self._keys)
                    w.writeheader()
                    for r in old:
                        w.writerow({k: r.get(k, "") for k in self._keys})
                write_header = False
        with open(self.csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self._keys)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self._keys})

    def plot(self) -> None:
        if not self._rows:
            return
        try:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            return
        series = [k for k in self._keys if k != "step"]
        if not series:
            return
        # One panel per metric: losses and diagnostics live on different scales,
        # and overlaying them is how a collapsing diagnostic gets hidden under a
        # descending loss.
        n = len(series)
        fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 3.6), squeeze=False)
        for j, key in enumerate(series):
            xs = [r["step"] for r in self._rows if key in r]
            ys = [r[key] for r in self._rows if key in r]
            if not xs:
                continue
            ax = axs[0, j]
            ax.plot(xs, ys, lw=1.0, color="#1f77b4", alpha=0.55)
            if len(ys) >= 20:                       # running median cuts noise
                w = max(3, len(ys) // 40)
                sm = np.convolve(ys, np.ones(w) / w, mode="valid")
                ax.plot(xs[w - 1:], sm, lw=1.9, color="#d62728",
                        label=f"mean of {w}")
                ax.legend(fontsize=7)
            if min(ys) > 0 and max(ys) / max(min(ys), 1e-30) > 50:
                ax.set_yscale("log")
            ax.set_title(f"{key}   (last {ys[-1]:.4g})", fontsize=10)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        fig.suptitle(self.name, fontsize=11)
        fig.tight_layout()
        fig.savefig(self.png_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        self._n_since_plot = 0
