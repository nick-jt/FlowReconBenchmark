"""Sample a run of the upstream PhyCoFlow-MultiFieldReconstruction trainer inside fleet_2d.

The upstream GL_rbf_CQ backbone (compact-query decoder) is trained by its own code on
its own copy of the Kolmogorov file (case kolmogorov_shu: the same Shu stride-4
trajectories, chronological 80/10/10 frames and training statistics). fleet_2d keeps
drawing the sensors and scoring on the FRB dataset; this module only maps each FRB
frame and point to the upstream sample, checks that both hold the same physical field,
and returns K posterior members.

Point order: the upstream loader sorts the grid points by coordinate
(`point_permutation`); FRB keeps the file order. Upstream position j holds file
point perm[j], so a file point p sits at upstream position inv[p].

Sensor consistency: the upstream sampler defaults to obs_consistency_mode
"endpoint_smooth" with a final clamp, which pulls each step toward a Gaussian
interpolant of the readings and then writes the readings into the output. The
benchmark scores the learned sampler alone, as for every other method, so the
mode is "none" unless a probe asks otherwise.

Run layout: <repo>/cases/<case>/runs/<experiment>/<stamp>, with the upstream
package under <repo>/src.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import torch

# Physical vorticity of the two loaders must agree to float32 rounding of a
# z-score round trip (|omega| <~ 30, std 4.8).
FIELD_ATOL = 1e-4


class UpstreamRun:
    def __init__(self, run_dir: Path, ckpt: str, device: str, steps: int, weights: str = "configured",
                 obs_mode: str = "none"):
        run_dir = Path(run_dir).resolve()
        src = run_dir.parents[4] / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from phycoflow_reconstruction.data.factory import open_field_dataset
        from phycoflow_reconstruction.evaluation.checkpoint import load_evaluation_runtime
        from phycoflow_reconstruction.training.common import sensor_protocol_from_config

        rt = load_evaluation_runtime(run_dir, split="validation", checkpoint=ckpt,
                                     sensor_config=None, generation_steps=steps,
                                     device_name=device, include_temporal_derivative=False)
        self.model, self.steps, self.device = rt.model, rt.generation_steps, rt.device
        self.normalizer = rt.dataset.normalizer
        self.weights = weights            # "configured" (EMA) or "live"
        self.obs_mode = obs_mode          # upstream obs_consistency_mode
        step = int(torch.load(rt.checkpoint_path, map_location="cpu", weights_only=False)["global_step"])
        n_train = len(open_field_dataset(dict(rt.config["dataset"]), split="train"))
        self.epoch = step // math.ceil(n_train / int(rt.config["optimization"]["batch_size"]))
        test = open_field_dataset(dict(rt.config["dataset"]), split="test")
        # Held-out frames of FRB (the last 20 %) are the upstream validation + test splits.
        self.frames = {int(t): (ds, i) for ds in (rt.dataset, test)
                       for i, t in enumerate(ds.selection.frame_indices)}
        perm = rt.dataset.point_permutation
        self.inv = torch.empty_like(perm)
        self.inv[perm] = torch.arange(perm.numel())
        # The run's protocol only validates the batch: the manifest pairs replace its draw.
        self.protocol = sensor_protocol_from_config(rt.config)
        fields = rt.config["observations"]["fields"].values()
        self.train_range = [[int(v.get("count", v.get("count_min"))) for v in fields],
                            [int(v.get("count", v.get("count_max"))) for v in fields]]

    def batch(self, frame: int, truth_phys: torch.Tensor, points: torch.Tensor,
              field_ids: torch.Tensor, readings: torch.Tensor, K: int):
        """K copies of the upstream sample of absolute `frame`, conditioned on the file
        points `points` / channels `field_ids` with the physical sensor `readings` [M]
        (fleet_2d's, after any measurement operator). `truth_phys` [N, C] is FRB's
        physical field of that frame, checked against the upstream one."""
        from phycoflow_reconstruction.data.sensor_protocols import build_observation_batch

        ds, i = self.frames[int(frame)]
        s = ds[i]
        err = float((self.normalizer.decode(s.values)[self.inv] - truth_phys.cpu()).abs().max())
        if err > FIELD_ATOL:
            raise SystemExit(f"[upstream] frame {frame}: upstream and FRB fields differ by {err:.2e}")
        pairs = [(int(self.inv[p]), int(f)) for p, f in zip(points.tolist(), field_ids.tolist())]
        # A batch needs unique sample ids; the members only differ in this label.
        members = [dataclasses.replace(s, trajectory_id=f"{s.trajectory_id}#{k}") for k in range(K)]
        batch = build_observation_batch(
            members, self.protocol, query_points=None,
            manifest_indices={f"{m.trajectory_id}:{m.time_index}": pairs for m in members})
        # The manifest copies the clean field values; replace them with the readings.
        z = (readings - self.normalizer.offset[field_ids]) / self.normalizer.scale[field_ids]
        batch.obs_values[:, :, 0] = z
        return batch.to(self.device)

    def sample(self, batch, seed: int) -> torch.Tensor:
        """Members [K, N, C] in physical units and FRB point order; one batched call,
        noise from a generator seeded with `seed`."""
        from phycoflow_reconstruction.training.model_lifecycle import selected_evaluation_weight_context

        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.no_grad(), selected_evaluation_weight_context(self.model, self.weights):
            pred = self.model.reconstruct(batch, steps=self.steps, generator=gen,
                                          obs_consistency_mode=self.obs_mode).prediction
        return self.normalizer.decode(pred)[:, self.inv.to(pred.device)]
