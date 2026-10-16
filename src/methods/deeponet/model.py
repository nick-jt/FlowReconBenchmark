"""DeepONet baseline for the ICLR cross-cube JHU sparse-reconstruction arm.

Upstream
--------
L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis, "Learning nonlinear
operators via DeepONet based on the universal approximation theorem of
operators", Nature Machine Intelligence 3:218-229 (2021).

Reference implementations consulted (cloned, commits recorded in the run
metadata):
  * lululxvi/deepxde  @ 99b6620386d18cefb1549dddb7b7fe468cfad607
        deepxde/nn/pytorch/deeponet.py, deepxde/nn/pytorch/fnn.py,
        deepxde/nn/deeponet_strategy.py, deepxde/nn/initializers.py
  * lululxvi/deeponet @ 8d62345afd39e1df9c2c8c8d0e7c41882b06a9bf
        src/deeponet_pde.py (the paper's own hyper-parameters)

Upstream conventions reproduced here EXACTLY
--------------------------------------------
  * unstacked DeepONet (`stacked=False` in deeponet_pde.py:275): one branch
    net, one trunk net, combined by an inner product over a p-dimensional
    latent plus a trainable scalar bias `b` initialised to zero
    (deeponet.py:121-123, :292-294).
  * branch net output is LINEAR (FNN applies the activation to every layer
    except the last: fnn.py:56-67).
  * trunk net output is ACTIVATED -- `activation_trunk(trunk(x_loc))` in
    deeponet_strategy.py:35.  This asymmetry is upstream, not a typo.
  * activation `relu`, initialiser `Glorot normal` (= torch xavier_normal_,
    initializers.py:134) on weights, zeros on biases (fnn.py:49-50).
  * Adam, lr 1e-3, constant, no weight decay, no schedule, no gradient
    clipping (deeponet_pde.py:259, :168).
  * multi-output via upstream's `split_branch` strategy
    (deeponet_strategy.py, SplitBranchStrategy): ONE shared trunk of width p
    supplies a single learned basis, and the branch emits num_outputs x p
    coefficients.  This is upstream's own multi-output extension
    (Lu et al., CMAME 393:114778, 2022, Sec. 3.1.6).

The one structural adaptation: the branch input
-----------------------------------------------
Vanilla DeepONet's branch consumes u evaluated at a FIXED, shared set of m
sensors, so its input is a fixed-length vector in R^m.  This benchmark draws
random sensor locations AND random counts per snapshot, and evaluation uses a
per-snapshot seeded draw shared byte-for-byte with every other method, so a
fixed-sensor branch cannot even be evaluated on the canonical draw.

We therefore make the branch permutation-invariant over the observed set:

    branch(u) = rho( mean_s phi(x_s, v_s, e_{f_s}) )

where phi and rho are upstream FNNs and the mean runs over valid sensors.
Equivalently: the branch net is a single upstream FNN
    [3 + 1 + E] -> w -> w -> w -> w -> num_outputs*p
with a permutation-invariant mean-pool inserted after the third hidden layer.
Everything downstream -- inner product with an activated trunk, plus a scalar
bias -- is untouched, so the model is still a p-term expansion of the field in
a single learned trunk basis.  This is DeepSets/PointNet pooling, NOT
attention: no sensor ever attends to a query, which is what keeps this
recognisably DeepONet rather than Senseiver.

Bottleneck
----------
The branch output, num_outputs*p scalars, is the ONLY path from the sensors to
the prediction.  Per channel the field is a p-term expansion in a shared basis,
whose rank is capped by min(p, trunk_width) regardless of how large p is made.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _fnn_layers(sizes: List[int]) -> nn.ModuleList:
    """Upstream deepxde FNN layer stack: Linear + Glorot-normal W + zero b."""
    layers = nn.ModuleList()
    for i in range(1, len(sizes)):
        lin = nn.Linear(sizes[i - 1], sizes[i])
        nn.init.xavier_normal_(lin.weight)   # initializers.py:134 "Glorot normal"
        nn.init.zeros_(lin.bias)             # fnn.py:50 initializer_zero
        layers.append(lin)
    return layers


class DeepONetSetBranch(nn.Module):
    """DeepONet with a permutation-invariant branch over (coord, value, field).

    forward(query_coords [B,Q,3], obs_coords [B,M,3], obs_values [B,M,1],
            obs_mask [B,M], obs_field_ids [B,M]) -> [B,Q,n_fields]
    """

    def __init__(
        self,
        n_fields: int = 4,
        coord_dim: int = 3,
        p: int = 768,
        branch_width: int = 736,
        branch_phi_layers: int = 3,
        branch_rho_layers: int = 2,
        trunk_width: int = 960,
        trunk_layers: int = 4,
        field_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_fields = int(n_fields)
        self.coord_dim = int(coord_dim)
        self.p = int(p)
        self.field_embed_dim = int(field_embed_dim)

        # OUR ADAPTATION (see module docstring): each sensor carries one scalar
        # plus an embedding naming the field it measures, because this protocol
        # gives every channel an INDEPENDENT mask.  Upstream's branch sees all
        # channels at every sensor and needs no such tag.
        self.field_embed = nn.Embedding(self.n_fields, self.field_embed_dim)
        nn.init.normal_(self.field_embed.weight, mean=0.0, std=1.0)

        in_dim = self.coord_dim + 1 + self.field_embed_dim
        phi_sizes = [in_dim] + [branch_width] * branch_phi_layers
        self.branch_phi = _fnn_layers(phi_sizes)

        rho_sizes = [branch_width] * branch_rho_layers + [self.n_fields * self.p]
        self.branch_rho = _fnn_layers(rho_sizes)

        trunk_sizes = [self.coord_dim] + [trunk_width] * (trunk_layers - 1) + [self.p]
        self.trunk = _fnn_layers(trunk_sizes)

        # deeponet.py:121-123 -- one trainable scalar bias per output, init 0.
        self.b = nn.Parameter(torch.zeros(self.n_fields))

        self.act = nn.ReLU()

    # -- branch --------------------------------------------------------------
    def encode(self, obs_coords, obs_values, obs_mask, obs_field_ids):
        fid = obs_field_ids.clamp_min(0).long()
        emb = self.field_embed(fid)
        x = torch.cat([2.0 * obs_coords - 1.0, obs_values, emb], dim=-1)
        for lin in self.branch_phi:
            x = self.act(lin(x))
        m = obs_mask.unsqueeze(-1).to(x.dtype)
        x = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)   # permutation-invariant
        for lin in self.branch_rho[:-1]:
            x = self.act(lin(x))
        x = self.branch_rho[-1](x)                             # LINEAR output (fnn.py:66)
        return x.view(x.shape[0], self.n_fields, self.p)

    # -- trunk ---------------------------------------------------------------
    def trunk_forward(self, query_coords):
        x = 2.0 * query_coords - 1.0
        for lin in self.trunk[:-1]:
            x = self.act(lin(x))
        # deeponet_strategy.py:35 -- the trunk OUTPUT is activated.
        return self.act(self.trunk[-1](x))

    def combine(self, coeff, basis):
        # deeponet.py:292 merge_branch_trunk, batched over queries.
        return torch.einsum("bkp,bqp->bqk", coeff, basis) + self.b.view(1, 1, -1)

    def forward(self, query_coords, obs_coords, obs_values, obs_mask, obs_field_ids):
        coeff = self.encode(obs_coords, obs_values, obs_mask, obs_field_ids)
        basis = self.trunk_forward(query_coords)
        return self.combine(coeff, basis)

    # -- accounting ----------------------------------------------------------
    def bottleneck_scalars(self) -> int:
        return self.n_fields * self.p

    def n_params(self) -> int:
        return sum(q.numel() for q in self.parameters() if q.requires_grad)


def build_deeponet(arch: dict) -> DeepONetSetBranch:
    return DeepONetSetBranch(
        n_fields=int(arch.get("n_fields", 4)),
        coord_dim=int(arch.get("coord_dim", 3)),
        p=int(arch["p"]),
        branch_width=int(arch["branch_width"]),
        branch_phi_layers=int(arch.get("branch_phi_layers", 3)),
        branch_rho_layers=int(arch.get("branch_rho_layers", 2)),
        trunk_width=int(arch["trunk_width"]),
        trunk_layers=int(arch.get("trunk_layers", 4)),
        field_embed_dim=int(arch.get("field_embed_dim", 32)),
    )


if __name__ == "__main__":
    import itertools
    TARGET = 6_506_253
    m = DeepONetSetBranch()
    n = m.n_params()
    print(f"default config: params={n} ({100*(n-TARGET)/TARGET:+.3f}% of {TARGET})")
    print(f"  p={m.p} bottleneck={m.bottleneck_scalars()} "
          f"compression={7_812_500/m.bottleneck_scalars():.1f}x")
    for name, mod in [("branch_phi", m.branch_phi), ("branch_rho", m.branch_rho),
                      ("trunk", m.trunk)]:
        print(f"  {name}: {sum(q.numel() for q in mod.parameters())}")
    print(f"  field_embed: {m.field_embed.weight.numel()}  b: {m.b.numel()}")
