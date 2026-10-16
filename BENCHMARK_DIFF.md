# BENCHMARK_DIFF

How the code of every method in FlowReconBench relates to its upstream repository. Upstream checkouts are git submodules under `third_party/` (pinned to the commits the benchmark was run against); `diff -r third_party/<repo>/<file> src/methods/<family>/<file>` shows the exact text differences for the vendored ports. The comparisons below are at the definition level: a definition is *identical* when its AST matches upstream after removing docstrings/comments and the `th`→`torch` import alias.

Every method is evaluated by FlowReconBench's own protocol code (`src/eval`, `src/methods/dmfgen/ensemble_eval.py`),.

## Protocol rules shared by every method (`src/protocol.py`)

* **Selection and scoring splits.** The held-out block of every dataset is split by index parity: odd indices are the
  TUNE half, even indices the TEST half. Every trainer selects `best.pt` on TUNE (`split="tune"`); every reported 2D number
  is a 50-frame TEST mean (`fleet_2d --split test`, stamped `split: test` in the JSON); the JHTDB rows are scored on all
  50 held-out snapshots and split by parity in `analysis.paired_ci_jhu`, so they are 25-snapshot TEST means. Rows scored on the whole held-out
  block (protocol stamp `kolm2d_matched_v1`) are not canonical.
* **Fixed-seed selection masks.** The validation pass draws `vis_n_obs_list` sensors per observed field under a fixed seed
  (`fixed_validation_draw`, seed 0), restoring the RNG state afterwards, so `best.pt` is the optimum of one fixed measurement.
  The same rule applies to DMF-Gen and every baseline; `best_val` is persisted in the checkpoint and survives resumes.
* **One checkpoint rule.** A deterministic row reports the minimum-validation-loss checkpoint (`best.pt`). A generative
  row's validation loss is a flow-matching / score objective that need not track reconstruction quality, so it reports
  whichever of `best.pt` and `last.pt` reconstructs the TUNE frames better (`tune_2d`, `eval.tune_select`), together
  with the sampler settings chosen there; the evaluator stamps `ckpt` and `ckpt_epoch`.
* **No sensor overwrite.** DMF-Gen samples are no longer overwritten with the true sensor values after each ODE step
  (`clamp_hard=False` in every evaluator, stamped into every JSON); no method's prediction is modified at the sensors.
* **Parameter budget.** Learned rows are sized to 6.5 M +- 10 % parameters by `numel` with complex elements counted once
  (`python -m tools.count_params --config <yaml>`); every trainer prints the count at build time. A published
  configuration outside the budget is a separately labelled row.
* **Training sensor ranges** cover the evaluated densities: Kolmogorov 65-6554 per field (0.1-10 %), cylinder mesh 24-476
  (0.1-2 %), JHU 1953-19531 (0.1-1 %); identical for every method on a dataset. The cylinder is run on the native mesh
  (`Cylinder2D_mesh.h5`) by every method; grid-only methods (latent FM, S3GM, patch SiT, grid FNO) have no cylinder row.
* **Relative L2 on the cylinder mesh** is the unweighted mean over the 23,800 cell centres (the mesh export carries no cell
  areas); the convention is the same for every method.
* **Coverage** of the 5-95 % percentile band is reported alongside the min-max band, whose ideal for a calibrated K-member
  ensemble is (K-1)/(K+1) = 0.78 at K = 8; the percentile-band ideals are below the nominal 0.50 / 0.90 at finite K.

## Upstream checkouts (git submodules)

| submodule | repository | commit |
|---|---|---|
| `third_party/S3GM` | https://github.com/lzy12301/S3GM | `2343293bf627` |
| `third_party/SiT` | https://github.com/willisma/SiT | `cbde832a40b1` |
| `third_party/CoNFiLD` | https://github.com/jx-wang-s-group/CoNFiLD | `449835e76a2b` |
| `third_party/Gen4Turbulence` | https://github.com/vivekoommen/Gen4Turbulence | `8af0a7b94353` |
| `third_party/Senseiver` | https://github.com/OrchardLANL/Senseiver | `e443eb0ca188` |
| `third_party/deepxde` | https://github.com/lululxvi/deepxde | `99b6620386d1` |
| `third_party/deeponet` | https://github.com/lululxvi/deeponet | `8d62345afd39` |
| `third_party/PhyCoFlow-MultiFieldReconstruction` | https://github.com/cosmos2w/PhyCoFlow-MultiFieldReconstruction | `1b29b55e82c0` |

Not submodules: Geo-FNO uses the `neuraloperator` pip package (pinned `neuraloperator==2.0.0`, `from neuralop.models import FNO`); latent flow matching and MLP-RBF are FlowReconBench implementations with no single upstream (a latent-diffusion-family stand-in after Rombach et al. 2022 / CoNFiLD, and a point-native deterministic control); the classical anchors (nearest-neighbour, IDW, gappy POD, LSE, train mean) are written from the cited papers.

## S3GM (`src/methods/s3gm/`)

Upstream `lzy12301/S3GM` @ `2343293`. The video score model files are the **upstream files verbatim** (`nn.py`, `fp16_util.py`, `rpe.py`, `unet_video.py`, `ema.py` = `third_party/S3GM/models/*.py`; `sde.py` = `third_party/S3GM/sampler/sde.py`), with exactly one change, in `nn.checkpoint`:

| file | vs `third_party/S3GM` |
|---|---|
| `s3gm/nn.py` | +6 / -2 lines (`models/nn.py`) |
| `s3gm/fp16_util.py` | byte-identical (`models/fp16_util.py`) |
| `s3gm/rpe.py` | byte-identical (`models/rpe.py`) |
| `s3gm/unet_video.py` | byte-identical (`models/unet_video.py`) |
| `s3gm/ema.py` | byte-identical (`models/ema.py`) |
| `s3gm/sde.py` | +2 / -2 lines (`sampler/sde.py`): `VESDE.discretize` moves the sigma table to the index device before both lookups (upstream indexes the CPU table with a CUDA index, rejected by current PyTorch); values unchanged |

* `nn.checkpoint`: gradient checkpointing goes through `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
  instead of upstream's custom `CheckpointFunction` (same recomputation, non-reentrant autograd; the benchmark rows were
  trained with this, `use_checkpoint: true` in the 2D configs). Everything else in these files, including upstream's
  `VPSDE` and `RPE.forward_safe_qk`, is unchanged; the benchmark instantiates `VESDE(sigma_min, sigma_max, N)` explicitly
  from the config in every call site.
* `adapter.py` (training loop, DPS sampling, visualisation, the 2D adapter) and `s3gm3d.py` (JHU 125^3 adaptation: the
  volume's z axis is mapped onto upstream's *frame* axis and reconstructed with upstream's own overlapping-window
  `generate_parallel_2d` mechanism; `JHU_AUGMENT=octahedral` rotates the slab axis over training) are FlowReconBench code.
  `norm_guidance.py` is the labelled *normalized-guidance* improvement arm (fitted on the tuning split), reported separately
  from the upstream-faithful row; `tuning/` is that campaign.
* 2D (Kolmogorov) recipe: the JHU architecture (nf 32, ch_mult [1,2,3,4], 6.35 M) is the reported row, the published
  300.8 M size a labelled variant; `optimizer: upstream` = Adam 2e-4 at a constant learning rate, no weight decay, no
  clipping, no step skipping (the benchmark trainer's warm-up + cosine, clip 0.5 and grad-norm skipping remain selectable,
  used by no reported row); VE-SDE with sigma_max 474 from the Song-Ermon rule and the eps score parametrisation;
  sampled with the upstream predictor (`fleet_2d --s3gm-sampler upstream`, 1000 steps), guidance alpha_case chosen on TUNE.
* 3D (JHU) recipe: `s3gm3d.py` takes the same `diffusion.score_param` (raw | eps) switch as the 2D adapter; the reported JHU
  prior (`configs/jhu3d/config_baseline_S3GM_xcube.yaml`) uses sigma_max 20 and the raw score output; at sigma_max 2270,
  the Song-Ermon value on the z-scored 10-plane slabs, the raw-output prior diverges unconditionally
  (`configs/jhu3d/config_baseline_S3GM_xcube_sigmamax2270.yaml`). `eval_jhu.py` scores the TUNE or TEST half and takes overlap, outer steps and
  guidance from the command line so the arm is chosen on TUNE.

## SiT-point (`src/methods/sit/`)

Upstream `willisma/SiT` @ `cbde832`. `transport/` is upstream's `transport/` package (interpolant paths, loss, samplers); `model.py` re-implements the SiT blocks for physical fields (no class labels, no VAE, point tokenizer) with the `SiT` name prefix.

10 definitions: 0 identical to upstream (modulo comments/docstrings and the `th`→`torch` alias), 5 adapted, 5 FlowReconBench-only

| definition | vs upstream | upstream file |
|---|---|---|
| `_sit_modulate` | differs (2 changed lines of 2) | SiT/models.py |
| `SiTTimestepEmbedder` | differs (5 changed lines of 20) | SiT/models.py |
| `SiTBlock` | differs (3 changed lines of 18) | SiT/models.py |
| `SiTFinalLayer` | differs (2 changed lines of 13) | SiT/models.py |
| `SiTPointTokenEmbedder` | ours only |  |
| `SiTPointFinalLayer` | ours only |  |
| `_sit_get_2d_sincos_pos_embed_nonsquare` | ours only |  |
| `_sit_get_1d_sincos_pos_embed` | differs (7 changed lines of 8) | SiT/models.py |
| `SiTPhysics` | ours only |  |
| `SiTEMA` | ours only |  |

`transport/` vs upstream `SiT/transport/`:

| definition | vs upstream | upstream file |
|---|---|---|
| `__init__.py` | identical | SiT/transport/__init__.py |
| `path.py` | identical | SiT/transport/path.py |
| `utils.py` | identical | SiT/transport/utils.py |
| `integrators.py` | differs (+33 / -31 lines) | SiT/transport/integrators.py |
| `transport.py` | differs (+23 / -2 lines) | SiT/transport/transport.py |

Training recipe: the reported SiT-point rows use upstream's optimizer (`optimizer: upstream`: AdamW, lr 1e-4, no
weight decay, constant learning rate, no gradient clipping, no step skipping, MSE velocity loss, EMA 0.9999 with the raw
weights saved alongside; raw vs EMA is chosen on TUNE and stamped). The benchmark trainer's own recipe (AdamW betas
0.9/0.95, lr 5e-5, weight decay 1e-4, warm-up + cosine, clip 0.5, loss/gradient spike skipping, optional Huber loss) remains
selectable and is used by no reported row.

Deliberate differences:
* `SiTBlock`: attention uses `qk_norm=True` with `nn.LayerNorm` (upstream: no QK-norm); `timm`'s `Attention`/`Mlp` are imported inside the block.
* `transport.py`: an optional Huber training loss (`huber_beta`, off in every config -> upstream MSE) and case-insensitive sampler names.
* `integrators.py`: the ODE sampler is a fixed-step Euler/Heun loop without the `torchdiffeq` dependency (upstream calls `odeint`); the SDE sampler is upstream's.
* FlowReconBench-only: point tokenizer (`SiTPointTokenEmbedder`/`SiTPointFinalLayer`, the 3D configuration), non-square 2D sin-cos embeddings, `SiTPhysics` (the conditional model), `SiTEMA`, and the chunked conditional samplers in `adapter.py`.

## Senseiver (`src/methods/senseiver/`)

Upstream `OrchardLANL/Senseiver` @ `e443eb0` (`model.py`, `positional.py`, `network_light.py`). `model.py` is a
**re-implementation** (the upstream code is a Lightning module tied to its own data pipeline); with `upstream_layout: true`
it reproduces upstream's module graph exactly, and the source comments cite the upstream lines for each choice:
per-axis top positional frequency (`positional.py:18-22`), every cross-attention as `Residual(CrossAttention) + Residual(mlp)`
with a width-preserving MLP (`model.py:24-43`), tied encoder layers `layer_1` + one reused `layer_n` (`model.py:194-216`),
dropout on the residual branch (`model.py:74-83`), a bare `Linear` decoder read-out (`model.py:242,276`), and
`optimizer: upstream` = plain Adam at a constant learning rate (`network_light.py:78`). `upstream_layout: false` is the
pre-audit layout kept only so frozen pre-audit run directories still load.

Fixed-mask TUNE-split validation, first used by the Senseiver capacity sweep, is now the benchmark-wide selection rule
(`src/protocol.py`); the sweep's patience early stopping is not used. The local residual paths `Senseiver+local`
(`SEN_LOCAL_IDW=1`) and `Senseiver+xattn` (`SEN_LOCAL_XATTN=1`) in `local_arms.py` are labelled improvement arms, inert
unless enabled and never mixed into the upstream-faithful row.

## DeepONet / DeepONet++ (`src/methods/deeponet/`)

Reference implementations `lululxvi/deepxde` @ `99b6620` (`deepxde/nn/pytorch/deeponet.py`, `fnn.py`, `deeponet_strategy.py`,
`initializers.py`) and `lululxvi/deeponet` @ `8d62345` (`src/deeponet_pde.py`, the paper's hyper-parameters). `model.py` is a
standalone PyTorch re-implementation that reproduces upstream's conventions exactly (unstacked DeepONet with a trainable
output bias, linear branch output, activated trunk output, ReLU + Glorot-normal init, Adam 1e-3 constant, `split_branch`
multi-output); the one declared structural adaptation is the branch input: a permutation-invariant set encoder over the
variable sensor set, because the benchmark draws a different sensor set per snapshot while vanilla DeepONet reads `u` at a
fixed sensor grid. `modelpp.py` (DeepONet++, multiresolution binned pooling in the branch) is the labelled improvement arm.
`eval_jhu.py` is the canonical evaluator; `eval_jhu_pp.py` reuses it with an explicit `build_fn` argument (the original
checkout swapped the builder by assigning into the module).

## CoNFiLD (`src/methods/confild/`)

Upstream `jx-wang-s-group/CoNFiLD` @ `449835e` is **imported directly from the submodule** (`third_party/CoNFiLD`, or
`confild_root` in `configs/site.yaml`): the SIREN auto-decoder (`ConditionalNeuralField.cnf.nf_networks.SIRENAutodecoder_film`),
the diffusion prior (`UnconditionalDiffusionTraining_and_Generation.src.script_util.create_model/create_gaussian_diffusion`)
and the posterior-sampling measurement noise (`ConditionalDiffusionGeneration.src.guided_diffusion.measurements.get_noise`).
`upstream_core.py` / `upstream_training.py` wrap them for the benchmark's H5 data, latent windows and budget accounting
(upstream model and optimiser semantics, case-4 3D recipe); `eval_unified.py` is the canonical sensor-guided DPS evaluator
scored with the shared seeded-ensemble estimator; `eval_unified2.py` adds the single-snapshot arm (`--window1`: only the
target row of each window carries sensors). Three arms are reported (published prior, capacity-matched, 384-d).

## Gen4Turb (`src/methods/gen4turb/`)

Upstream `vivekoommen/Gen4Turbulence` @ `8af0a7b`: `eval_jhu.py` imports `utils.architecture.Unet` and
`utils.diffusion.ElucidatedDiffusion` from `third_party/Gen4Turbulence/3_flow_reconstruction/dm` and scores its checkpoints
on the benchmark's matched snapshots and sensors (central 120^3 crop). `train_jhu.py` is upstream's `dm/train_model.py`
training loop with the U_x/U_z observation mask; its docstring lists the differences (one symmetric min-max range for the
velocity components, octahedral augmentation, the cosine decay stepped every batch, seed 2027, no held-out validation).

## Latent flow matching, MLP-RBF, Geo-FNO, classical anchors

* `latent_fm/`: convolutional autoencoder (stage 1) + conditional rectified flow in latent space with image/interp/pointnet
  conditioning (stage 2). No upstream. The two fidelity repairs of the JHU row (`cond_mode: image_norm` masked-mean
  conditioning and `latent_scale_mode: per_channel` latent rescaling) are used in every regime (disclosed); the original
  code path remains selectable. Sized to the budget including the autoencoder; both stages count towards the step budget.
* `mlp_rbf/`: point-native deterministic control (softmax-RBF gather over sensors + pooled global summary). No upstream.
* `geofno/`: `FNOSupervisedGrid` / `FNOSupervisedIrregular` wrap `neuralop.models.FNO` (pip `neuraloperator==2.0.0`) reading
  rasterised sensors as value + mask channels.
  These two are plain FNOs on a regular grid (`geofno_variant: fno | irregular`); neither has a learned deformation.
  `model_deform.py` (`geofno_variant: deform`) is Geo-FNO proper, ported from the reference implementation
  (neuraloperator/Geo-FNO, `elasticity/elas_geofno_v2.py`: `IPHI`, `SpectralConv2d`, `FNO2d`) and run on the cylinder's
  native mesh: the first Fourier layer transforms the sensor point cloud, the last is evaluated at the mesh points, and
  nothing is interpolated onto a grid. Differences from the reference file:
  * sparse-to-full task: the input point cloud is the sensor set (value + mask channels per field, as in the grid variant),
    the output point cloud is the full mesh; the sensor sum is a per-field mean, since the sensor count varies per sample;
  * all transforms use one Fourier-series normalisation, and the point evaluation is the exact real trigonometric
    polynomial `irfft2` computes (the reference mirrors the half spectrum with a one-index offset along k1), so a point
    cloud on the latent grid reproduces the grid layers (`tests/test_geofno_mesh.py`);
  * the mesh is embedded in the central `[0.1, 0.9]^2` of the latent torus (`domain_padding`), since the wake domain is
    not periodic;
  * the deformation starts at the identity (zero-initialised last layer), takes its angle/radius features about the
    cylinder centre in isotropic physical units, and has no geometry code (one fixed mesh);
  * the reference recipe (`optimizer: upstream`): Adam 1e-3 operator / 1e-4 deformation, weight decay 1e-4, StepLR halving
    five times over the run, relative-L2 loss, no gradient clipping; modes (20, 20), width 40, 80x80 latent grid, 6.47 M
    (reference: 12, 32, 40x40). The Kolmogorov FNO uses the same recipe at width 54 (6.38 M).
* `classical/`: nearest-neighbour (KD-tree), inverse-distance weighting (k=8), gappy POD (train-split basis, least-squares
  coefficients), ridge gappy POD, linear stochastic estimation (ridge map from the readings to the field, refitted per
  sensor layout) and the train-mean constant, for the 2D regimes and JHU; non-periodic distances by default.

## DMF-Gen (`src/methods/dmfgen/`) vs the upstream fork

The model under test is the published DMF-Gen code from `cosmos2w/PhyCoFlow_demo` (`upstream/main` @ `f163ad7`, `0_demo_TurbulentCombustion/src/`), evaluated as published at the benchmark's parameter budget. File names follow the upstream fork so the two trees diff directly; the benchmark's additions (canonical seeded sensor draws, measurement operators, spectral objectives, 3D symmetry augmentation, FNO-3D backbone, SHIFT-WING dataset, ensemble evaluation) are the differences.

The Kolmogorov row uses the compact-query decoder (GL_rbf_CQ) of the upstream multi-field code,
`cosmos2w/PhyCoFlow-MultiFieldReconstruction` @ `1b29b55` (`third_party/PhyCoFlow-MultiFieldReconstruction`), trained by that
code from the case directory `src/methods/dmfgen/kolmogorov_shu_case/` on the shared 65-6,554 sensor range.
`src/eval/upstream_cq.py` samples it inside `fleet_2d` with the upstream observation-consistency step switched off
(`obs_consistency_mode: none`), so its prediction is not overwritten at the sensors.

| file | upstream lines | FlowReconBench lines | added | removed |
|---|---|---|---|---|
| `dmfgen/Model.py` (upstream `Model.py`) | 2322 | 2380 | +256 | -198 |
| `dmfgen/helpers.py` (upstream `helpers.py`) | 1060 | 1315 | +356 | -101 |
| `dmfgen/obs_consistency.py` (upstream `obs_consistency.py`) | 211 | 211 | +0 | -0 |
| `dmfgen/train.py` (upstream `train_pointcloud_ffm.py`) | 1942 | 1334 | +636 | -1244 |
| `dmfgen/evaluate.py` (upstream `evaluate_ffm.py`) | 1189 | 1304 | +655 | -540 |
| `dmfgen/{measurement_ops,spectral_loss,spectral_prior,spectral_utils,augment_symmetry,fno3d_backbone,dataset_shiftwing,ensemble_eval}.py` | — | new | | |

Not carried over (not part of any reported row): the in-repo port of the GL_rbf_CQ backbone (`phycoflow_pointcloud/`, `model_cq.py`, `cq_flash_patch.py` and its tests; the Kolmogorov row runs the upstream code instead), the fine-tuning / LoRA / coherence-evaluation tooling of the upstream fork (`train_finetune.py`, `evaluate_coherence.py`, `coherence_dist.py`, `lora_finetune.py`, `model_finetune.py`, `evaluate_full_dataset.py`), the combustion-era dataset viewers, and the definitions the dead-code analysis found unreferenced (`ConditionalPointFFM`, `reflect_points_augment`, four unused dataset classes and the duplicate DMF-Gen classes that lived inside the consolidated baseline module).
