# FlowReconBench

Code for *FlowReconBench: A benchmark for sparse-sensor flow-field reconstruction across
flow regimes, with probabilistic scoring* (N. Tricard, L. Wang, J. Chen, S. Deng).

The benchmark reconstructs full flow fields from sparse sensors in three regimes: a 2D
laminar cylinder wake on its native mesh (sensors in the volume or on the wall), 2D
Kolmogorov turbulence, and 3D isotropic turbulence from JHTDB. Every method shares the
data, splits, normalization, training sensor ranges, seeded sensor draws and metric code,
except Gen4Turb, which keeps its own min-max normalization and runs on the central 120^3 crop.
Generative methods are scored as K = 8 member ensembles (relative L2 of the mean, CRPS,
spread-error ratio, coverage).

## Layout

```
src/                     Python sources; run modules as `python -m <package>.<module>` with PYTHONPATH=src
  methods/               the methods, one package each:
    classical/           training mean, nearest neighbour, IDW, gappy POD, ridge gappy POD, LSE
    senseiver/ mlp_rbf/ geofno/ deeponet/       deterministic learners (FNO and Geo-FNO in geofno/)
    latent_fm/ confild/ s3gm/ gen4turb/ sit/   generative baselines
    dmfgen/              DMF-Gen; its trainer also trains FNO3D and FNO3D-FM (fno3d backbone)
    common/              shared configuration, data, adapter and registry code
    train_generative.py, train_deterministic.py, evaluate_*.py   shared trainers and evaluators
  eval/                  2D evaluator (fleet_2d), reported-row and TUNE selection, probes, result guard
  analysis/              paired tests, seed variance, cross-Re split, diagnostics, native reproductions
  figures/               field dumps; figures/pof/ writes the paper's figures and tables
  spectra/ calibration/ benchmarks/ tools/ data_prep/
  paths.py protocol.py launch.py
configs/                 run configurations by regime: cylinder2d, cylinder2d_surface, kolmogorov2d,
                         their _seed7 / _seed1337 replicates, jhu3d, upstream; site.yaml (cluster)
launchers/cases/         SLURM job cases: data, smoke, classical, 2d/{train,eval}, 3d/{train,eval},
                         figures, analysis
tests/                   pytest tests
third_party/             upstream repositories (git submodules, pinned)
data/, outputs/          datasets, runs and results (git-ignored)
BENCHMARK_DIFF.md        how each method's code relates to its upstream repository
```

Paths in configs and cases are repository-relative and resolve against the repository root
(`src/paths.py`), never against the working directory.

## Setup

```bash
git clone --recurse-submodules <this repository>
pip install -r requirements.txt          # exact versions: requirements-lock.txt
export PYTHONPATH=$PWD/src
```

Cluster settings (SLURM account, partition, GPU request, the line that activates the Python
environment, dataset and upstream locations) are in `configs/site.yaml`. `FRB_PROFILE=<name>`
selects a profile from that file and `FRB_<KEY>=...` overrides any key.

Sensor draws use `torch.randperm` on the GPU, so they depend on the GPU type. The JHTDB
evaluators check a fingerprint of the canonical draw and stop when it does not match; the
paper's draws were made on NVIDIA H100 SXM GPUs. The 2D evaluator records its fingerprint
in every result file.

## Data

| regime | file | source |
|---|---|---|
| cylinder wake | `data/cylinder2d/Cylinder2D_mesh.h5` | OpenFOAM 9 `pimpleFoam` runs at Re = 60, 80, 100, 150, 200, 250 (not included), converted by case `convert_cylinder` |
| Kolmogorov | `data/kolmogorov2d/Kolmogorov2D_shu_stride4.h5` | vorticity of Shu et al. (2023) as `data/kolmogorov_shu.npy`, converted by `convert_kolmogorov` |
| JHTDB | `data/JHU_4cubes_stride100.h5` | `download_jhtdb` (needs a JHTDB token in `JHTDB_TOKEN` and the `givernylocal` package) |
| JHTDB, 20 cubes (SM row) | `data/JHU_20cubes_plus_cube3.h5` | `download_jhtdb_scale` |

The converters also write the split manifests (`*_manifest.json`) next to each file.
`export_gen4turb_jhu` writes the JHTDB fields in the layout the upstream Gen4Turb loader reads.

## Running jobs

Every job is a case in `launchers/cases/`. The launcher adds the site header, exports
`FRB_ROOT` and `FRB_LOGS`, sets `PYTHONPATH` and runs the body from the repository root;
logs go to `outputs/logs/`.

```bash
python -m launch list [group]                      # cases, e.g. group 2d/eval
python -m launch show <case> [--variant V]         # the rendered SLURM script
python -m launch submit [sbatch options] <case> [--variant V] [-- args]
python -m launch run <case> [args]                 # drivers that submit other cases
```

Environment variables listed in a case's comments select models, settings or run directories.

## Reproducing the paper

Splits: every trainer selects its checkpoint on the TUNE half of the held-out frames (odd
indices); reported numbers are means over the TEST half (even indices). Sampler settings
(checkpoint, NFE, raw or EMA weights, guidance) are chosen on TUNE.

1. **Data**: `convert_cylinder`, `convert_kolmogorov`, `download_jhtdb`, `download_jhtdb_scale`,
   `export_gen4turb_jhu`.
2. **2D training**: `python -m launch run submit_2d_fleet` (add `SEEDS="7 1337"` for the seed
   replicates) and `python -m launch run submit_cyl_surface_fleet`. The Kolmogorov DMF-Gen row
   (`train_kolm_dmfgen_cq`) is trained by the upstream DMF-Gen code
   (`third_party/PhyCoFlow-MultiFieldReconstruction`) from the case directory
   `src/methods/dmfgen/kolmogorov_shu_case/`.
3. **2D evaluation**: `python -m launch run submit_2d_eval tune`, inspect the choices with
   `python -m eval.tune_select <dataset>`, then `python -m launch run submit_2d_eval test`, which
   submits the TEST rows, seed replicates, density sweep, operators, band metrics,
   CRPS-selected settings and the Kolmogorov DMF-Gen evaluation (`eval_kolm_dmfgen_cq`) at the
   paper's choices; then `eval_kolm_probes`. Classical anchors: `run_classical_kolm`, `run_classical_kolm_sweep`,
   `run_classical_cyl`, `run_classical_cyl_surface`, `run_classical_ridge_lse`, `run_classical_lse`.
4. **3D training** (`3d/train`): `train_iclr_jhu_xcube_aug` (variants: DMF-Gen, its spectral-penalty
   SM row, FNO3D-FM, FNO3D), `train_iclr_jhu_scale`, `train_baseline_lfm_xcube_aug` then
   `train_latent_fm_xcube_stage2`, `train_senseiver_iclr`, `train_sit_xcube_matched`,
   `train_deeponet`, `train_deeponetpp`, `train_confild_unified` (one per CoNFiLD config),
   `s3gm_train`, `train_gen4turb_jhu`.
5. **3D evaluation** (`3d/eval`, `classical`): `rescore_classical_jhu`, `run_classical_jhu`,
   `rescore_jhu`, `eval_dmfgen_jhu_tune`, `band_jhu_test`, `eval_fno3d_det_jhu`,
   `eval_sit_jhu_tune`, `eval_sit_jhu_test`, `tune_s3gm_jhu_norm`, `eval_s3gm_jhu_norm_test`,
   `s3gm_finalize`, `eval_gen4turb_jhu`, `confild_eval_canonical` (one per arm), `confild_window1`,
   `confild_window1_scale`, `confild_window1_full`, `eval_dmfgen_spec_jhu`,
   `eval_dmfgen_scale_sharedstats`. SM-only: `eval_jhu_insample`, `eval_sit_matched_seeded`,
   `s3gm_jhu_sweep`, `dump_calib_latentfm` then `run_conformal_all`.
6. **Field dumps**: `dump_2d_gallery`, `dump_surface_gallery`, `dump_sit_2d`, `dump_jhu_frame152`.
7. **Cost**: `bench_dmfgen_cost`, `bench_fno3d_cost`, `instr_sit`, `run_scaling_bench`,
   `run_scaling_fixup`, `run_scaling_baselines`.
8. **Tables, figures and cited numbers**: `paper_figures_tables`, `diagnostics`, `native_s3gm`,
   `native_confild`, `native_confild_rescore`, `native_gen4turb`, `native_summaries`. Outputs go to
   `outputs/paper/`.

Notes:

* The analysis, table and figure scripts name the run directories of record (DemoN number and
  time stamp, e.g. in `src/analysis/paired_ci_jhu.py`). A new run gets a new time stamp, so
  point these names at it or link it under the old name.
* `configs/upstream/kolm_shu_gl_rbf_cq_w296_range.yaml` is the resolved configuration that the
  upstream trainer wrote for the reported run. `eval.fleet_select` requires it to equal the
  run's `resolved_config.yaml`, so replace it with the copy from a new run.
* The latent-FM stage-2 config (`configs/jhu3d/config_baseline_LatentFM_xcube_fixed.yaml`) names
  its stage-1 checkpoint explicitly.
* The in-sample probe's FNO3D-FM leg used an earlier FNO3D-FM run, whose arguments are recorded in
  `configs/jhu3d/recovered_fno3d_matched_args.json`.
* The cost tables were assembled from the outputs of the cost cases and the training logs
  (`losses.csv`, `cost_train.json`, `instrumentation_*.json`); no script writes them.

## Tests

```bash
python -m pytest tests -k "not every_real_artifact"   # that test needs a populated outputs/runs
```

## Upstream code

`third_party/` pins the upstream repositories at the commits the benchmark was run against:
S3GM, SiT, CoNFiLD, Gen4Turbulence, Senseiver, deepxde and deeponet, and
PhyCoFlow-MultiFieldReconstruction for the Kolmogorov DMF-Gen row. `BENCHMARK_DIFF.md` lists, per
method, what was taken from upstream and what was adapted.
