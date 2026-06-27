# SPARC — Symmetry- and Property-Aware Regularized Crystal generation

SPARC fine-tunes symmetry-aware crystal diffusion models (SymmCD / DiffCSP) with reinforcement
learning to generate crystals optimized for target properties — band gap, dielectric response,
and solar-cell efficiency (SLME) — scored by fast E3NN/ALIGNN surrogates with optional MLIP
(MatterSim) relaxation.

> Paper: _add citation / link here._

## Repository layout
- `main.py` — Hydra entry point (instantiates `pipeline.mat_invent.SPARC` and calls `run_rl`)
- `configs/` — Hydra configs (`base.yaml`, `model/`, `pipeline/`, `reward/`, `logger/`)
- `models/` — model suites (`suite/symmcd.py`, `suite/diffcsp.py`, `suite/mattergen.py`) + vendored model code
- `symmcd/`, `pipeline/`, `rewards/`, `memory/` — diffusion model, RL loop, reward calculators, replay buffer
- `utils/assets.py` — auto-downloads large assets from HuggingFace on first use
- `scripts/` — portable helpers; `scripts/nersc/` holds the author's original SLURM scripts (reference only)
- `RUNS.md` — catalog of published runs and their commands

## 1. Install (single `uv` environment)
Requires [uv](https://docs.astral.sh/uv/) and Linux with a recent NVIDIA driver. The pinned
wheels target **CUDA 11.8 (`cu118`)**; any driver supporting CUDA ≥ 11.8 works. Python 3.10 is
pinned via `.python-version` (uv fetches it automatically).

```bash
git clone git@github.com:AngusHsuPhys/sparc_v0.git
cd sparc_v0
uv sync --frozen
```
This installs `torch 2.2.1+cu118`, PyG (`torch_scatter/sparse/cluster`), DGL, MatterGen (pinned
to `v1.0.3`), the materials stack (pymatgen / ase / e3nn / alignn / atomate2 / mattersim), and
`sparc` itself (editable). Then either `source .venv/bin/activate` or prefix commands with `uv run`.

## 2. Assets (auto-downloaded from HuggingFace)
Model checkpoints and datasets (~13 GB) are **not** in git — they live in the HuggingFace repo
`AngusHsuPhys/sparc-assets` and download automatically the first time each file is needed, landing
at the exact paths the configs/checkpoints expect (no config edits required). While the repo is
private, authenticate once:
```bash
huggingface-cli login        # or: export HF_TOKEN=hf_xxx
```
Optionally prefetch (handy before submitting a job, or on a login node that shares the filesystem
with offline compute nodes):
```bash
python scripts/download_assets.py        # ~4.2 GB "core": surrogates + mp_20 + SymmCD base
python scripts/download_assets.py --all  # also the ~8.6 GB caches (only needed to re-train surrogates)
```
Groups: **core** (run the experiments), **cached** (re-train surrogates), **optional** (also in git).
Env knobs: `SPARC_HF_REPO`, `SPARC_HF_REPO_TYPE`, `SPARC_SKIP_ASSET_DOWNLOAD`, `HF_HUB_OFFLINE`.

## 3. Quickstart smoke test
```bash
source scripts/env.sh        # sets PROJECT_ROOT (required) + runtime dirs
uv run python main.py \
  expname=smoke pipeline=sparc model=symmcd reward=band_gap \
  logger=csv device=cuda eval_size=12 rl_epoch=2 \
  model.model_path="$MODEL_PATH" seed=1 deterministic_torch=true
```
Downloads the SymmCD base + band-gap surrogate, samples a few crystals, scores them, runs 2 RL
steps, and writes `exp_res/smoke/{metrics.csv,samples/}`.

## 4. Reproduce the key experiments
Run from the repo root (so `${hydra:runtime.cwd}` resolves to the project root). These are
representative commands; the exact published overrides live in `scripts/nersc/` and `RUNS.md`.

SLME (headline: band gap + η):
```bash
uv run python main.py expname=slme pipeline=sparc model=symmcd \
  reward=tsenn_slme_optimate_bg02_eta08 logger=csv device=cuda \
  eval_size=30 rl_epoch=200 seed=1 deterministic_torch=true model.model_path="$MODEL_PATH"
```
SLME with the E3NN band-gap gate: as above with `reward=tsenn_slme_optimate_bg02_eta08_e3nngap`.

Static dielectric (layered/uniaxial, with relaxation):
```bash
uv run python main.py expname=dielectric pipeline=sparc model=symmcd \
  reward=fom_layered_uniaxial_gapgate_moderate logger=csv device=cuda \
  eval_size=24 rl_epoch=200 seed=1 deterministic_torch=true \
  model.model_path="$MODEL_PATH" model.sample_cfg.filter.relax=true
```
Band-gap target:
```bash
uv run python main.py expname=bandgap pipeline=sparc model=symmcd \
  reward=band_gap logger=csv device=cuda eval_size=24 rl_epoch=200 \
  seed=1 deterministic_torch=true model.model_path="$MODEL_PATH"
```

## 5. Reproducibility
`seed=1 deterministic_torch=true` seeds python/numpy/torch and enables deterministic algorithms
(`CUBLAS_WORKSPACE_CONFIG`, cuDNN deterministic). Results reproduce as **trends**; bitwise identity
additionally requires the same GPU architecture + driver and the pinned `cu118` wheels. A few
third-party ops remain nondeterministic (run with `warn_only`).

## 6. Clusters
`scripts/slurm_gpu.sbatch` and `scripts/slurm_shared.sbatch` are generic templates — fill in
`<ACCOUNT>/<PARTITION>/<QOS>`. The author's exact NERSC submission scripts are kept verbatim in
`scripts/nersc/` for reference (they hardcode NERSC paths/account and are not portable).
`scripts/*.py` are paper-figure / analysis utilities; many expect run directories under `exp_res/`
that are not shipped.

## License
See [`LICENSE`](LICENSE).
