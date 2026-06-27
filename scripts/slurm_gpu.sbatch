#!/bin/bash
# Generic single-GPU SLURM template. Replace the <...> placeholders for your cluster.
# The author's exact NERSC submission scripts are kept in scripts/nersc/ for reference.
#SBATCH --job-name=sparc
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=exp_res/slurm_%j.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
source scripts/env.sh

# Prefetch assets (skip with SPARC_SKIP_ASSET_DOWNLOAD=1 if compute nodes are offline
# and you already prefetched on a login node sharing this filesystem).
uv run python scripts/download_assets.py || true

# Example: headline SLME run. See README.md / RUNS.md for the full set of experiments.
uv run python main.py \
  expname=sparc_slme pipeline=sparc model=symmcd \
  reward=tsenn_slme_optimate_bg02_eta08 logger=csv device=cuda \
  eval_size=30 rl_epoch=200 seed=1 deterministic_torch=true \
  model.model_path="${MODEL_PATH}"
