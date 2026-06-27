#!/bin/bash
# Generic "shared"/multi-GPU SLURM template. Replace the <...> placeholders.
# Relaxation (relax=true) is CPU-bound, so request several CPUs per GPU.
#SBATCH --job-name=sparc
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=exp_res/slurm_%j.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
source scripts/env.sh
uv run python scripts/download_assets.py || true

# Example: layered/uniaxial static-dielectric run with on-the-fly relaxation.
uv run python main.py \
  expname=sparc_dielectric pipeline=sparc model=symmcd \
  reward=fom_layered_uniaxial_gapgate_moderate logger=csv device=cuda \
  eval_size=24 rl_epoch=200 seed=1 deterministic_torch=true \
  model.model_path="${MODEL_PATH}" \
  model.sample_cfg.filter.relax=true
