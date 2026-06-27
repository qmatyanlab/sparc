#!/usr/bin/env bash
# Set up environment for SPARC testing

# Get the absolute path to this repo
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export environment variables
export PROJECT_ROOT="${PROJECT_ROOT}"
export HYDRA_JOBS="${PROJECT_ROOT}/runtime/hydra_jobs"
export WANDB_DIR="${PROJECT_ROOT}/runtime/wandb"
export WANDB_CACHE_DIR="${PROJECT_ROOT}/runtime/wandb_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Create necessary directories if they don't exist
mkdir -p "${HYDRA_JOBS}"
mkdir -p "${WANDB_DIR}"
mkdir -p "${WANDB_CACHE_DIR}"

echo "Environment Setup:"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  HYDRA_JOBS=${HYDRA_JOBS}"
echo "  WANDB_DIR=${WANDB_DIR}"
echo "  WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"

# Run the test command
python main.py \
  expname=test \
  pipeline=mat_invent \
  model=symmcd \
  reward=band_gap \
  device=cuda \
  logger=csv \
  eval_size=12 \
  rl_epoch=30 \
  model.model_path=${PROJECT_ROOT}/runtime/symmcd_pretrained/mp_20 \
  model.sample_cfg.generation_batch_size=128 \
  model.sample_cfg.sg_temperature=3.0 \
  model.finetune_cfg.lr=1e-5 \
  pipeline.finetune_cfg.sigma=0.025 \
  pipeline.finetune_cfg.sigma_decay_steps=50 \
  pipeline.finetune_cfg.lr_decay=1.0
