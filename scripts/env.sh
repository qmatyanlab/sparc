#!/usr/bin/env bash
# Source before running any experiment:   source scripts/env.sh
#
# Sets PROJECT_ROOT (REQUIRED: the pretrained-checkpoint hparams reference
# ${oc.env:PROJECT_ROOT}) plus the runtime directories used by the code. It does NOT
# hardcode a Python interpreter — activate the uv venv (`source .venv/bin/activate`)
# or prefix commands with `uv run`.

if PROJECT_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PROJECT_ROOT
export MODEL_PATH="${PROJECT_ROOT}/data/symmcd_pretrained/mp_20"
export SYMMCD_ROOT="${PROJECT_ROOT}"
export WANDB_DIR="${PROJECT_ROOT}/exp_res/wandb"
export WANDB_CACHE_DIR="${PROJECT_ROOT}/exp_res/wandb_cache"

# Avoid CUDA OOM from fragmented PyTorch cache during MatterSim relaxation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${PROJECT_ROOT}/exp_res/hydra_jobs"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
