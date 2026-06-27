#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --gpus=2
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_dielectric_layered_diffcsp_v1

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_static_dielectric_layered_uniaxial_diffcsp_v1_${JOB_TAG}"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
export HF_HUB_DISABLE_TELEMETRY=1
cd "${PROJECT_ROOT}"

# Ablation of the symmcd_v1 run: identical reward and RL settings, but the
# unconstrained DiffCSP sampler (model_path=null -> auto-download from HF).
# Adaptive-spacegroup overrides are dropped: DiffCSP cannot condition on SG.
"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=diffcsp reward=tsenn_static_dielectric_layered_uniaxial \
    logger=csv device=cuda \
    eval_size=30 rl_epoch=200 \
    pipeline.save_freq=10 \
    +sample_cfg.filter.relax=false \
    'sample_cfg.filter.metrics=[validity,novel,unique]' \
    model.finetune_cfg.lr=3e-5 \
    pipeline.topk_ratio=0.5 \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    +pipeline.checkpoint_eval.prop_key=tsenn_static_dielectric_layered_uniaxial \
    +pipeline.checkpoint_eval.hit_low=0.08 \
    +pipeline.checkpoint_eval.hit_high=0.20
