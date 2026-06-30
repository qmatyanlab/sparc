#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --gpus=2
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_dielectric_xz_v4

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_static_dielectric_xz_ratio_symmcd_v4_${JOB_TAG}"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
export HF_HUB_DISABLE_TELEMETRY=1
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=symmcd reward=tsenn_static_dielectric_xz_ratio \
    logger=csv device=cuda \
    eval_size=30 rl_epoch=200 \
    pipeline.save_freq=10 \
    +sample_cfg.filter.relax=false \
    'sample_cfg.filter.metrics=[validity,novel,unique]' \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=1.0 \
    sample_cfg.adaptive_spacegroup.enabled=true \
    sample_cfg.adaptive_spacegroup.ema_decay=0.9 \
    sample_cfg.adaptive_spacegroup.reward_scale=4.0 \
    sample_cfg.adaptive_spacegroup.prior_mix=0.3 \
    sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
    +sample_cfg.adaptive_spacegroup.symprec=0.01 \
    model.finetune_cfg.lr=3e-5 \
    pipeline.topk_ratio=0.5 \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    reward.prop_cfg.0.maxv=0.02 \
    +pipeline.checkpoint_eval.prop_key=tsenn_static_dielectric_xz_ratio \
    +pipeline.checkpoint_eval.hit_low=0.01 \
    +pipeline.checkpoint_eval.hit_high=0.05
