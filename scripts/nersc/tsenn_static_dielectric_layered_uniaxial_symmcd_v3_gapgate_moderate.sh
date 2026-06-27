#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 128
#SBATCH --gpus=2
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_dielectric_layered_v3_gapgate_moderate

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_static_dielectric_layered_uniaxial_symmcd_v3_gapgate_moderate_${JOB_TAG}"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
export HF_HUB_DISABLE_TELEMETRY=1
cd "${PROJECT_ROOT}"

# MODERATE-gate hedge sibling of ..._gapgate_min.sh (run in parallel). ONLY the reward
# changes between the two: reward=fom_layered_uniaxial_gapgate_moderate -> gap gate ramp
# 0.3/1.0 (midpoint ~0.65 eV) instead of the conservative 0.5/1.2. Everything else
# (model=symmcd, seed=1, adaptive SG, finetune, checkpoint_eval, resources) IDENTICAL to
# run 54584243 / the conservative run, so the three runs are directly comparable.
"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=symmcd reward=fom_layered_uniaxial_gapgate_moderate \
    logger=csv device=cuda \
    eval_size=30 rl_epoch=200 \
    pipeline.save_freq=10 \
    seed=1 deterministic_torch=true \
    +sample_cfg.filter.relax=true \
    'sample_cfg.filter.metrics=[validity,novel,unique]' \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=100.0 \
    sample_cfg.adaptive_spacegroup.enabled=true \
    sample_cfg.adaptive_spacegroup.ema_decay=0.9 \
    sample_cfg.adaptive_spacegroup.reward_scale=5.0 \
    sample_cfg.adaptive_spacegroup.prior_mix=0.15 \
    sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
    +sample_cfg.adaptive_spacegroup.symprec=0.1 \
    model.finetune_cfg.lr=3e-5 \
    pipeline.topk_ratio=0.5 \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    +pipeline.checkpoint_eval.prop_key=tsenn_static_dielectric_layered_uniaxial \
    +pipeline.checkpoint_eval.hit_low=0.08 \
    +pipeline.checkpoint_eval.hit_high=0.20
