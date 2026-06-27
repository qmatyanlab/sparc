#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 128
#SBATCH --gpus=1
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_tsenn_v3_seed1_best_reward_sample

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_symmcd_v3_seed1_best_reward_sample_${JOB_TAG}"
CHECKPOINT_DIR="${PROJECT_ROOT}/exp_res/tsenn_slme_03um_symmcd_v3_seed1/models/best_reward"

mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
export HF_HUB_DISABLE_TELEMETRY=1
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=symmcd reward=tsenn_slme \
    logger=csv device=cuda \
    eval_size=3000 \
    pipeline.rl_epoch=1 \
    sample_cfg.num_batches=32 \
    +sample_cfg.filter.relax=false \
    'sample_cfg.filter.metrics=[validity,novel,unique]' \
    model.model_path="${CHECKPOINT_DIR}" \
    pipeline.finetune_cfg.epochs=0 \
    pipeline.replay=False \
    pipeline.div_filter=False \
    model.sample_cfg.batch_size=128 \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=3.0 \
    reward.prop_cfg.1.calculator.thickness_um=0.3 \
    reward.prop_cfg.2.calculator.thickness_um=0.3 \
    reward.prop_cfg.3.calculator.thickness_um=0.3
