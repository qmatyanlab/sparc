#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 128
#SBATCH --gpus=1
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_tsenn_optad_best_sample_relaxed

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
SOURCE_RUN="${PROJECT_ROOT}/exp_res/tsenn_slme_03um_symmcd_v3_seed1_optimate_adaptive_53904266"
CHECKPOINT_DIR="${SOURCE_RUN}/models/best_reward"
ADAPTIVE_SG_CSV="${SOURCE_RUN}/adaptive_spacegroup/step_0091.csv"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_symmcd_v3_seed1_optimate_adaptive_best_reward_sample_relaxed_${JOB_TAG}"

mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
export HF_HUB_DISABLE_TELEMETRY=1
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=symmcd reward=tsenn_slme_optimate \
    logger=csv device=cuda \
    eval_size=4096 \
    pipeline.rl_epoch=1 \
    sample_cfg.num_batches=32 \
    seed=1 deterministic_torch=true \
    'sample_cfg.filter.metrics=[validity,unique,stable]' \
    model.model_path="${CHECKPOINT_DIR}" \
    pipeline.finetune_cfg.epochs=0 \
    pipeline.replay=False \
    pipeline.div_filter=False \
    model.sample_cfg.batch_size=128 \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=1.0 \
    sample_cfg.adaptive_spacegroup.enabled=true \
    sample_cfg.adaptive_spacegroup.ema_decay=0.9 \
    sample_cfg.adaptive_spacegroup.reward_scale=3.0 \
    sample_cfg.adaptive_spacegroup.prior_mix=0.1 \
    sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
    +sample_cfg.adaptive_spacegroup.mix_previous_distribution=true \
    +sample_cfg.adaptive_spacegroup.symprec=0.01 \
    +sample_cfg.adaptive_spacegroup.initial_distribution_csv="${ADAPTIVE_SG_CSV}" \
    reward.prop_cfg.1.calculator.thickness_um=0.3 \
    reward.prop_cfg.2.calculator.thickness_um=0.3 \
    reward.prop_cfg.3.calculator.thickness_um=0.3
