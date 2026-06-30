#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 128
#SBATCH --gpus=1
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_slme_54346667_step69_sample

# Fig 5 discovery sampling: best_reward checkpoint (STEP 69) of the SLME run
# tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1_54346667.
# ~2000 samples, relax=true, filter validity+unique+stable. Compared head-to-head
# with the step-99 (loop_0099) sample to pick the Fig-5 checkpoint.

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
SOURCE_RUN="${PROJECT_ROOT}/exp_res/tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1_54346667"
CHECKPOINT_DIR="${SOURCE_RUN}/models/best_reward"
ADAPTIVE_SG_CSV="${SOURCE_RUN}/adaptive_spacegroup/step_0069.csv"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1_54346667_best_reward_step69_sample_relaxed_${JOB_TAG}"

mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
export HF_HUB_DISABLE_TELEMETRY=1
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=symmcd reward=tsenn_slme_optimate_bg02_eta08 \
    logger=csv device=cuda \
    eval_size=2048 \
    pipeline.rl_epoch=1 \
    sample_cfg.num_batches=16 \
    seed=1 deterministic_torch=true \
    'sample_cfg.filter.metrics=[validity,unique,stable]' \
    +sample_cfg.filter.relax=true \
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
    reward.prop_cfg.1.calculator.energy_max=20.0 \
    reward.prop_cfg.1.calculator.integration_lower_bound=band_gap \
    reward.prop_cfg.1.calculator.eta_backend=pymatgen \
    reward.prop_cfg.1.calculator.thickness_um=0.3 \
    reward.prop_cfg.2.calculator.thickness_um=0.3 \
    reward.prop_cfg.3.calculator.thickness_um=0.3
