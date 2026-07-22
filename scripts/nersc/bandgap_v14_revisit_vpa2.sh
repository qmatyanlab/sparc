#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 36:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_bg_v14_vpa2

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="bandgap3_symmcd_sgtemp3_test_v14_revisit_vpa2_${JOB_TAG}"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=symmcd reward=band_gap_gaussian \
    logger=csv device=cuda \
    eval_size=24 rl_epoch=200 \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=3.0 \
    model.finetune_cfg.lr=3e-5 \
    pipeline.topk_ratio=0.25 \
    pipeline.finetune_cfg.advantage_mode=centered \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    pipeline.replay=true \
    pipeline.div_filter=false \
    sample_cfg.max_atoms_per_structure=40 \
    sample_cfg.min_volume_per_atom=2 \
    sample_cfg.pre_relax_max_num=null \
    sample_cfg.filter.max_n_steps=5000 \
    sample_cfg.filter.max_natoms_per_batch=128 \
    sample_cfg.filter.max_total_atoms_per_relax_call=200 \
    reward.prop_cfg.0.reward_mode=gaussian_target \
    reward.prop_cfg.0.tau=0.75 \
    reward.prop_cfg.0.success_tol=0.3 \
    reward.prop_cfg.0.success_bonus=0.2
