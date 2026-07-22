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
#SBATCH --job-name=sparc_bandgap_v21_shared

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
EXPNAME="sparc_bandga_v21"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=symmcd reward=band_gap_wide \
    logger=csv device=cuda \
    eval_size=30 rl_epoch=200 \
    pipeline.save_freq=10 \
    seed=42 deterministic_torch=true \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=1.0 \
    sample_cfg.adaptive_spacegroup.enabled=true \
    sample_cfg.adaptive_spacegroup.ema_decay=0.8 \
    sample_cfg.adaptive_spacegroup.reward_scale=3.0 \
    sample_cfg.adaptive_spacegroup.prior_mix=0.3 \
    sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
    model.finetune_cfg.lr=3e-5 \
    pipeline.topk_ratio=0.25 \
    pipeline.finetune_cfg.advantage_mode=centered \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    +pipeline.checkpoint_eval.prop_key=band_gap \
    +pipeline.checkpoint_eval.hit_low=2.5 \
    +pipeline.checkpoint_eval.hit_high=3.5
