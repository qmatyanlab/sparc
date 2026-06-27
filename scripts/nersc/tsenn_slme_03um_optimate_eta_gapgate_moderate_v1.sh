#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 128
#SBATCH --gpus=2
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_slme_eta_gapgate_mod_test

# ============================================================================
# 10-STEP TEST of the min()-GATED SLME reward (escape fix for the e3nngap collapse).
#
# Background: the weighted-sum e3nngap run 55008527 COLLAPSED -- with the (lower-scale)
# E3NN gap, ~66% of structures sit at gap<0.3 eV; the 0.2-weighted band_gap term + eta
# floor still scored them ~0.11-0.19, so the policy camped (gap mass <0.3 grew 66->75%,
# reward flat ~0.28). Fix: reward = min(eta_term, g_gap) with a HARD-ZERO gate (0.3/1.0),
# so gap<0.3 structures score exactly 0 -> negative advantage -> expelled. Validated in
# principle on the dielectric gapgate_moderate run 55018082 (gap<0.3: 34%->16%).
# Config: configs/reward/tsenn_slme_optimate_eta_gapgate_moderate.yaml.
#
# This launcher is rl_epoch=10 (short) to confirm escape begins (meanGap rising, <0.3
# fraction falling) before committing a full run. Everything else matches 55008527.
# Memory: [[slme-reward-functions]], [[bandgap-model-network-class]].
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_optimate_eta_gapgate_moderate_v1_${JOB_TAG}"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    hydra.job.chdir=true \
    pipeline=sparc model=symmcd reward=tsenn_slme_optimate_eta_gapgate_moderate \
    logger=csv device=cuda \
    eval_size=30 rl_epoch=10 \
    seed=1 deterministic_torch=true \
    pipeline.save_freq=10 \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=1.0 \
    sample_cfg.adaptive_spacegroup.enabled=true \
    sample_cfg.adaptive_spacegroup.ema_decay=0.9 \
    sample_cfg.adaptive_spacegroup.reward_scale=3.0 \
    sample_cfg.adaptive_spacegroup.prior_mix=0.1 \
    sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
    +sample_cfg.adaptive_spacegroup.mix_previous_distribution=true \
    +sample_cfg.adaptive_spacegroup.symprec=0.1 \
    model.finetune_cfg.lr=3e-5 \
    pipeline.topk_ratio=0.5 \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0
