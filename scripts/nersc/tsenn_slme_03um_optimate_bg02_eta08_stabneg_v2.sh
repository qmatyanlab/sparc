#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 64
#SBATCH --gpus=2
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_slme_bg02eta08_stabneg_v2

# ============================================================================
# stabneg v2 (2026-06-18) — SOFTENED unstable hard-negative penalty.
#
# v1 (job 54523927) CRASHED at step 8 in a generator death-spiral: from step 0
# the hard negatives OUTNUMBERED the positives (11 neg vs 5 stable survivors),
# the floored (reward=0, mean-baseline) negatives dominated every update, the
# policy ran away from the prior (loss_kl 1.6 -> 454), the generator degraded
# until 0 stable structures survived -> mattergen MetricsEvaluator "No data
# provided" assertion. See [[session-checkpoint]] / [[slme-reward-functions]].
#
# Softening changes vs v1 (everything else identical to run A = 54346667):
#   hard_negative_ratio=0.5  NEW cap (pipeline/sparc.py): negatives <= 0.5x
#                            the positive batch, so they can never dominate again.
#   n_hard_negatives=8       lower ceiling (ratio cap does the real work).
#   unstable_floor=0.1       milder penalty: with mean baseline ~0.3 the negative
#                            advantage is ~-0.2 instead of -0.3 (floor 0.0).
#   sigma=0.05               STRONGER KL anchor (was 0.025) to hold the policy.
#   sigma_decay_steps=100    slower KL decay (was 50).
#   advantage_baseline=mean  kept (negatives need a central baseline to be < 0).
#
# A = run 54346667 (penalty off). Compare filter_stable_ratio (survival) vs step,
# eta hit-rate, eta-vs-e_hull frontier, and loss_kl (must stay bounded this time).
# Residual risk: if it still collapses to 0 survivors it will crash (no guard yet);
# watch loss_kl / filter_kept early.
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_optimate_bg02_eta08_stabneg_v2_${JOB_TAG}"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=symmcd reward=tsenn_slme_optimate_bg02_eta08 \
    logger=csv device=cuda \
    eval_size=30 rl_epoch=200 \
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
    +sample_cfg.adaptive_spacegroup.symprec=0.01 \
    sample_cfg.filter.penalize_unstable=true \
    pipeline.finetune_cfg.advantage_baseline=mean \
    pipeline.finetune_cfg.n_hard_negatives=8 \
    +pipeline.finetune_cfg.hard_negative_ratio=0.5 \
    pipeline.finetune_cfg.unstable_floor=0.1 \
    model.finetune_cfg.lr=3e-5 \
    pipeline.topk_ratio=0.5 \
    pipeline.finetune_cfg.sigma=0.05 \
    pipeline.finetune_cfg.sigma_decay_steps=100 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    reward.prop_cfg.1.calculator.energy_max=20.0 \
    reward.prop_cfg.1.calculator.integration_lower_bound=band_gap \
    reward.prop_cfg.1.calculator.eta_backend=pymatgen \
    reward.prop_cfg.1.calculator.thickness_um=0.3 \
    reward.prop_cfg.2.calculator.thickness_um=0.3 \
    reward.prop_cfg.3.calculator.thickness_um=0.3
