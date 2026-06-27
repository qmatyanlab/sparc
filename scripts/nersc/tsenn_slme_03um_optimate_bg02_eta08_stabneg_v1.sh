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
#SBATCH --job-name=sparc_slme_bg02eta08_stabneg

# ============================================================================
# NOTE TO SELF (Claude) — "stabneg" treatment run (2026-06-15):
#
# B-arm of an A/B on the unstable hard-negative penalty. IDENTICAL to
# scripts/tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1.sh (the 0.2/0.8
# ascending SLME reward, run A = job 54346667) EXCEPT three new flags:
#   sample_cfg.filter.penalize_unstable=true     keep e_hull-unstable structures
#                                                 (tagged is_unstable) instead of
#                                                 dropping them at the stability gate
#   pipeline.finetune_cfg.n_hard_negatives=15    append up to 15 of them to the
#                                                 finetune batch (~= the 15 top-k
#                                                 positives at eval_size=30, topk 0.5)
#   pipeline.finetune_cfg.advantage_baseline=mean  central baseline so the floored
#                                                 (reward=0) negatives get a NEGATIVE
#                                                 advantage and push the policy away
#
# Hypothesis (see plan + funnel diagnostic): stability is the binding survival
# funnel (~60% of relaxed structures die at e_hull; everything else passes ~99%).
# The reinforce-only loop barely moves it (filter_stable_ratio drifted 0.347->0.407
# over ~100 steps in run A). Explicit negative gradient should raise the stable
# fraction => higher survival. Watch for an eta-vs-e_hull Pareto tension (penalty
# trading some eta for stability) rather than a free survival gain.
#
# A = run 54346667 (no re-run needed; penalty was off / didn't exist there).
# Compare: filter_stable_ratio (survival) vs step (B up, A flat ~0.40), eta hit
# rate, eta-vs-e_hull frontier, reward/KL curves. Plot with scripts/plot_run.py
# and scripts/plot_tsenn_slme_results.py. Memory: [[slme-reward-functions]].
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_optimate_bg02_eta08_stabneg_v1_${JOB_TAG}"
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
    +sample_cfg.adaptive_spacegroup.symprec=0.1 \
    sample_cfg.filter.penalize_unstable=true \
    pipeline.finetune_cfg.advantage_baseline=mean \
    pipeline.finetune_cfg.n_hard_negatives=15 \
    pipeline.finetune_cfg.unstable_floor=0.0 \
    model.finetune_cfg.lr=3e-5 \
    pipeline.topk_ratio=0.5 \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    reward.prop_cfg.1.calculator.energy_max=20.0 \
    reward.prop_cfg.1.calculator.integration_lower_bound=band_gap \
    reward.prop_cfg.1.calculator.eta_backend=pymatgen \
    reward.prop_cfg.1.calculator.thickness_um=0.3 \
    reward.prop_cfg.2.calculator.thickness_um=0.3 \
    reward.prop_cfg.3.calculator.thickness_um=0.3
