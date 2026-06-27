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
#SBATCH --job-name=sparc_slme_optimate_bg02eta08_e3nngap_adaptive

# ============================================================================
# NOTE TO SELF (Claude) — context for revisiting this run:
#
# E3NN-GAP VARIANT of tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1 (2026-06-24).
# Identical to that run EXCEPT the band gap is the project's own E3NN model
# (optuna_bandgap_trial_2) instead of ALIGNN, in BOTH places ALIGNN was used:
#   - the standalone band_gap reward  (rewards.calculators.E3NNBandGap)
#   - the SLME integration lower bound (TSENNSLME eg_mode=e3nn)
# via config configs/reward/tsenn_slme_optimate_bg02_eta08_e3nngap.yaml.
# Loaded through the softplus-OFF Network; do NOT route the band-gap weights
# through the TSENN Network (softplus-on-0e floors them to ~0.69 eV). Memory:
# [[bandgap-model-network-class]].
#
# Resources: premium QOS (node-exclusive full node) -> -c 128, keep --gpus=2.
#
# Reward = 0.2*band_gap(ascending [0.5,3.0]) + 0.8*eta(ascending [0,0.35]),
# thickness 0.3 um. Same reward shape as 54346667; only the gap MODEL changes,
# so this is a NEW reward (gap feeds the SLME integration bound) -> eta differs
# from 54346667; not directly comparable to its stored ALIGNN eta.
#
# Backstory (shared with the ALIGNN run):
#  - run 54218097 (etaheavy, CENTERED band_gap) COLLAPSED to band_gap->0.
#  - Scheme F "tent" REFUTED: baseline=min(top-k)+centered advantage subtracts
#    the corner reward, so centering can't fix the collapse.
#  - Decision: ASCENDING (band_gap = corner-escape engine; eta = 1.25 eV target),
#    reweighted 0.4/0.6 -> 0.2/0.8 (scripts/compare_ascending_weights.py).
#
# A/B comparisons when revisiting:
#  - vs 54346667 (ALIGNN gap, same reward shape) — does training against the E3NN
#    gap change which structures/space groups win, and the eta>0.25 hit rate?
# Plot with: scripts/plot_tsenn_slme_results.py <this run dir>
# Memory: [[slme-reward-functions]].
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_optimate_bg02_eta08_e3nngap_adaptive_v1_${JOB_TAG}"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    hydra.job.chdir=true \
    pipeline=sparc model=symmcd reward=tsenn_slme_optimate_bg02_eta08_e3nngap \
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
