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
#SBATCH --job-name=sparc_slme_optimate_bg02eta08_adaptive

# ============================================================================
# NOTE TO SELF (Claude) — context for revisiting this run:
#
# This is the chosen SLME reward fix after a long investigation (2026-06-12).
# Reward = 0.2*band_gap(ascending [0.5,3.0]) + 0.8*eta(ascending [0,0.35]),
# config configs/reward/tsenn_slme_optimate_bg02_eta08.yaml. thickness 0.3 um.
#
# Backstory:
#  - run 54218097 (tsenn_slme_03um_optimate_etaheavy) used a CENTERED band_gap
#    term (Scheme E, centered@1.25) and COLLAPSED to band_gap->0 (82% of
#    structures <0.5 eV by loop 54).
#  - We considered a "tent" fix (Scheme F, centered maxv=0.75) but REFUTED it:
#    the finetune uses baseline=min(top-k) + centered advantage, which subtracts
#    the absolute corner reward, so centering can't fix the collapse. Scheme F
#    allocates top-k advantage ~identically to eta_heavy (which collapsed).
#  - Decision: stay ASCENDING (band_gap = broad corner-escape engine; eta =
#    1.25 eV targeting), but reweight 0.4/0.6 -> 0.2/0.8 to sharpen the eta
#    target and shrink the wide-gap overshoot (ascending's known flaw).
#  - Weight sweep (scripts/compare_ascending_weights.py) picked 0.2/0.8: keeps
#    ~79% of top-k advantage on escaped structures while raising target[1,1.5]
#    reward 0.555->0.640 and cutting wide-gap(>2eV) reward 0.489->0.377.
#
# A/B comparisons when revisiting:
#  - vs 54218097 (eta_heavy centered, collapsed) — does ascending+reweight avoid
#    the band_gap->0 collapse?
#  - vs the optimate 0.4/0.6 baseline (tsenn_slme_03um_symmcd_v3_seed1_optimate_
#    adaptive_53904266 per memory) — does more eta weight raise the eta>0.25 hit
#    rate without losing escape?
# Plot with: scripts/plot_tsenn_slme_results.py <this run dir>
# Memory: [[slme-reward-functions]].
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1_${JOB_TAG}"
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
