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
#SBATCH --job-name=sparc_slme_bg02eta08_uniform

# ============================================================================
# Uniform-anchor SLME run (2026-06-18) — "what space group does the SLME reward
# actually favor?"  Same as run A (job 54346667, the 0.2/0.8 ascending SLME
# reward tsenn_slme_optimate_bg02_eta08) EXCEPT the adaptive-SG anchor is made
# ~uniform over the ~169 supported SGs, so the reward — not the MP-20 prior —
# decides which space groups win. Direct analog of the dielectric uniform-anchor
# run (scripts/tsenn_static_dielectric_layered_uniaxial_symmcd_v3_uniform_anchor.sh,
# jobs 54304128 / 54584243), which killed the cubic-225 dud and revived hexagonal
# layered SGs. Hypothesis (user): SLME (band-gap x eta) is far less symmetry-
# selective than the layered-uniaxial dielectric, so NO strong SG winner emerges.
#
# Uniform-anchor flags vs run A (everything else identical):
#   sg_temperature=100   base^(1/100) ~ uniform over supported SGs (was 1.0)
#   reward_scale=5.0     concentration via exp(scale*ema) (was 3.0)
#   prior_mix=0.15       permanent prior_mix*uniform exploration floor (was 0.1)
#   mix_base             OMIT mix_previous_distribution -> mix with the (uniform)
#                        base each step, no carryover succession (run A used
#                        +mix_previous_distribution=true)
#
# Analyze with scripts/plot_adaptive_spacegroup_gif.py + the SG-mass/effSG read
# (compare final current_prob distribution + realized SGs vs run A 54346667).
# Note: uniform anchor proposes more diverse/larger cells -> slower per-step;
# 48h may reach fewer steps than run A's 120 (cf. dielectric uniform 71/200 in 24h).
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_optimate_bg02_eta08_uniform_anchor_v1_${JOB_TAG}"
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
    model.sample_cfg.sg_temperature=100.0 \
    sample_cfg.adaptive_spacegroup.enabled=true \
    sample_cfg.adaptive_spacegroup.ema_decay=0.9 \
    sample_cfg.adaptive_spacegroup.reward_scale=5.0 \
    sample_cfg.adaptive_spacegroup.prior_mix=0.15 \
    sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
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
