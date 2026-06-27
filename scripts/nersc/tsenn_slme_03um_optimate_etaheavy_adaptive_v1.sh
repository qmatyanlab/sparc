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
#SBATCH --job-name=sparc_slme_optimate_etaheavy_adaptive

# A/B vs tsenn_slme_03um_symmcd_v3_seed1_optimate_adaptive_53904266:
# identical config (adaptive SG, sg_temperature=1.0, eval_size=30, lr=3e-5,
# thickness 0.3 um), ONLY the reward changes:
#   reward=tsenn_slme_optimate  ->  reward=tsenn_slme_optimate_eta_heavy
# (band gap centered at 1.25 eV weight 0.2, eta weight 0.8). Goal: raise the
# eta>0.25 hit rate by removing the ascending-band-gap pull toward wide gaps.

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_slme_03um_optimate_etaheavy_adaptive_v1_${JOB_TAG}"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=symmcd reward=tsenn_slme_optimate_eta_heavy \
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
