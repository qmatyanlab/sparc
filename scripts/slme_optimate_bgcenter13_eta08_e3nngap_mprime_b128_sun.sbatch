#!/bin/bash
#SBATCH --job-name=slme_bgc13_sun
#SBATCH --partition=gpu-shared
#SBATCH --exclude=c2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=exp_res/slurm_slme_bgc13_sun_%j.log

# SLME run, rerun with the FULL S.U.N. sample gate pinned explicitly.
# NOTE: cancelled at step 78/120 and not resumed -- see RUNS.md.
# Identical to slme_optimate_bgcenter13_eta08_e3nngap_mprime_b128.sbatch except:
#   - expname/job-name (_sun)
#   - filter metrics pinned to [validity,unique,novel,stable] (+relax=true) instead of
#     inheriting the base default
# 120 RL loops, eval_size=30, generation batch 128 (unchanged).

set -uo pipefail
cd ~/sparc_v1
source scripts/env.sh
export HF_HUB_OFFLINE=1 SPARC_SKIP_ASSET_DOWNLOAD=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
echo "=== node: $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
.venv/bin/python main.py \
  expname=tsenn_slme_03um_optimate_bgcenter13_eta08_e3nngap_mprime_lrdecay098_b128_sun \
  pipeline=sparc model=symmcd reward=tsenn_slme_optimate_bgcenter13_eta08_e3nngap \
  logger=csv device=cuda \
  eval_size=30 rl_epoch=120 \
  pipeline.save_freq=10 \
  seed=1 deterministic_torch=true \
  'sample_cfg.filter.metrics=[validity,unique,novel,stable]' \
  +sample_cfg.filter.relax=true \
  model.model_path="$MODEL_PATH" \
  model.sample_cfg.generation_batch_size=128 \
  model.sample_cfg.batch_size=128 \
  model.sample_cfg.sg_temperature=1.0 \
  model.sample_cfg.finetune_on_representative=true \
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
  pipeline.finetune_cfg.lr_decay=0.98 \
  reward.prop_cfg.1.calculator.energy_max=20.0 \
  reward.prop_cfg.1.calculator.integration_lower_bound=band_gap \
  reward.prop_cfg.1.calculator.eta_backend=pymatgen \
  reward.prop_cfg.1.calculator.thickness_um=0.3 \
  reward.prop_cfg.2.calculator.thickness_um=0.3 \
  reward.prop_cfg.3.calculator.thickness_um=0.3
echo "=== run exit=$? ($(date)) ==="
