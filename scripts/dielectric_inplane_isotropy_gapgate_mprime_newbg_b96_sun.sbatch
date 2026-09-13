#!/bin/bash
#SBATCH --job-name=diel_ipiso_sun
#SBATCH --partition=gpu-shared
#SBATCH --exclude=c2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=exp_res/slurm_diel_ipiso_sun_%j.log

# In-plane-isotropy dielectric run, rerun under the FULL S.U.N. sample gate.
# Identical to dielectric_inplane_isotropy_gapgate_mprime_newbg_b96.sbatch except:
#   - expname/job-name (_sun)
#   - filter metrics [validity,novel,unique] -> [validity,unique,novel,stable] (+relax=true)
# 120 RL loops, eval_size=30, generation batch 96 (unchanged).

set -uo pipefail
cd ~/sparc_v1
source scripts/env.sh
export HF_HUB_OFFLINE=1 SPARC_SKIP_ASSET_DOWNLOAD=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
echo "=== node: $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
.venv/bin/python main.py \
  expname=dielectric_inplane_isotropy_gapgate_mprime_newbg_b96_sun \
  pipeline=sparc model=symmcd reward=fom_inplane_isotropy_gapgate \
  logger=csv device=cuda \
  eval_size=30 rl_epoch=120 \
  pipeline.save_freq=10 \
  seed=1 deterministic_torch=true \
  'sample_cfg.filter.metrics=[validity,unique,novel,stable]' \
  +sample_cfg.filter.relax=true \
  model.model_path="$MODEL_PATH" \
  model.sample_cfg.generation_batch_size=96 \
  model.sample_cfg.batch_size=96 \
  model.sample_cfg.sg_temperature=100.0 \
  model.sample_cfg.finetune_on_representative=true \
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
  +pipeline.checkpoint_eval.prop_key=tsenn_static_dielectric_inplane_isotropy \
  +pipeline.checkpoint_eval.hit_low=0.90 \
  +pipeline.checkpoint_eval.hit_high=1.00
echo "=== run exit=$? ($(date)) ==="
