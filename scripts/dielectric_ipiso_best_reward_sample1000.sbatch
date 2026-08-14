#!/bin/bash
#SBATCH --job-name=diel_s1000
#SBATCH --partition=gpu-shared
#SBATCH --exclude=c2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=exp_res/slurm_diel_s1000_%j.log

set -uo pipefail
cd ~/sparc_v1
source scripts/env.sh
export HF_HUB_OFFLINE=1 SPARC_SKIP_ASSET_DOWNLOAD=1
echo "=== node: $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

RUN="$PROJECT_ROOT/exp_res/dielectric_inplane_isotropy_gapgate_mprime_newbg_b96"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
.venv/bin/python main.py \
  expname=dielectric_ipiso_best_reward_sample1000 \
  pipeline=sparc model=symmcd reward=fom_inplane_isotropy_gapgate \
  logger=csv device=cuda \
  eval_size=1000 \
  pipeline.rl_epoch=1 \
  pipeline.finetune_cfg.epochs=0 \
  pipeline.replay=False \
  pipeline.div_filter=False \
  seed=1 deterministic_torch=true \
  'sample_cfg.filter.metrics=[validity,unique,novel,stable]' \
  +sample_cfg.filter.relax=true \
  model.model_path="$RUN/models/best_reward" \
  model.sample_cfg.batch_size=100 \
  model.sample_cfg.generation_batch_size=100 \
  sample_cfg.num_batches=10 \
  model.sample_cfg.sg_temperature=100.0 \
  model.sample_cfg.finetune_on_representative=true \
  sample_cfg.adaptive_spacegroup.enabled=true \
  sample_cfg.adaptive_spacegroup.ema_decay=0.9 \
  sample_cfg.adaptive_spacegroup.reward_scale=5.0 \
  sample_cfg.adaptive_spacegroup.prior_mix=0.15 \
  sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
  +sample_cfg.adaptive_spacegroup.symprec=0.01 \
  +sample_cfg.adaptive_spacegroup.initial_distribution_csv="$RUN/adaptive_spacegroup/step_0070.csv"
echo "=== run exit=$? ($(date)) ==="
