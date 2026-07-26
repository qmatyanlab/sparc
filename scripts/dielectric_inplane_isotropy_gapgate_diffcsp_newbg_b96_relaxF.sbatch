#!/bin/bash
#SBATCH --job-name=diel_ipiso_dcF
#SBATCH --partition=gpu-shared
#SBATCH --exclude=c2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=exp_res/slurm_diel_ipiso_dcF_%j.log
# 2x2 ablation cell: DiffCSP (plain M, no M-prime / no adaptive-SG), relax=FALSE, IN-PLANE
# ISOTROPY reward. Same as dielectric_inplane_isotropy_gapgate_diffcsp_newbg_b96_relaxT.sbatch
# EXCEPT +sample_cfg.filter.relax=false. DiffCSP loads its own base (diffcsp_mp20, pre-cached).
set -uo pipefail
cd ~/sparc_v1
source scripts/env.sh
export HF_HUB_OFFLINE=1 SPARC_SKIP_ASSET_DOWNLOAD=1
echo "=== node: $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
~/.local/bin/uv run --frozen python main.py \
  expname=dielectric_inplane_isotropy_gapgate_diffcsp_newbg_b96_relaxF \
  pipeline=sparc model=diffcsp reward=fom_inplane_isotropy_gapgate \
  logger=csv device=cuda \
  eval_size=30 rl_epoch=120 \
  pipeline.save_freq=10 \
  seed=1 deterministic_torch=true \
  'sample_cfg.filter.metrics=[validity,novel,unique]' \
  +sample_cfg.filter.relax=false \
  model.sample_cfg.batch_size=96 \
  model.finetune_cfg.lr=3e-5 \
  pipeline.topk_ratio=0.5 \
  pipeline.finetune_cfg.sigma=0.025 \
  pipeline.finetune_cfg.sigma_decay_steps=50 \
  pipeline.finetune_cfg.lr_decay=1.0 \
  +pipeline.checkpoint_eval.prop_key=tsenn_static_dielectric_inplane_isotropy \
  +pipeline.checkpoint_eval.hit_low=0.90 \
  +pipeline.checkpoint_eval.hit_high=1.00
echo "=== run exit=$? ($(date)) ==="
