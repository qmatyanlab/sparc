#!/bin/bash
#SBATCH --job-name=diel_ipiso
#SBATCH --partition=gpu-shared
#SBATCH --exclude=c2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=exp_res/slurm_diel_ipiso_%j.log
# IN-PLANE ISOTROPY run. Reward = fom_inplane_isotropy_gapgate: targets ONLY the in-plane relation
# eps_xx == eps_yy (score = 1 - inplane_mismatch, window [0.90,1.0]) gated by band gap (0.3->0.8,
# kept for DFT-worthiness). Rewards the in-plane rotation groups (trig/tet/hex) AND cubic at ~1.0,
# pushes orthorhombic to ~0.35. NO z-anisotropy term. Everything else identical to the
# inplane_emphasis launcher (SymmCD + M-prime, relax=TRUE, b96, rl_epoch=120, adaptive-SG knobs).
set -uo pipefail
cd ~/sparc_v1
source scripts/env.sh
export HF_HUB_OFFLINE=1 SPARC_SKIP_ASSET_DOWNLOAD=1
echo "=== node: $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
~/.local/bin/uv run --frozen python main.py \
  expname=dielectric_inplane_isotropy_gapgate_mprime_newbg_b96 \
  pipeline=sparc model=symmcd reward=fom_inplane_isotropy_gapgate \
  logger=csv device=cuda \
  eval_size=30 rl_epoch=120 \
  pipeline.save_freq=10 \
  seed=1 deterministic_torch=true \
  'sample_cfg.filter.metrics=[validity,novel,unique]' \
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
