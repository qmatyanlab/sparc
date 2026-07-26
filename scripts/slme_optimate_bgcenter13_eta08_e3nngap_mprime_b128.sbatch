#!/bin/bash
#SBATCH --job-name=slme_bgc13
#SBATCH --partition=gpu-shared
#SBATCH --exclude=c2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=exp_res/slurm_slme_bgc13_%j.log
# SLME run: sparc_v1 rerun of sparc_v0's
#   tsenn_slme_03um_optimate_bgcenter13_eta08_e3nngap_mprime_lrdecay098_b128_c2
# Reward = 0.2*band_gap + 0.8*eta (reduce: weight, NOT gated). band_gap is a HARD-ZERO TENT
# centered at 1.3 eV (Shockley-Queisser PV optimum; E3NN gap = TSENN_bandgap.torch, minv=0,maxv=1).
# eta = TSENNSLME (dielectric-spectra surrogate), integrated over AM1.5G from the E3NN gap up at
# 0.3 um thickness. SymmCD + M-prime (finetune_on_representative + adaptive space groups),
# generation batch 128, sg_temperature=1.0, lr=3e-5, lr_decay=0.98. rl_epoch=120 (v1 rerun).
# Overrides replicate exp_res/.../\.hydra/overrides.yaml exactly except rl_epoch 200->120 and expname.
set -uo pipefail
cd ~/sparc_v1
source scripts/env.sh
export HF_HUB_OFFLINE=1 SPARC_SKIP_ASSET_DOWNLOAD=1
echo "=== node: $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
~/.local/bin/uv run --frozen python main.py \
  expname=tsenn_slme_03um_optimate_bgcenter13_eta08_e3nngap_mprime_lrdecay098_b128_v1 \
  pipeline=sparc model=symmcd reward=tsenn_slme_optimate_bgcenter13_eta08_e3nngap \
  logger=csv device=cuda \
  eval_size=30 rl_epoch=120 \
  pipeline.save_freq=10 \
  seed=1 deterministic_torch=true \
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
