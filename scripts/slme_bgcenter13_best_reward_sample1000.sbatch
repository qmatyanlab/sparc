#!/bin/bash
#SBATCH --job-name=slme_s1000
#SBATCH --partition=gpu-shared
#SBATCH --exclude=c2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=exp_res/slurm_slme_s1000_%j.log

set -uo pipefail
cd ~/sparc_v1
source scripts/env.sh
export HF_HUB_OFFLINE=1 SPARC_SKIP_ASSET_DOWNLOAD=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
echo "=== node: $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
C2=/shared/home/angush/sparc_v0/exp_res/tsenn_slme_03um_optimate_bgcenter13_eta08_e3nngap_mprime_lrdecay098_b128_c2
.venv/bin/python main.py \
  expname=slme_bgcenter13_best_reward_sample1000 \
  pipeline=sparc model=symmcd reward=tsenn_slme_optimate_bgcenter13_eta08_e3nngap \
  logger=csv device=cuda \
  eval_size=1000 \
  pipeline.rl_epoch=1 \
  pipeline.finetune_cfg.epochs=0 \
  pipeline.replay=False \
  pipeline.div_filter=False \
  seed=1 deterministic_torch=true \
  'sample_cfg.filter.metrics=[validity,unique,novel,stable]' \
  +sample_cfg.filter.relax=true \
  model.model_path="$C2/models/best_reward" \
  model.sample_cfg.batch_size=100 \
  model.sample_cfg.generation_batch_size=100 \
  sample_cfg.num_batches=10 \
  model.sample_cfg.sg_temperature=1.0 \
  model.sample_cfg.finetune_on_representative=true \
  sample_cfg.adaptive_spacegroup.enabled=true \
  sample_cfg.adaptive_spacegroup.ema_decay=0.9 \
  sample_cfg.adaptive_spacegroup.reward_scale=3.0 \
  sample_cfg.adaptive_spacegroup.prior_mix=0.1 \
  sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
  +sample_cfg.adaptive_spacegroup.mix_previous_distribution=true \
  +sample_cfg.adaptive_spacegroup.symprec=0.01 \
  +sample_cfg.adaptive_spacegroup.initial_distribution_csv="$C2/adaptive_spacegroup/step_0189.csv" \
  reward.prop_cfg.1.calculator.energy_max=20.0 \
  reward.prop_cfg.1.calculator.integration_lower_bound=band_gap \
  reward.prop_cfg.1.calculator.eta_backend=pymatgen \
  reward.prop_cfg.1.calculator.thickness_um=0.3 \
  reward.prop_cfg.2.calculator.thickness_um=0.3 \
  reward.prop_cfg.3.calculator.thickness_um=0.3
echo "=== run exit=$? ($(date)) ==="
