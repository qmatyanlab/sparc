#!/bin/bash
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --gpus=2
#SBATCH --output=/global/cfs/cdirs/m2663/angush/sparc/exp_res/slurm_%j.log
#SBATCH --job-name=sparc_dielectric_layered_diffcsp_relax

# ============================================================================
# NOTE TO SELF (Claude) — "steelman / completeness" condition for the DiffCSP
# ablation (created 2026-06-12).
#
# This reruns the DiffCSP layered-uniaxial ablation with relaxation IN THE LOOP
# (filter.relax=true). It is the completeness counterpart to the relax=false run
# 54289718, which showed DiffCSP generates 97.6% P1 (as-generated, symprec 0.01)
# while SymmCD spreads across genuine high-symmetry groups.
#
# Why this run: answer the "but you'd just relax it" reviewer. The only NEW
# question vs the post-hoc relax-recovery analysis is whether relaxation INSIDE
# the RL loop lets DiffCSP bootstrap toward symmetry/reward it couldn't reach
# with relax=false (gradient now flows through relaxed structures).
#
# Design decisions (see memory [[diffcsp-ablation-static-dielectric]]):
#  - filter.relax=true: score the MatterSim-relaxed minima, not frozen samples.
#    This is the REAL faithfulness lever: un-relaxed structures sit ~0.05 A off
#    their minimum, and that (not symprec) is what makes scoring unfaithful.
#  - standardize_structure stays refined @ standardize_symprec=0.1 (reward
#    default — NOT overridden). standardize defines the x/y/z frame so the
#    layered_uniaxial scalar (reads diagonal eps_xx/eps_yy/eps_zz) is meaningful.
#    Post-relaxation, symprec 0.1 vs 0.01 converge (relaxation removes the
#    approximate-symmetry limbo), and the refined-vs-raw score effect was already
#    measured tiny (+0.0075). So 0.1 is faithful here AND matches run A.
#  - filter metrics [validity,novel,unique] kept (same as run A / 54289718).
#  - seed=1 deterministic_torch=true (publication reproducibility convention).
#  - adaptive_spacegroup NOT set: DiffCSP can't condition on SG (auto-disabled).
#  - model.model_path unset: DiffCSP auto-downloads diffcsp_mp20 from HF.
#
# FAIR COMPARISON CAVEAT: run A (SymmCD 53154012) used relax=false. For an
# apples-to-apples relax=true comparison, also run the matched SymmCD script
# scripts/tsenn_static_dielectric_layered_uniaxial_symmcd_relax_v1.sh (relax
# slightly HURTS SymmCD's symmetry 83%->69% uniaxial, so it must be matched).
#
# Analyze with: scripts/plot_symmetry_architecture.py, scripts/relax_recovery_figure.py
# NOTE: with relax=true the saved eval structures are ALREADY relaxed, so the
# as-generated P1 architecture claim must still use the relax=false run 54289718.
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
JOB_TAG="${SLURM_JOB_ID:-manual}"
EXPNAME="tsenn_static_dielectric_layered_uniaxial_diffcsp_relax_v1_${JOB_TAG}"
mkdir -p "${PROJECT_ROOT}/exp_res/${EXPNAME}"
exec >"${PROJECT_ROOT}/exp_res/${EXPNAME}/main.log" 2>&1
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/scripts/env.sh"
export HF_HUB_DISABLE_TELEMETRY=1
cd "${PROJECT_ROOT}"

"${PYTHON}" -u main.py \
    expname="${EXPNAME}" \
    pipeline=sparc model=diffcsp reward=tsenn_static_dielectric_layered_uniaxial \
    logger=csv device=cuda \
    eval_size=30 rl_epoch=200 \
    seed=1 deterministic_torch=true \
    pipeline.save_freq=10 \
    +sample_cfg.filter.relax=true \
    'sample_cfg.filter.metrics=[validity,novel,unique]' \
    model.finetune_cfg.lr=3e-5 \
    pipeline.topk_ratio=0.5 \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    +pipeline.checkpoint_eval.prop_key=tsenn_static_dielectric_layered_uniaxial \
    +pipeline.checkpoint_eval.hit_low=0.08 \
    +pipeline.checkpoint_eval.hit_high=0.20
