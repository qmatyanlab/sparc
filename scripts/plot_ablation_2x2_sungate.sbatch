#!/bin/bash
#SBATCH --job-name=fig3_sungate
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=exp_res/slurm_fig3_sungate_%j.log

# Rebuild the 2x2 ablation on the S.U.N.-gated protocol once job 3063 finishes.
# relax-on cells are S.U.N.-gated (2817 SymmCD, 3063 DiffCSP); relax-off cells reuse
# 1058 / 1060 (V.U.N -- `stable` is not computable without relaxation).
# Submit with: sbatch --dependency=afterok:<jobid of 3063> scripts/plot_ablation_2x2_sungate.sbatch

set -uo pipefail
cd ~/sparc_v1
source scripts/env.sh
export HF_HUB_OFFLINE=1 SPARC_SKIP_ASSET_DOWNLOAD=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
echo "=== node: $(hostname)  $(date) ==="
.venv/bin/python ~/sparc_v0/exp_res/_abrepr/plot_ablation_2x2.py \
  --symmcd-relaxt "$PROJECT_ROOT/exp_res/dielectric_inplane_isotropy_gapgate_mprime_newbg_b96_sun" \
  --symmcd-relaxf "$PROJECT_ROOT/exp_res/dielectric_inplane_isotropy_gapgate_mprime_newbg_b96_relaxF" \
  --diffcsp-relaxt "$PROJECT_ROOT/exp_res/dielectric_inplane_isotropy_gapgate_diffcsp_newbg_b96_sun_relaxT" \
  --diffcsp-relaxf "$PROJECT_ROOT/exp_res/dielectric_inplane_isotropy_gapgate_diffcsp_newbg_b96_relaxF" \
  --decomp-pt ~/sparc_v0/exp_res/_abrepr/crte3w/step_0137_eval.pt --decomp-index 19 \
  --diffcsp-decomp-cif ~/sparc_v0/exp_res/_abrepr/crte3w/CrTe3W_diffcsp_panelC.cif \
  --output "$PROJECT_ROOT/exp_res/_abrepr/ablation/fig3_ablation_full_sungate.png"
echo "=== plot exit=$? ($(date)) ==="
