#!/usr/bin/env bash
# Regenerate the FULL deliverables set for a run's E3NN-rescored "view" folder.
#
# Each plot script defaults its output to <run_dir>/deliverables/..., so we pass the
# rescored view (<run>/e3nn_bandgap) as the run_dir and outputs land in
# <run>/e3nn_bandgap/deliverables/. Resilient: a failing script is logged and skipped.
#
# Usage: scripts/replot_e3nn_bandgap_deliverables.sh [RUN_DIR] [VIEW_SUBDIR]
set -uo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m2663/angush/sparc"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RUN_DIR="${1:-${PROJECT_ROOT}/exp_res/tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1_54346667}"
VIEW_SUBDIR="${2:-e3nn_bandgap}"
VIEW="${RUN_DIR}/${VIEW_SUBDIR}"
DELIV="${VIEW}/deliverables"
TRAJ="${DELIV}/trajectory"
cd "${PROJECT_ROOT}"

if [[ ! -d "${VIEW}/rewards/bandgap" ]]; then
  echo "ERROR: ${VIEW}/rewards/bandgap missing -- run rescore_run_rewards.py first." >&2
  exit 1
fi
mkdir -p "${DELIV}" "${TRAJ}"

# step range for distribution-shift early/steered windows
MAX_STEP=$(ls "${VIEW}/rewards/bandgap"/step_*.txt | sed -E 's/.*step_0*([0-9]+)\.txt/\1/' | sort -n | tail -1)
EARLY="0:10"
STEERED="$(( MAX_STEP > 9 ? MAX_STEP-9 : 0 )):$(( MAX_STEP+1 ))"
echo "view=${VIEW}  max_step=${MAX_STEP}  early=${EARLY}  steered=${STEERED}"

OK=(); FAIL=()
run() {  # run <label> <cmd...>
  local label="$1"; shift
  echo "=== [${label}] $* ==="
  if "$@" > "${DELIV}/.replot_${label}.log" 2>&1; then OK+=("${label}");
  else FAIL+=("${label}"); echo "  !! FAILED (${DELIV}/.replot_${label}.log)"; fi
}

# 1. harvest first (composites read trajectory/candidate_shortlist.csv)
run harvest        "${PYTHON}" scripts/harvest_trajectory.py "${VIEW}"

# 2. core SLME + run plots (output -> view/deliverables by default)
run slme_results   "${PYTHON}" scripts/plot_tsenn_slme_results.py "${VIEW}"
run run_plots      "${PYTHON}" scripts/plot_run.py "${VIEW}"
run story_v2       "${PYTHON}" scripts/plot_slme_story_composite_v2.py "${VIEW}"
run story_v3       "${PYTHON}" scripts/plot_slme_story_composite_v3.py "${VIEW}"
run run_story      "${PYTHON}" scripts/plot_run_story_composite.py "${VIEW}"
run fig4_panelc    "${PYTHON}" scripts/plot_slme_fig4_panelc.py "${VIEW}"
run novelty        "${PYTHON}" scripts/plot_novelty_diversity.py "${VIEW}"
run symmetry_top20 "${PYTHON}" scripts/plot_symmetry_shift.py "${VIEW}" --mode sg --top 20
run symmetry_group "${PYTHON}" scripts/plot_symmetry_shift.py "${VIEW}" --mode group
run adaptive_sg    "${PYTHON}" scripts/plot_adaptive_spacegroup_gif.py "${VIEW}"

# 3. element condensation / periodic / migration (eta- and reward-weighted)
run element_cond   "${PYTHON}" scripts/plot_element_condensation.py "${VIEW}"
run periodic_none  "${PYTHON}" scripts/plot_element_periodic_compare.py "${VIEW}" --weight none
run periodic_rew   "${PYTHON}" scripts/plot_element_periodic_compare.py "${VIEW}" --weight reward
run migration_gif  "${PYTHON}" scripts/demo_element_migration_gif.py "${VIEW}" --weight reward

# 4. distribution-shift variants (early vs steered window)
ds() {  # ds <label> <extra-args...> <out.png> <scatter.png>
  local label="$1"; shift
  local out="${TRAJ}/$1"; local scat="${TRAJ}/$2"; shift 2
  run "ds_${label}" "${PYTHON}" scripts/plot_distribution_shift.py \
    --early "${VIEW}" --steered "${VIEW}" --early-steps "${EARLY}" --steered-steps "${STEERED}" \
    --output "${out}" --scatter-output "${scat}" "$@"
}
ds base   distribution_shift.png           distribution_shift_bg_eta.png
ds kde    distribution_shift_kde.png       distribution_shift_kde_bg_eta.png       --kde
ds eta025 distribution_shift_kde_eta025.png distribution_shift_kde_eta025_bg_eta.png --kde --target-eta-min 0.25
ds eta030 distribution_shift_kde_eta030.png distribution_shift_kde_eta030_bg_eta.png --kde --target-eta-min 0.30

echo
echo "==== replot summary ===="
echo "OK  (${#OK[@]}): ${OK[*]}"
echo "FAIL(${#FAIL[@]}): ${FAIL[*]}"
echo "deliverables in ${DELIV}"
