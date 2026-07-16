#!/bin/bash
# Active-learning GPU driver: attribution (Phase A) -> ensemble fine-tune
# (Phase C) -> evaluate base vs fine-tuned against the QE DFT set.
#   sbatch active_learning/al_gpu.slurm
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 128
#SBATCH --gpus=1
#SBATCH -t 01:30:00
#SBATCH -J slme_al
#SBATCH -o active_learning/artifacts/al_%j.out
#SBATCH -e active_learning/artifacts/al_%j.out

set -o pipefail
ROOT=/global/cfs/cdirs/m2663/angush/sparc_clean
cd "$ROOT"
source .venv/bin/activate
mkdir -p active_learning/artifacts
export OMP_NUM_THREADS=8

echo "########## Phase A: SLME attribution (gap vs spectra) ##########"
python -u active_learning/attribution.py --thickness-um 0.3

echo "########## Phase C: ensemble fine-tune (5 seeds) ##########"
for SEED in 0 1 2 3 4; do
  echo "----- seed $SEED -----"
  python -u active_learning/finetune_bandgap.py --seed $SEED --epochs 120 --lr 1e-4 \
      --batch-size 32 --patience 20
done

echo "########## Evaluate base vs fine-tuned ensemble on the QE set ##########"
python -u active_learning/evaluate_on_qe.py

echo "AL GPU driver done"
