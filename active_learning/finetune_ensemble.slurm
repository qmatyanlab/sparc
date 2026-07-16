#!/bin/bash
# Fine-tune the E3NN band-gap surrogate as a deep ensemble (5 seeds) on a GPU node.
#   sbatch active_learning/finetune_ensemble.slurm
#SBATCH -A m2663_g
#SBATCH -C gpu
#SBATCH -q gpu_regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 01:00:00
#SBATCH -J slme_ftgap
#SBATCH -o active_learning/artifacts/ft_%j.out
#SBATCH -e active_learning/artifacts/ft_%j.out

set -o pipefail
ROOT=/global/cfs/cdirs/m2663/angush/sparc_clean
cd "$ROOT"
source .venv/bin/activate
mkdir -p active_learning/artifacts

for SEED in 0 1 2 3 4; do
  echo "##### fine-tune seed $SEED #####"
  python active_learning/finetune_bandgap.py --seed $SEED --epochs 120 --lr 1e-4 \
      --batch-size 32 --patience 20
done
echo "ensemble fine-tune done"
