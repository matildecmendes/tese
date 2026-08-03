#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext     # Reserved partition
#SBATCH --qos=gpu_min11gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=metrics_land_100  # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate metrics_env

cd "$SLURM_SUBMIT_DIR"

echo "============================================"
echo "METRICS"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not defined}"
echo "Start: $(date)"
echo "============================================"

# Commands / scripts to run (e.g., python3 train.py)
python metrics.py \
  --fake_dir /nas-ctm01/homes/mcmendes/samples/stylegan3_nlst \
  --real_dir /nas-ctm01/datasets/public/lidc-mcmendes/slices_png_nlst \
  --output_dir /nas-ctm01/homes/mcmendes/outputs/evaluation/stylegan3/metrics_nlst \
  --compute_fid \
  --compute_cmmd \
  --compute_msssim \
  --compute_frd \
  --compute_mae_mse \
  --compute_pr \
  --cmmd_path /nas-ctm01/homes/mcmendes/tese/cmmd-pytorch
