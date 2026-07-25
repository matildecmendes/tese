#!/bin/bash
#
#SBATCH --partition=gpu_min32gb_ext     # Reserved partition
#SBATCH --qos=gpu_min32gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_land    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate land

# Garante que o job corre a partir da raiz do repositório.
cd "$SLURM_SUBMIT_DIR"

# Commands / scripts to run (e.g., python3 train.py)
srun bash scripts/train_vae_2d.sh \
  /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  LAND2D_VAE_LIDC \
  --model_dir /nas-ctm01/homes/mcmendes/outputs/land \
  --log_dir /nas-ctm01/homes/mcmendes/outputs/land/logs \
  --max_samples 20000 \
  --epochs 150 \
  --batch_size 8 \
  --val_batch_size 8 \
  --num_workers 4 \
  --seed 42