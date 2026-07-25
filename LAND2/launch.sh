#!/bin/bash
#
#SBATCH --partition=gpu_min32gb_ext     # Reserved partition
#SBATCH --qos=gpu_min32gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_land    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate land


# Commands / scripts to run (e.g., python3 train.py)
python scripts/train_vae_2d.sh \
  --dataset_path /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --run_name LAND2D_VAE_LIDC