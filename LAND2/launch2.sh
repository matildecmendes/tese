#!/bin/bash
#
#SBATCH --partition=gpu_min32gb_ext     # Reserved partition
#SBATCH --qos=gpu_min32gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_st3    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate land


# Commands / scripts to run (e.g., python3 train.py)
python scripts/train_ldm_2d.sh \
  --dataset_path /path/to/preprocessed_LIDC \
  --vae_path checkpoints/vae_2d/LAND2D_VAE_LIDC/vae_best_epoch \
  --run_name LAND2D_LDM_LIDC