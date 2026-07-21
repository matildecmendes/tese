#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_st2    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stylegan3

# Commands / scripts to run (e.g., python3 train.py)
python gen_images.py \
  --network /nas-ctm01/homes/mcmendes/outputs/stylegan3/lidc/00000-stylegan3-t-slices-gpus1-batch32-gamma2-lidc-npy-256/network-snapshot-001000.pkl \
  --outdir /caminho/generated_stylegan3 \
  --seeds=0-4999 \
  --trunc=1 \
  --noise-mode=const