#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext     # Reserved partition
#SBATCH --qos=gpu_min11gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python train.py \
  --outdir /nas-ctm01/homes/mcmendes/outputs/diffusion-stylegan2/1 \
  --data /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --gpus 1 \
  --cfg paper256 \
  --batch 16 \
  --snap 5 \
  --mirror False \
  --aug no \
  --noise_sd 0.02 \
  --t_min 5 \
  --t_max 500 \
  --exp_id lungCT_npy

