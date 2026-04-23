#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_st2    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python train.py \
  --outdir /nas-ctm01/homes/mcmendes/outputs/diffusion-stylegan2/final2 \
  --data /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --subset 6000 \
  --kimg 500 \
  --gpus 1 \
  --cfg paper256 \
  --batch 8 \
  --snap 5 \
  --mirror False \
  --aug diff \
  --noise_sd 0.02 \
  --t_min 5 \
  --t_max 500 \
  --exp_id debug 
