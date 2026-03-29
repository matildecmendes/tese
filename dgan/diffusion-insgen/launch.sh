#!/bin/bash
#
#SBATCH --partition=gpu_min12gb_ext     # Reserved partition
#SBATCH --qos=gpu_min12gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python train.py \
  --outdir=/nas-ctm01/homes/mcmendes/outputs/dgan \
  --data_path /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --gpus=1 \
  --cfg=paper256 \
  --batch=4 \
  --mirror=0 \
  --aug=ada \
  --daug=ADA \
  --target=0.6 \
  --kimg=5000 \
  --gamma=0.5 \
  --noise_sd=0.02 \
  --t_min=10 \
  --t_max=250 \
  --no_insgen=0 \
  --workers=2 \
  --kimg=100
