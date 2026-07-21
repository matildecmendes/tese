#!/bin/bash
#
#SBATCH --partition=gpu_min32gb_ext     # Reserved partition
#SBATCH --qos=gpu_min32gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_st3    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.


module purge
module load CUDA/11.1.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stylegan3

# Commands / scripts to run (e.g., python3 train.py)
python train.py \
  --outdir /nas-ctm01/homes/mcmendes/outputs/stylegan3 \
  --data /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --cfg=stylegan3-t \
  --gpus=1 \
  --batch=8 \
  --batch-gpu=4 \
  --gamma=2 \
  --aug=noaug \
  --metrics=none \
  --kimg=1000 \
  --tick=4 \
  --snap=25 \
  --workers=4 \
  --seed=0 \
  --desc=lidc-npy-256 