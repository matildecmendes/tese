#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=debug    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
CUDA_LAUNCH_BLOCKING=1 python train.py \
  --data_path /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --out_dir /nas-ctm01/homes/mcmendes/outputs/preliminary_work/3 \
  --epochs 30 \
  --batch_size 8 \
  --max_samples 6000