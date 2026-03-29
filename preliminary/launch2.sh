#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=debug    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python todo.py \
  --data_path /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --out_dir /nas-ctm01/homes/mcmendes/outputs/run1 \
  --epochs 200 \
  --batch_size 8 \
  --max_samples 8