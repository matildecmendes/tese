#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=evaluation_stylegan3   # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python analysis3.py \
  --save_real_heatmap \
  --real_mean_path /nas-ctm01/homes/mcmendes/outputs/evaluation/real_stats/NLST/real_mean.npy \
  --out_dir /nas-ctm01/homes/mcmendes/outputs/evaluation/paper/real/NLST \
  --title "Real (NLST)"