#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext     # Reserved partition
#SBATCH --qos=gpu_min11gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=debug_ddpm    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

cd /nas-ctm01/homes/mcmendes/tese/ddpm

export PYTHONPATH=$PWD:$PYTHONPATH

# Commands / scripts to run (e.g., python3 train.py)
python scripts/run_lungct.py train \
    --exp_name lungct_diffusion \
    --data_dir /nas-ctm01/datasets/public/lidc-mcmendes/slices \
    --save_dir /nas-ctm01/homes/mcmendes/outputs/ddpm/2 \
    --total_bs 2 \
    --lr 2e-5 \
    --num_diffusion_timesteps 1000 \
    --beta_schedule linear \
    --dropout 0.0 \
    --randflip 0

