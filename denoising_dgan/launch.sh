#!/bin/bash
#
#SBATCH --partition=gpu_min12gb_ext     # Reserved partition
#SBATCH --qos=gpu_min12gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=debug    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python train_ddgan.py --data_path /nas-ctm01/datasets/public/lidc-mcmendes/slices --batch_size 8 --num_channels 1 --image_size 256 --ch_mult 1 2 2 4 --centered
