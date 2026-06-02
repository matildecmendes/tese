#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext     # Reserved partition
#SBATCH --qos=gpu_min11gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=rs_ddgan    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python -u random_search.py $1
