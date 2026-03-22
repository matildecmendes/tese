#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=debug    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

set -x
set -e

echo "===== START ====="
date

echo "PWD:"
pwd

echo "FILES:"
ls -lah

echo "PYTHON (before env):"
which python
python --version

echo "ACTIVATE ENV"

eval "$(conda shell.bash hook)"  
conda activate ddgan_env

echo "PYTHON (after env):"
which python
python --version

echo "RUNNING SCRIPT"
python train.py

echo "===== END ====="