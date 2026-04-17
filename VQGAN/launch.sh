#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=debug_gan    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python main.py \
  --base configs/custom_ct_vqgan.yaml \
  -t True \
  --gpus 1, \
  --precision 16 \
  --log_every_n_steps 50 \
  -- max_epochs 1