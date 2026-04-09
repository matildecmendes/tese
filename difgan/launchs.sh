#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=test    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python sampling_difgan.py \
  --model_path /nas-ctm01/homes/mcmendes/outputs/denoising_dgan/2/test_run/netG_90.pth \
  --output_dir /nas-ctm01/homes/mcmendes/samples/difgan/1 \
  --num_samples 5000 \
  --batch_size 8 \


