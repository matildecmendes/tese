#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=fid_real  # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python metrics.py \
  --fid_dir /nas-ctm01/homes/mcmendes/samples/denoising_dgan/2 \
  --fake_dir /nas-ctm01/homes/mcmendes/samples/denoising_dgan/NLST \
  --real_dir /nas-ctm01/datasets/public/lidc-mcmendes/slices_png_nlst \
  --output_dir /nas-ctm01/homes/mcmendes/outputs/evaluation/FID \
  --compute_same_fid


