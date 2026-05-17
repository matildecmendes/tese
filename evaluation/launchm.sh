#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=metrics   # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python metrics.py \
  --fake_dir /nas-ctm01/homes/mcmendes/samples/vqgan \
  --real_dir /nas-ctm01/datasets/public/lidc-mcmendes/slices_png \
  --output_dir /nas-ctm01/homes/mcmendes/outputs/evaluation/metrics/vqgan/metrics \
  --compute_fid \
  --compute_cmmd \
  --compute_msssim \
  --compute_frd \
  --compute_mae_mse \
  --compute_pr \
  --cmmd_path /nas-ctm01/homes/mcmendes/tese/cmmd-pytorch


