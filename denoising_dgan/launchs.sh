#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=s    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python sampling_ddgan.py \
  --model_path /nas-ctm01/homes/mcmendes/outputs/denoising_dgan/test_run/netG_25.pth \
  --output_dir /nas-ctm01/homes/mcmendes/samples/denoising_dgan \
  --real_img_dir /nas-ctm01/datasets/public/lidc-mcmendes/slices_png \
  --num_samples 5000 \
  --batch_size 8 \
  --compute_fid

