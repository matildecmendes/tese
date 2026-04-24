#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=test    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python sampling_difstylegan2.py \
  --model_path /nas-ctm01/homes/mcmendes/outputs/diffusion-stylegan2/final2/00000-npy_dataset-subset6000-paper256-batch8-target0.6-ada_kimg100-ts_dist-priority-image_augdiff-noise_sd0.02-debug/network-snapshot-000500.pkl \
  --output_dir /nas-ctm01/homes/mcmendes/samples/diffusion-stylegan2 \
  --real_img_dir /nas-ctm01/datasets/public/lidc-mcmendes/slices_png \
  --num_samples 5000 \
  --batch_size 8 \
  --compute_fid

