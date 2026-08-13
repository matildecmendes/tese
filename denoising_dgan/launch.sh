#!/bin/bash
#
#SBATCH --partition=gpu_min80gb_ext     # Reserved partition
#SBATCH --qos=gpu_min80gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_NLST    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

source ~/.bashrc
conda activate ddgan_env

# Commands / scripts to run (e.g., python3 train.py)
python train_ddgan.py \
  --resume \
  --data_path /nas-ctm01/datasets/public/lidc-mcmendes/slices_nlst \
  --output_dir /nas-ctm01/homes/mcmendes/outputs/denoising_dgan/NLST \
  --exp nlst_random_20k \
  --max_samples 20000 \
  --subset_seed 42 \
  --batch_size 8 \
  --num_epoch 150 \
  --num_channels 1 \
  --image_size 256 \
  --ch_mult 1 2 2 4 \
  --dataset custom \
  --save_ckpt_every 25 \
  --save_content \
  --save_content_every 25