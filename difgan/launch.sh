#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext     # Reserved partition
#SBATCH --qos=gpu_min11gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python train.py \
  --data_path /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --output_dir /nas-ctm01/homes/mcmendes/outputs/difgan/4 \
  --exp test_run \
  --batch_size 8 \
  --num_epoch 100 \
  --num_channels 1 \
  --image_size 256 \
  --ch_mult 1 2 2 4 \
  --dataset custom \
  --save_ckpt_every 5 \
  --num_timesteps 4 \
  --use_ema 
