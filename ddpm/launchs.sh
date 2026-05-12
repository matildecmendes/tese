#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext     # Reserved partition
#SBATCH --qos=gpu_min11gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=debug_ddpm    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

cd /nas-ctm01/homes/mcmendes/tese/ddpm

export PYTHONPATH=$PWD

python scripts/sample_lungct.py \
  --model_dir /nas-ctm01/homes/mcmendes/outputs/ddpm/2/2026-05-12/lungct_diffusion_lungct_unet2d16b2_adam_bs2_lr2e-05w5000_beta0.0001-0.02-linear_t1000_eps-fixedlarge-mse_dropout0.0_randflip0 \
  --ckpt model.ckpt-1000 \
  --output_dir /nas-ctm01/homes/mcmendes/samples/ddpm/1000 \
  --data_dir /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --num_samples 100 \
  --batch_size 4