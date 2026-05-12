#!/bin/bash
#
#SBATCH --partition=gpu_min24gb_ext     # Reserved partition
#SBATCH --qos=gpu_min24gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=w_ddpm    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

cd /nas-ctm01/homes/mcmendes/tese/ddpm

export PYTHONPATH=$PWD

# Commands / scripts to run (e.g., python3 train.py)
python scripts/watch_sampling.py \
    --checkpoint_dir /nas-ctm01/homes/mcmendes/outputs/ddpm/2/2026-05-12/lungct_diffusion_lungct_unet2d16b2_adam_bs2_lr2e-05w5000_beta0.0001-0.02-linear_t1000_eps-fixedlarge-mse_dropout0.0_randflip0 \
    --output_dir /nas-ctm01/homes/mcmendes/outputs/ddpm/2/2026-05-12/lungct_diffusion_lungct_unet2d16b2_adam_bs2_lr2e-05w5000_beta0.0001-0.02-linear_t1000_eps-fixedlarge-mse_dropout0.0_randflip0/samples

