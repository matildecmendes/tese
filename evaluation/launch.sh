#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=evaluation_ddpm4   # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python analysis.py \
  --fake_dir /nas-ctm01/homes/mcmendes/samples/diffusion-stylegan2 \
  --out_dir /nas-ctm01/homes/mcmendes/outputs/evaluation/difstylegan2/hists2 \
  --real_hist_path /nas-ctm01/homes/mcmendes/outputs/evaluation/real_stats/real_hist.npy \
  --real_mean_path /nas-ctm01/homes/mcmendes/outputs/evaluation/real_stats/real_mean.npy \
  --heatmaps --histograms
