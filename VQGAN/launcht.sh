#!/bin/bash
#
#SBATCH --partition=gpu_min24gb_ext     # Reserved partition
#SBATCH --qos=gpu_min24gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_t    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python main.py \
  --base configs/transformer_ct.yaml \
  -t \
  --gpus 0, \
  --name transformer_ct \
  --logdir /nas-ctm01/homes/mcmendes/outputs/VQGAN/transformer_ct \
  --ckpt /nas-ctm01/homes/mcmendes/outputs/VQGAN/2026-04-19T15-50-17_lung_ct/checkpoints/last.ckpt \
  --max_epochs 250 \
  data.params.train.params.data_root=/nas-ctm01/datasets/public/lidc-mcmendes/slices
