#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=sample    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python sampling.py \
  --model_path /nas-ctm01/homes/mcmendes/outputs/preliminary_work/checkpoint_epoch_30.pt \
  --real_dir /nas-ctm01/datasets/public/lidc-mcmendes/slices_png \
  --out_dir /nas-ctm01/homes/mcmendes/samples/preliminary \
  --num_samples 100 
