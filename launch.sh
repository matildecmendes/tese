#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=debug    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
#python convert_to_slices.py --input_folder /nas-ctm01/homes/mcmendes/dataset_pre_processing/Pre_processed_dataset --output_folder /nas-ctm01/datasets/public/lidc-mcmendes/slices
python train.py --data_path /nas-ctm01/datasets/public/lidc-mcmendes/slices --output_path /nas-ctm01/homes/mcmendes/ddgan/outputs --batch_size 4 --epochs 1 --limit 50
