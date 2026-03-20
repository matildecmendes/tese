#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=preprocess    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python convert_to_slices.py --input_folder "Z:\nas-ctm01\homes\mcmendes\dataset_pre_processing\Pre_processed_dataset" --output_folder "Z:\nas-ctm01\datasets\public\lidc-mcmendes\slices"
#python pre_processing.py --ct_folder /nas-ctm01/datasets/public/lidc_npy/3d/ct --mask_folder /nas-ctm01/datasets/public/lidc_npy/3d/lungmask --output_folder /nas-ctm01/homes/mcmendes/dataset_pre_processing/Pre_processed_dataset
#python train.py --data_path "Z:\nas-ctm01\homes\mcmendes\dataset_pre_processing\dataset_slices" --output_path "Z:\nas-ctm01\homes\mcmendes\ddgan_outputs" --batch_size 4 --epochs 1 --limit 50