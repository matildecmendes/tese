#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=pre_p   # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python pre_processing_nlst.py \
    --input_folder /nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/raw_data/manifest-1741808882472/NLST \
    --output_folder /nas-ctm01/datasets/public/lidc-mcmendes/slices_nlst

