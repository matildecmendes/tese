#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=slices   # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python imagens.py \
    --ct /nas-ctm01/datasets/public/lidc_npy/3d/ct/LIDC-IDRI-0002.npy \
    --mask /nas-ctm01/datasets/public/lidc_npy/3d/lungmask/LIDC-IDRI-0002.npy \
    --output /nas-ctm01/homes/mcmendes/outputs/images/patient_2/50 \
    --slice 50
