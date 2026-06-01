#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext     # Reserved partition
#SBATCH --qos=gpu_min11gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=test_seg    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.


# Commands / scripts to run (e.g., python3 train.py)
python test_png_segmentation.py \
    --input_dir /nas-ctm01/homes/mcmendes/samples/denoising_dgan/2 \
    --output_dir /nas-ctm01/homes/mcmendes/outputs/segmentation \
    --weights /nas-ctm01/homes/pfsousa/seg_piu/bestmodel.epoch05.hdf5 \
    --num_images 10

