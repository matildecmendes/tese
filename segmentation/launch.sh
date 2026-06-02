#!/bin/bash
#
#SBATCH --partition=gpu_min32gb_ext     # Reserved partition
#SBATCH --qos=gpu_min32gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=test_seg    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "======================================="
echo "HOSTNAME"
hostname

echo "======================================="
echo "NVIDIA-SMI"
which nvidia-smi
nvidia-smi

echo "======================================="
echo "CUDA"
nvcc --version

echo "======================================="
echo "PYTHON"
which python

echo "======================================="
echo "TENSORFLOW"
python - << EOF
import tensorflow as tf
print("TF VERSION:", tf.__version__)
print("GPUS:", tf.config.list_physical_devices('GPU'))
EOF

echo "======================================="
echo "STARTING SEGMENTATION"

# Commands / scripts to run (e.g., python3 train.py)
python test_png_segmentation.py \
    --input_dir /nas-ctm01/homes/mcmendes/samples/denoising_dgan/2 \
    --output_dir /nas-ctm01/homes/mcmendes/outputs/segmentation \
    --weights /nas-ctm01/homes/pfsousa/fedseg/seg_piu/bestmodel.epoch05.hdf5 \
    --num_images 10

