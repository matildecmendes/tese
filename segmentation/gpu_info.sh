#!/bin/bash

#SBATCH --partition=gpu_min8gb
#SBATCH --job-name=gpuinfo
#SBATCH --output=gpuinfo.out
#SBATCH --error=gpuinfo.err
#SBATCH --time=00:05:00

nvidia-smi

python - << EOF
import tensorflow as tf
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
EOF