#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=test_ddpm    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

cd /nas-ctm01/homes/mcmendes/tese/ddpm

source ~/.bashrc
conda activate ddpm_env

export PYTHONPATH=$PWD:$PYTHONPATH

echo "========== NVIDIA SMI =========="
nvidia-smi

echo "========== TF GPU CHECK =========="
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPU device:', tf.test.gpu_device_name())"

python scripts/sample_lungct.py \
    --checkpoint_dir /nas-ctm01/homes/mcmendes/outputs/ddpm/565000 \
    --output_dir /nas-ctm01/homes/mcmendes/samples/ddpm \
    --num_samples 2633