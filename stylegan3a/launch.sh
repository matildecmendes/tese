#!/bin/bash
#
#SBATCH --partition=gpu_min32gb_ext     # Reserved partition
#SBATCH --qos=gpu_min32gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_st3    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stylegan3

# Force StyleGAN3 to use the CUDA toolkit installed inside Conda.
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

# Use a clean, model-specific extension cache.
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions_stylegan3_cuda111"

hash -r

echo "========================================"
echo "Environment verification"
echo "========================================"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Conda prefix:      $CONDA_PREFIX"
echo "Python:            $(which python)"
echo "NVCC:              $(which nvcc)"
nvcc --version

python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA_HOME detected by PyTorch:", CUDA_HOME)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# Stop immediately if the system CUDA is still being used.
NVCC_PATH="$(which nvcc)"

if [[ "$NVCC_PATH" != "$CONDA_PREFIX/bin/nvcc" ]]; then
    echo "ERROR: Wrong NVCC selected: $NVCC_PATH"
    echo "Expected: $CONDA_PREFIX/bin/nvcc"
    exit 1
fi

echo "========================================"

# Commands / scripts to run (e.g., python3 train.py)
python train.py \
  --outdir /nas-ctm01/homes/mcmendes/outputs/stylegan3 \
  --data /nas-ctm01/datasets/public/lidc-mcmendes/slices \
  --cfg=stylegan3-t \
  --gpus=1 \
  --batch=8 \
  --batch-gpu=4 \
  --gamma=2 \
  --aug=noaug \
  --metrics=none \
  --kimg=1000 \
  --tick=4 \
  --snap=25 \
  --workers=4 \
  --seed=0 \
  --desc=lidc-npy-256 