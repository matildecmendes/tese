#!/bin/bash
#
#SBATCH --partition=gpu_min32gb_ext     # Reserved partition
#SBATCH --qos=gpu_min32gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_st3    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stylegan3

# CUDA toolkit inside the Conda environment.
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

# Use GCC 9 and the Conda Linux sysroot.
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export NVCC_CCBIN="$CXX"
export NVCC_PREPEND_FLAGS="-ccbin=$CXX"

# Separate cache for this compiler configuration.
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions_stylegan3_cuda111_gcc9"

echo "CUDA compiler:"
which nvcc
nvcc --version

echo "Host compiler:"
"$CC" --version
"$CXX" --version

echo "CC=$CC"
echo "CXX=$CXX"
echo "NVCC_CCBIN=$NVCC_CCBIN"

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