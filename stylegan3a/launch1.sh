#!/bin/bash
#
#SBATCH --partition=gpu_min32gb_ext     # Reserved partition
#SBATCH --qos=gpu_min32gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_st3    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

set -eo pipefail

# Paths and training configuration


CHECKPOINT="/nas-ctm01/homes/mcmendes/outputs/stylegan3/00003-stylegan3-t-slices-gpus1-batch32-gamma2-lidc-test-fp32-batch32/network-snapshot-000300.pkl"

OUT_DIR="/nas-ctm01/homes/mcmendes/outputs/stylegan3"

DATA_DIR="/nas-ctm01/datasets/public/lidc-mcmendes/slices"


TOTAL_KIMG=2000

RESUME_KIMG=300


# Conda environment

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stylegan3

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export NVCC_CCBIN="$CXX"
export NVCC_PREPEND_FLAGS="-ccbin=$CXX"

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

# Validation


if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found:"
    echo "$CHECKPOINT"
    exit 1
fi

echo "========================================"
echo "Resume checkpoint: $CHECKPOINT"
echo "Resume kimg:        $RESUME_KIMG"
echo "Final target:       $TOTAL_KIMG"
echo "Additional kimg:    $((TOTAL_KIMG - RESUME_KIMG))"
echo "Snapshots:          every 100 kimg"
echo "========================================"

cd /nas-ctm01/homes/mcmendes/tese/stylegan3a

# Training


python train.py \
  --outdir="$OUT_DIR" \
  --data="$DATA_DIR" \
  --cfg=stylegan3-t \
  --gpus=1 \
  --batch=32 \
  --batch-gpu=4 \
  --gamma=2 \
  --aug=noaug \
  --metrics=none \
  --fp32=1 \
  --kimg="$TOTAL_KIMG" \
  --tick=4 \
  --snap=25 \
  --workers=4 \
  --seed=0 \
  --desc=lidc-fp32-batch32-resume300 \
  --resume="$CHECKPOINT" \
  --resume-kimg="$RESUME_KIMG"