#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext     # Reserved partition
#SBATCH --qos=gpu_min8gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_st2    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

set -eo pipefail

# ============================================================
# Sampling configuration
# Alterar apenas estes três valores quando necessário.
# ============================================================

CHECKPOINT="/nas-ctm01/homes/mcmendes/outputs/stylegan3/00001-stylegan3-t-slices-gpus1-batch8-gamma2-lidc-npy-256/network-snapshot-002000.pkl"

OUT_DIR="/nas-ctm01/homes/mcmendes/samples/stylegan3"

NUM_IMAGES=5

# ============================================================

START_SEED=0
END_SEED=$((NUM_IMAGES - 1))


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

# Move to the StyleGAN3 repository.
cd /nas-ctm01/homes/mcmendes/tese/stylegan3a


if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found:"
    echo "$CHECKPOINT"
    exit 1
fi

if [ "$NUM_IMAGES" -lt 1 ]; then
    echo "ERROR: NUM_IMAGES must be at least 1."
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Checkpoint: $CHECKPOINT"
echo "Output directory: $OUT_DIR"
echo "Number of images: $NUM_IMAGES"

# Commands / scripts to run (e.g., python3 train.py)
python gen_images.py \
  --network="$CHECKPOINT" \
  --outdir="$OUT_DIR" \
  --seeds="${START_SEED}-${END_SEED}" \
  --trunc=1 \
  --noise-mode=const