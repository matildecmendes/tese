#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext     # Reserved partition
#SBATCH --qos=gpu_min11gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=s_land  # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate land

cd "$SLURM_SUBMIT_DIR"

# -------------------- Modelo LDM -------------------- #
LDM_RUN=$(cat /nas-ctm01/homes/mcmendes/outputs/land/ldm/LATEST_RUN.txt)

if [[ -z "${LDM_RUN:-}" ]]; then
  echo "ERROR: Não foi encontrado o LDM_RUN."
  exit 1
fi

if [[ ! -d "$LDM_RUN/unet" ]]; then
  echo "ERROR: Não existe a pasta do UNet: $LDM_RUN/unet"
  exit 1
fi

if [[ ! -d "$LDM_RUN/scheduler" ]]; then
  echo "ERROR: Não existe a pasta do scheduler: $LDM_RUN/scheduler"
  exit 1
fi

# -------------------- VAE -------------------- #
VAE_RUN_DIR=$(ls -td /nas-ctm01/homes/mcmendes/outputs/land/LAND2D_VAE_LIDC_* 2>/dev/null | head -n 1)

if [[ -z "${VAE_RUN_DIR:-}" ]]; then
  echo "ERROR: Não foi encontrado nenhum run do VAE."
  exit 1
fi

VAE_PATH="$VAE_RUN_DIR/vae_best_epoch"

if [[ ! -d "$VAE_PATH" ]]; then
  echo "ERROR: Não existe o checkpoint do VAE: $VAE_PATH"
  exit 1
fi

# --------------------------------------------------------------- #
OUTPUT_DIR="/nas-ctm01/homes/mcmendes/samples/land"
NUM_SAMPLES=5000
BATCH_SIZE=8
STEPS=250
SEED=42

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "LAND-2D SAMPLE GENERATION"
echo "LDM run:              $LDM_RUN"
echo "VAE checkpoint:       $VAE_PATH"
echo "Output directory:     $OUTPUT_DIR"
echo "Number of samples:    $NUM_SAMPLES"
echo "Batch size:           $BATCH_SIZE"
echo "Inference steps:      $STEPS"
echo "Seed:                 $SEED"
echo "============================================"

# Commands / scripts to run (e.g., python3 train.py)
srun --export=ALL bash scripts/inference_ldm_2d.sh \
  "$LDM_RUN" \
  "$VAE_PATH" \
  "$OUTPUT_DIR" \
  "$NUM_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --seed "$SEED"