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

# Se quiseres passar argumentos ao sbatch:
# sbatch launch_inference_ldm_2d.sh /caminho/output 5000 8 250 42 0
#
# Ordem:
#   $1 = OUTPUT_DIR
#   $2 = NUM_SAMPLES
#   $3 = BATCH_SIZE
#   $4 = STEPS
#   $5 = SEED
#   $6 = START_INDEX

# -------------------- Modelo LDM -------------------- #
LDM_RUN="/nas-ctm01/homes/mcmendes/outputs/land/nlst/ldm/LAND2D_LDM_NLST_20260801-094257/epoch_ckpts/epoch-0100"
LDM_CKPT_NAME="epoch-0100"

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
VAE_RUN_DIR="/nas-ctm01/homes/mcmendes/outputs/land/nlst/LAND2D_VAE_LIDC_20260731-102332"
VAE_PATH="$VAE_RUN_DIR/vae_best_epoch"

if [[ ! -d "$VAE_PATH" ]]; then
  echo "ERROR: Não existe o checkpoint do VAE: $VAE_PATH"
  exit 1
fi

# --------------------------------------------------------------- #
OUTPUT_DIR="${1:-/nas-ctm01/homes/mcmendes/samples/land_256}"
NUM_SAMPLES="${2:-5000}"
BATCH_SIZE="${3:-8}"
STEPS="${4:-250}"
SEED="${5:-42}"
START_INDEX="${6:-0}"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "LAND-2D SAMPLE GENERATION"
echo "LDM run:              $LDM_RUN"
echo "LDM checkpoint:       $LDM_CKPT_NAME"
echo "VAE checkpoint:       $VAE_PATH"
echo "Output directory:     $OUTPUT_DIR"
echo "Number of samples:    $NUM_SAMPLES"
echo "Batch size:           $BATCH_SIZE"
echo "Inference steps:      $STEPS"
echo "Seed:                 $SEED"
echo "Start index:          $START_INDEX"
echo "============================================"

# Necessário para imports do repositório
export PYTHONPATH="$(dirname "$SLURM_SUBMIT_DIR"):${PYTHONPATH:-}"

# Commands / scripts to run (e.g., python3 train.py)
srun --export=ALL python -B src/land2d/inference_ldm_2d.py \
  --vae_path "$VAE_PATH" \
  --model_path "$LDM_RUN" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --resolution 256 \
  --seed "$SEED" \
  --start_index "$START_INDEX" \
  --save_png

