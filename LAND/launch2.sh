#!/bin/bash
#
#SBATCH --partition=gpu_min32gb_ext     # Reserved partition
#SBATCH --qos=gpu_min32gb_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train_st3    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate land

cd "$SLURM_SUBMIT_DIR"

# -------------------- Dados e VAE -------------------- #
DATASET_PATH="/nas-ctm01/datasets/public/lidc-mcmendes/slices"

# Seleciona automaticamente o run mais recente do VAE.
VAE_RUN_DIR=$(ls -td \
  /nas-ctm01/homes/mcmendes/outputs/land/LAND2D_VAE_LIDC_* \
  2>/dev/null | head -n 1)

if [[ -z "${VAE_RUN_DIR:-}" ]]; then
  echo "ERROR: Não foi encontrado nenhum run LAND2D_VAE_LIDC."
  exit 1
fi

VAE_PATH="$VAE_RUN_DIR/vae_best_epoch"

if [[ ! -d "$VAE_PATH" ]]; then
  echo "ERROR: O checkpoint do VAE não existe: $VAE_PATH"
  exit 1
fi

# -------------------- Outputs -------------------- #
OUTPUT_ROOT="/nas-ctm01/homes/mcmendes/outputs/land/ldm"
RUN_NAME="LAND2D_LDM_LIDC"
RUN_ID="${RUN_NAME}_$(date +%Y%m%d-%H%M%S)"

OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
LOG_DIR="$OUTPUT_ROOT/logs/$RUN_ID"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "$OUTPUT_DIR" > "$OUTPUT_ROOT/LATEST_RUN.txt"

echo "============================================"
echo "LAND-2D LDM TRAINING"
echo "Dataset:          $DATASET_PATH"
echo "VAE checkpoint:   $VAE_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Logging directory:$LOG_DIR"
echo "Samples:          20000"
echo "Epochs:           100"
echo "Batch size:       8"
echo "============================================"

# Commands / scripts to run (e.g., python3 train.py)
srun --export=ALL bash scripts/train_ldm_2d.sh \
  "$DATASET_PATH" \
  "$VAE_PATH" \
  "$RUN_ID" \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOG_DIR" \
  --max_samples 20000 \
  --num_epochs 100 \
  --train_batch_size 8 \
  --dataloader_num_workers 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine \
  --lr_warmup_steps 10 \
  --prediction_type v_prediction \
  --ddpm_num_steps 1000 \
  --snr_gamma 5.0 \
  --mixed_precision fp16 \
  --save_model_epochs 5 \
  --save_images_epochs 5 \
  --sample_inference_steps 100 \
  --num_sample_images 4 \
  --seed 42