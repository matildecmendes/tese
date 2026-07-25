#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/train_ldm_2d.sh DATASET_PATH VAE_PATH RUN_NAME [extra Python arguments]"
  echo "Example smoke test: bash scripts/train_ldm_2d.sh /data/LIDC /checkpoints/vae_best_epoch LAND2D_test --max_samples 100 --num_epochs 1 --max_train_steps 10 --train_batch_size 2"
  exit 1
fi

DATASET_PATH="$1"
VAE_PATH="$2"
RUN_NAME="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$(dirname "$REPO_ROOT"):${PYTHONPATH:-}"

python -B "$REPO_ROOT/src/land2d/train_ldm_2d.py" \
  --dataset_path "$DATASET_PATH" \
  --vae_path "$VAE_PATH" \
  --output_dir "$REPO_ROOT/checkpoints/ldm_2d/$RUN_NAME" \
  --logging_dir "$REPO_ROOT/logs/ldm_2d/$RUN_NAME" \
  --resolution 256 \
  --num_epochs 500 \
  --train_batch_size 8 \
  --learning_rate 1e-5 \
  --prediction_type v_prediction \
  --ddpm_num_steps 1000 \
  --snr_gamma 5.0 \
  --save_model_epochs 10 \
  --save_images_epochs 25 \
  --sample_inference_steps 100 \
  --mixed_precision no \
  --attention \
  "$@"
