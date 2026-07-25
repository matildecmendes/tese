#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/inference_ldm_2d.sh VAE_PATH MODEL_PATH OUTPUT_DIR [NUM_SAMPLES] [STEPS] [BATCH_SIZE] [extra Python arguments]"
  exit 1
fi

VAE_PATH="$1"
MODEL_PATH="$2"
OUTPUT_DIR="$3"
NUM_SAMPLES="${4:-16}"
STEPS="${5:-1000}"
BATCH_SIZE="${6:-8}"
if [[ $# -ge 6 ]]; then shift 6; else shift $#; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$(dirname "$REPO_ROOT"):${PYTHONPATH:-}"

python -B "$REPO_ROOT/src/land2d/inference_ldm_2d.py" \
  --vae_path "$VAE_PATH" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --resolution 256 \
  "$@"
