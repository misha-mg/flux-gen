#!/usr/bin/env bash
set -euo pipefail

# Convenience script to run generation with both alina-body and alina-face LoRAs.
# Usage: bash scripts/generate_with_alina_both.sh --prompt "..." [other args]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BODY="$PROJECT_ROOT/lora/alina-body.safetensors"
FACE="$PROJECT_ROOT/lora/alina-face.safetensors"

if [ ! -f "$BODY" ]; then
  echo "Missing LoRA weights: $BODY" >&2
  exit 2
fi
if [ ! -f "$FACE" ]; then
  echo "Missing LoRA weights: $FACE" >&2
  exit 2
fi

python3 "$PROJECT_ROOT/src/generate.py" \
  --prompt "female body, slim physique, toned abdomen, sculpted hips; soft facial features, large eyes, defined brows, natural makeup" \
  --lora_paths "$BODY" \
  --lora_paths "$FACE" \
  --lora_trigger_words "alina-body" \
  --lora_trigger_words "alina-face" \
  --lora_scales 1.0 \
  --lora_scales 1.0 \
  "$@"


