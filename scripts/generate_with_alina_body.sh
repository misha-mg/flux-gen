#!/usr/bin/env bash
set -euo pipefail

# Convenience script to run generation with the alina-body LoRA.
# Usage: scripts/generate_with_alina_body.sh [additional args passed to CLI]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$PROJECT_ROOT/src/generate.py" \
  --lora_path "$PROJECT_ROOT/lora/alina-body.safetensors" \
  --lora_trigger_word "alina-body" \
  "$@"


