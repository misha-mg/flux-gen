#!/bin/bash
# Пример использования LoRA с вашими файлами

# Предполагая, что файлы LoRA находятся в директории lora/
LORA_WEIGHTS="lora/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors"
LORA_CONFIG="lora/rfNiXmg5PsAsGzH4OaKIi_config.json"

# Базовый пример с LoRA
python src/generate.py \
  --prompt "portrait photo, beautiful woman, detailed face, professional lighting" \
  --lora_path "$LORA_WEIGHTS" \
  --lora_config_path "$LORA_CONFIG" \
  --lora_scale 1.0

# Пример с кастомным масштабом LoRA
python src/generate.py \
  --prompt "artistic portrait, elegant pose, studio lighting" \
  --lora_path "$LORA_WEIGHTS" \
  --lora_scale 0.8 \
  --height 1024 \
  --width 1024 \
  --guidance_scale 4.0

# Пример без config файла (если он не нужен)
python src/generate.py \
  --prompt "cinematic scene, dramatic lighting" \
  --lora_path "$LORA_WEIGHTS" \
  --lora_scale 1.2
