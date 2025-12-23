#!/bin/bash
# Пример использования LoRA с вашими файлами
# Важно: установите PEFT для поддержки LoRA: pip install peft>=0.7.0

# Предполагая, что файлы LoRA находятся в директории lora/
LORA_WEIGHTS="lora/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors"
LORA_CONFIG="lora/rfNiXmg5PsAsGzH4OaKIi_config.json"

# Базовый пример с LoRA (ВАЖНО: используйте trigger word "alina-face"!)
python src/generate.py \
  --prompt "alina-face, portrait photo, beautiful woman, detailed face, professional lighting" \
  --lora_path "$LORA_WEIGHTS" \
  --lora_config_path "$LORA_CONFIG" \
  --lora_scale 1.0

# Пример с кастомным масштабом LoRA
python src/generate.py \
  --prompt "alina-face, artistic portrait, elegant pose, studio lighting" \
  --lora_path "$LORA_WEIGHTS" \
  --lora_scale 0.8 \
  --height 1024 \
  --width 1024 \
  --guidance_scale 4.0

# Пример без config файла (если он не нужен)
python src/generate.py \
  --prompt "alina-face, cinematic scene, dramatic lighting" \
  --lora_path "$LORA_WEIGHTS" \
  --lora_scale 1.2

# НОВЫЙ СПОСОБ: автоматическое добавление trigger word
python src/generate.py \
  --prompt "cinematic scene, dramatic lighting, professional photo" \
  --lora_path "$LORA_WEIGHTS" \
  --lora_trigger_word "alina-face" \
  --lora_scale 1.2

echo "Примечание: trigger word 'alina-face' будет автоматически добавлен в начало промпта"

# 🎨 РЕАЛІСТИЧНИЙ ПОРТРЕТ: Жінка в чорному топі (робочий варіант)
python src/generate.py \
  --prompt "alina-face, blonde woman, black glasses, high bun, black off-shoulder top with neck tie, white background, direct gaze, elegant, studio portrait, film photography, kodak portra, dslr, 85mm lens, f/2.8, professional headshot, realistic skin texture" \
  --lora_path "lora/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors" \
  --lora_scale 0.6 \
  --height 896 \
  --width 1152 \
  --guidance_scale 3.5 \
  --num_inference_steps 30 \
  --out_dir "src/outputs/studio_portrait_realistic"

# 🔥 НОВИЙ ПРОМПТ: Сексуальна жінка в чорному комбінезоні
# 📝 Оригінал: ~180 слів → Скорочено до ~45 слів для CLIP (77 токенів макс)
# 🎨 Ключові елементи: чорне волосся, комбінезон, вінтажна Lomography естетика
# Варіант 6: Вінтажний Lomography стиль (скорочений з довгого опису)
# ➡️ Повний набір Lomography елементів: зерно, faded film, vignette, Kodachrome
python src/generate.py \
  --prompt "alina-face, sexy woman long sleek straight black hair bangs, unzipped black jumpsuit cleavage, dynamic pose sitting, ambient backlighting, indoor neutral colors, decorative lighting, grainy vintage 35mm photo, faded film ISO 200, vignette stain, Lomography Kodachrome style, impressionistic painting" \
  --lora_path "lora/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors" \
  --lora_scale 0.6 \
  --height 896 \
  --width 1152 \
  --guidance_scale 3.8 \
  --num_inference_steps 32 \
  --out_dir "src/outputs/vintage_lomo_sexy"

# Варіант 6A: Чистий вінтажний фото-стиль (мінімальніший промпт)
# ➡️ Спрощений варіант, якщо виникають проблеми з токенами CLIP
python src/generate.py \
  --prompt "alina-face, sexy woman, unzipped black jumpsuit cleavage, sitting dynamic pose, backlighting, indoor setting, grainy 35mm vintage photo, faded film look, Lomography style, vignette, impressionistic" \
  --lora_path "lora/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors" \
  --lora_scale 0.5 \
  --height 768 \
  --width 1024 \
  --guidance_scale 3.5 \
  --num_inference_steps 28 \
  --out_dir "src/outputs/vintage_lomo_clean"

# Варіант 6B: З додатковими деталями освітлення
# ➡️ Підвищений lora_scale (0.7) для сильнішого впливу + більше кроків для деталей
python src/generate.py \
  --prompt "alina-face, sexy woman long black hair bangs, unzipped jumpsuit showing cleavage, sitting pose ambient backlighting depth, neutral indoor flat colors decorative lights, vintage 35mm grainy photo faded film ISO 200 vignette slight stain, Kodachrome Lomography impressionistic detailed painting" \
  --lora_path "lora/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors" \
  --lora_scale 0.7 \
  --height 1024 \
  --width 1024 \
  --guidance_scale 4.0 \
  --num_inference_steps 35 \
  --out_dir "src/outputs/vintage_lomo_detailed"