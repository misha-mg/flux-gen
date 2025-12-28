#!/bin/bash
# Приклади використання LoRA з негативним промптом, референс фото і різними силами LoRA
# Важливо: переконайтесь, що залежності встановлені (наприклад, PEFT)

# Шлях до папки референсних зображень (покладіть зображення у reference_images/)
REFERENCE_DIR="reference_images"
REFERENCE_IMAGE="ref1.png" # приклад: reference_images/ref1.png

# Типові шляхи до LoRA (оновіть під свої файли)
LORA_BODY="lora/alina-body.safetensors"
LORA_FACE="lora/alina-face.safetensors"

echo "Example 1 — Reference + two LoRA models + negative prompt"
python3 src/generate.py \
  --reference_image "${REFERENCE_IMAGE}" \
  --ip_adapter_scale 0.5 \
  --lora_paths "${LORA_BODY}" \
  --lora_paths "${LORA_FACE}" \
  --lora_trigger_words "alina-body" \
  --lora_trigger_words "alina-face" \
  --lora_scales 1.0 \
  --lora_scales 1.0 \
  --negative_prompt "low quality, blurry, out of focus, artifacts, watermark, text, extra limbs, mutated hands, bad anatomy, unnatural skin, oversaturated, deformed face, explicit nudity" \
  --prompt "photorealistic mirror selfie, modern bathroom, adult blonde with short layered bob, natural makeup, soft smile, tight white cropped tee, pink underwear, slim athletic pose (hip forward), holding smartphone covering face, bathtub & toiletries, speckled stone wall, soft natural light, realistic skin, high detail, shallow DOF, 35mm, sharp focus" \
  --height 1024 \
  --width 1024 \
  --guidance_scale 3.5 \
  --num_inference_steps 28

echo "Example 2 — Stronger face LoRA, weaker body LoRA"
python3 src/generate.py \
  --reference_image "${REFERENCE_IMAGE}" \
  --ip_adapter_scale 0.6 \
  --lora_paths "${LORA_BODY}" \
  --lora_paths "${LORA_FACE}" \
  --lora_trigger_words "alina-body" \
  --lora_trigger_words "alina-face" \
  --lora_scales 0.6 \
  --lora_scales 1.2 \
  --negative_prompt "bad anatomy, extra limbs, deformed, lowres, watermark, text" \
  --prompt "studio portrait, soft natural light, candid smile, realistic skin texture, detailed eyes, 85mm" \
  --height 896 \
  --width 1152 \
  --guidance_scale 4.0 \
  --num_inference_steps 30

echo "Example 3 — Single LoRA (face) with lower influence and stricter negative prompt"
python3 src/generate.py \
  --reference_image "${REFERENCE_IMAGE}" \
  --ip_adapter_scale 0.4 \
  --lora_paths "${LORA_FACE}" \
  --lora_trigger_words "alina-face" \
  --lora_scales 0.8 \
  --negative_prompt "low quality, blurry, artifacts, watermark, text, mutated hands, bad anatomy, deformed face" \
  --prompt "clean headshot, neutral background, natural makeup, high detail, realistic skin" \
  --height 768 \
  --width 768 \
  --guidance_scale 3.0 \
  --num_inference_steps 25

echo "Примітка: змінюйте `--lora_scales` щоб контролювати вплив кожного LoRA; повторюйте `--lora_paths`/`--lora_scales` у потрібній кількості."

python3 src/generate.py \
  --reference_image "src/reference_images/ref1.png" \
  --ip_adapter_scale 0.6 \
  --lora_paths "lora/alina-body.safetensors" \
  --lora_paths "lora/alina-face.safetensors" \
  --lora_trigger_words "alina-body" \
  --lora_trigger_words "alina-face" \
  --lora_scales 0.6 \
  --lora_scales 1.2 \
  --negative_prompt "bad anatomy, extra limbs, deformed, lowres, watermark, text" \
  --prompt "studio portrait, soft natural light, candid smile, realistic skin texture, detailed eyes, 85mm" \
  --height 896 \
  --width 1152 \
  --guidance_scale 4.0 \
  --num_inference_steps 30


  python3 src/generate.py \
  --reference_image "reference_images/example.png" \
  --ip_adapter_scale 0.9 \
  --lora_paths "lora/alina-body.safetensors" \
  --lora_paths "lora/alina-face.safetensors" \
  --lora_trigger_words "alina-body" \
  --lora_trigger_words "alina-face" \
  --lora_scales 0.6 \
  --lora_scales 0.6 \
  --negative_prompt "bad anatomy, extra limbs, deformed, lowres, watermark, text" \
  --prompt "adult blonde woman on a bicycle, leaning forward in a black thong and oversized striped shirt, turned back smiling over her shoulder, sunlit tropical road, glistening skin, playful seductive pose emphasizing hips and buttocks, high contrast daylight, shallow depth of field, 85mm, photorealistic" \
  --height 1024 \
  --width 1024 \
  --guidance_scale 4.0 \
  --num_inference_steps 30