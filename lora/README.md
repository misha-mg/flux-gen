# LoRA Models Directory

Поместите сюда ваши LoRA файлы для использования с FLUX моделью.

**Важно:** Для использования LoRA требуется установить PEFT библиотеку:
```bash
pip install peft>=0.7.0
```

## Структура файлов LoRA

Для каждой LoRA модели обычно нужны два файла:
1. `*_pytorch_lora_weights.safetensors` - веса модели
2. `*_config.json` - конфигурация LoRA

## Примеры использования

```bash
# С вашими файлами
python src/generate.py \
  --prompt "portrait photo, detailed face" \
  --lora_path "lora/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors" \
  --lora_config_path "lora/rfNiXmg5PsAsGzH4OaKIi_config.json"

# С кастомным масштабом
python src/generate.py \
  --prompt "artistic portrait" \
  --lora_path "lora/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors" \
  --lora_scale 0.8
```

## Советы по использованию LoRA

- **Масштаб (scale)**: Обычно 0.5-1.5. Начните с 1.0
- **Совместимость**: Убедитесь, что LoRA обучена на FLUX или совместимой архитектуре
- **Качество**: Высококачественные LoRA могут значительно улучшить результат генерации
