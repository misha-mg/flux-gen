## alina-base

Проект для генерации изображений с помощью модели FLUX.

### Структура
- `src/`: вихідний код
- `src/outputs/`: результати/артефакти (генерації, експорти тощо)
- `tests/`: unit-тесты

### Швидкий старт

1. (Опційно) створіть venv

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Встановіть залежності

```bash
pip install -r requirements.txt
```

3. Запустите генерацию

```bash
python src/generate.py --prompt "ваш промпт здесь"
```

По умолчанию изображения сохраняются в `src/outputs/flux_schnell.png`.

### Использование с LoRA

Для использования собственной LoRA модели:

```bash
python src/generate.py \
  --prompt "ваш промпт здесь" \
  --lora_path "path/to/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors" \
  --lora_config_path "path/to/rfNiXmg5PsAsGzH4OaKIi_config.json" \
  --lora_scale 1.0
```

Параметры LoRA:
- `--lora_path`: путь к файлу весов LoRA (.safetensors)
- `--lora_config_path`: путь к конфигурационному файлу LoRA (.json), опционально
- `--lora_scale`: масштаб применения LoRA (по умолчанию 1.0)

### Детальная документация

Подробные инструкции по запуску на Runpod, настройке GPU и устранению неисправностей см. в [`src/README_RUNPOD.md`](src/README_RUNPOD.md).
