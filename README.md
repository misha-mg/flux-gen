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

**Для LoRA підтримки (опціонально):**
```bash
pip install peft>=0.7.0
```

3. Запустите генерацию

```bash
python src/generate.py --prompt "ваш промпт здесь"
```

По умолчанию изображения сохраняются в `src/outputs/flux_schnell.png`.

### Використання з референсним фото (reference image)

1. Покладіть файл `png/jpg/webp` у `reference_images/` (наприклад `reference_images/alina.png`).
2. Запустіть:

```bash
python src/generate.py \
  --reference_image alina.png \
  --prompt "portrait photo, studio light, high detail"
```

Це увімкне **IP-Adapter** для FLUX і передасть референс-зображення як `ip_adapter_image`.

> Важливо: для цього потрібна свіжа версія `diffusers` (див. `requirements.txt`). Якщо бачите помилку про відсутній `load_ip_adapter` — оновіть `diffusers`.

### Использование с LoRA

**Важно:** Для использования LoRA требуется установить PEFT библиотеку:
```bash
pip install peft>=0.7.0
```

#### Способ 1: Ручное добавление trigger word в промпт

Если ваша LoRA обучена с trigger word (например, "alina-face"), добавьте его в начало промпта:

```bash
python src/generate.py \
  --prompt "alina-face, ваш промпт здесь, portrait photo, beautiful woman" \
  --lora_path "path/to/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors" \
  --lora_config_path "path/to/rfNiXmg5PsAsGzH4OaKIi_config.json" \
  --lora_scale 1.0
```

#### Способ 2: Автоматическое добавление trigger word

Используйте параметр `--lora_trigger_word` для автоматического добавления:

```bash
python src/generate.py \
  --prompt "portrait photo, beautiful woman, detailed face" \
  --lora_path "path/to/KMvFWS9iTsDBW7gksxGlK_pytorch_lora_weights.safetensors" \
  --lora_trigger_word "alina-face" \
  --lora_scale 1.0
```

В этом случае эффективный промпт будет: `"alina-face, portrait photo, beautiful woman, detailed face"`

Параметры LoRA:
- `--lora_path`: путь к файлу весов LoRA (.safetensors)
- `--lora_config_path`: путь к конфигурационному файлу LoRA (.json), опционально
- `--lora_scale`: масштаб применения LoRA (по умолчанию 1.0)

### Детальная документация

Подробные инструкции по запуску на Runpod, настройке GPU и устранению неисправностей см. в [`src/README_RUNPOD.md`](src/README_RUNPOD.md).
