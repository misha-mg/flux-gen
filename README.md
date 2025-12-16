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

### Детальная документация

Подробные инструкции по запуску на Runpod, настройке GPU и устранению неисправностей см. в [`src/README_RUNPOD.md`](src/README_RUNPOD.md).
