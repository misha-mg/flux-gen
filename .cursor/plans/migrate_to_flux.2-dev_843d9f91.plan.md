---
name: Migrate to FLUX.2-dev
overview: Switch default model from FLUX.1-dev to FLUX.2-dev, update diffusers/compat layer, and keep existing CLI/LoRA flow while attempting IP-Adapter support when available (with clear fallback/errors).
todos:
  - id: deps-flux2
    content: "Оновити `requirements.txt`: прибрати `diffusers<0.40.0`, підібрати мінімальну версію diffusers з Flux2 pipeline і вирівняти transformers/accelerate за потреби."
    status: pending
  - id: model-id-default
    content: Змінити дефолт `MODEL_ID` на `black-forest-labs/FLUX.2-dev` і оновити README/README_RUNPOD приклади.
    status: pending
    dependencies:
      - deps-flux2
  - id: pipeline-router
    content: Додати в `src/flux_gen/pipeline.py` роутинг Flux1 vs Flux2 (вибір pipeline-класу), підтримку `caption_upsample_temperature`, і чіткі помилки/фолбеки для IP-Adapter та LoRA на Flux2.
    status: pending
    dependencies:
      - deps-flux2
  - id: cli-config-flux2
    content: Додати `caption_upsample_temperature` в `GenerationConfig` та прапорець у CLI, прокинути в генерацію.
    status: pending
    dependencies:
      - pipeline-router
  - id: cleanup-invalid-files
    content: Прибрати shell-команди з `.gitignore` і `src/generate.py` (перенести приклади в README або .sh).
    status: pending
  - id: tests-update
    content: Оновити/додати тести для нового Flux2 branch у pipeline loader і перевірити smoke-cases (без ref, з ref, з LoRA).
    status: pending
    dependencies:
      - pipeline-router
      - cli-config-flux2
---

# План переходу на FLUX.2 [dev]

### 1) Поточний стан (що в нас зараз)

- **Модель задається тільки через `model_id`**: дефолт у [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/config.py`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/config.py) (`MODEL_ID = ... "black-forest-labs/FLUX.1-dev"`).
- **Завантаження пайплайна** робиться у [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/pipeline.py`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/pipeline.py) через `FluxPipeline.from_pretrained(gen_config.model_id, ...)`.
- **`--reference_image` зараз = IP-Adapter для Flux1** (`pipe.load_ip_adapter(...)` + `ip_adapter_image` в `pipe(**kwargs)`).
- **LoRA**: застосовується через `pipe.load_lora_weights(...); pipe.fuse_lora(...)`.
- **Критичні “побічні” баги**:
- `.gitignore` містить шматок shell-команди і ламає `rg`/пошук у репо (у вас це вже проявилося).
- `src/generate.py` містить shell-команду в кінці файлу (це робить файл невалідним Python).

### 2) Цільова модель і ризики сумісності

- За документацією diffusers, **офіційний repo_id для Flux2**: **`black-forest-labs/FLUX.2-dev`** (див. `Flux2Transformer2DModel` docs у diffusers).
- Flux2 — “new architecture”, тому **майже напевно потрібен новий pipeline-клас** у diffusers (не `FluxPipeline`).
- Ви хочете:
- **зберегти IP-Adapter підхід** (`--reference_image`) *якщо можливо*.
- **спробувати залишити поточні LoRA** (або дати чітку помилку/фолбек, якщо несумісні).

### 3) План робіт по коду (мінімально інвазивно)

#### 3.1 Оновити дефолтну модель і документацію

- Змінити `MODEL_ID` на **`black-forest-labs/FLUX.2-dev`** у:
- [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/config.py`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/config.py)
- Оновити згадки в:
- [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/README.md`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/README.md)
- [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/src/README_RUNPOD.md`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/src/README_RUNPOD.md)

#### 3.2 Оновити залежності під Flux2

- У [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/requirements.txt`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/requirements.txt):
- **прибрати/послабити верхню межу** `diffusers<0.40.0` (Flux2 підтримка з’явилась пізніше).
- зафіксувати мінімум (після швидкої перевірки) як **`diffusers>=X`**, де `X` — перша версія, в якій імпортується Flux2 pipeline (ми визначимо це простим smoke-check’ом: `python -c "from diffusers import Flux2Pipeline"` / `DiffusionPipeline.from_pretrained('black-forest-labs/FLUX.2-dev')`).
- за потреби підняти `transformers/accelerate` до сумісних версій (якщо diffusers цього вимагатиме).

#### 3.3 Додати “router” пайплайна: Flux1 vs Flux2

- У [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/pipeline.py`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/pipeline.py):
- Додати логіку вибору pipeline-класу:
    - Якщо `model_id` містить `FLUX.2` → намагатися завантажити **Flux2 pipeline** (наприклад `Flux2Pipeline` або `DiffusionPipeline` з Flux2 конфігом; залежить від API у вашій версії diffusers).
    - Інакше → залишити поточний `FluxPipeline` для Flux1.
- Уніфікувати виклик `pipe(**pipe_kwargs)` так, щоб працював і для Flux2.
- Додати опціональні Flux2-аргументи (мінімум):
    - `caption_upsample_temperature` (із дефолтом `None`), бо diffusers docs для Flux2 це згадують.

#### 3.4 IP-Adapter на Flux2 “якщо можливо”

- У тому ж [`pipeline.py`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/pipeline.py):
- Залишити поточну спробу `pipe.load_ip_adapter(...)`, але зробити **більш явну сумісність**:
    - Якщо `--reference_image` заданий, а у вибраного пайплайна немає `load_ip_adapter`, кидати **чіткий RuntimeError**: “IP-Adapter не підтримується для цього пайплайна/версії diffusers; або оновіть diffusers, або використовуйте Flux1”.
- Це дасть вам бажану поведінку: “зберегти IP-Adapter якщо можливо”, але не ламати генерацію тихо.

#### 3.5 LoRA

- Залишити існуючий механізм LoRA (через `load_lora_weights`/`fuse_lora`) і додати:
- якщо для Flux2 pipeline ці методи відсутні → **чіткий меседж** про несумісність LoRA з цим пайплайном/моделлю.
- (опційно) додати “skip” режим: якщо LoRA несумісна — логувати warning і продовжувати без LoRA.

#### 3.6 CLI/Config під Flux2-опції

- У [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/config.py`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/config.py):
- Додати поле `caption_upsample_temperature: float | None = None`.
- У [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/cli.py`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/src/flux_gen/cli.py):
- Додати `--caption_upsample_temperature` (optional), прокинути в `GenerationConfig`.

### 4) Прибирання блокерів (обов’язково, бо ламає запуск/пошук)

- Почистити `.gitignore`: прибрати рядки з shell-командою (їм місце в README або в `.sh`).
- Почистити `src/generate.py`: прибрати shell-команду з кінця файлу.
- Ідеально: перенести цей приклад у `README.md` або окремий `example_*.sh` (у вас вже є `example_lora_usage.sh`).

### 5) Тести/смоук-перевірки

- Оновити unit-тести, які мокають `FluxPipeline.from_pretrained`, щоб вони також покривали Flux2 branch (новий pipeline-клас/роутер).
- Файл: [`/Users/mishahorodnytskyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/IT/AI-GEN/alina-base/tests/test_pipeline.py`](/Users/mishahorodnytskyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Documents/IT/AI-GEN/alina-base/tests/test_pipeline.py)
- Додати простий smoke-test сценарій (локально/Runpod):
- без `--reference_image` (щоб підтвердити базовий T2I)
- з `--reference_image` (перевірити, чи підтримує Flux2 `load_ip_adapter`; якщо ні — отримати очікувану “чітку помилку”)