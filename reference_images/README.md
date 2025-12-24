## reference_images

Це директорія для **референсних зображень** (png/jpg/webp), які можна використовувати як приклад у CLI генерації.

### Як використовувати

1. Покладіть файл (наприклад `alina.png`) у цю директорію.
2. Запускайте генерацію з параметром `--reference_image`:

```bash
python src/generate.py --reference_image alina.png --prompt "portrait photo, studio light"
```

Або вкажіть абсолютний/відносний шлях:

```bash
python src/generate.py --reference_image /abs/path/to/alina.png --prompt "portrait photo"
```

### Важливо

- Репозиторій за замовчуванням ігнорує `*.png` (див. `.gitignore`), тому ваші фото **не будуть комітитися**.
- Якщо `--reference_image` заданий, генерація використовує **IP-Adapter** для FLUX (conditioning по зображенню).


