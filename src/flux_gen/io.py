"""Input/Output operations for FLUX generation."""

import os
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from PIL import Image

from .config import REFERENCE_IMAGES_DIRNAME

def ensure_output_directory(out_dir: Path):
    """Create output directory if it doesn't exist."""
    os.makedirs(out_dir, exist_ok=True)


def save_generated_image(image, output_path: Path):
    """Save the generated image to the specified path."""
    image.save(output_path)
    print(f"Saved: {output_path}")


def upscale_image(image: Image.Image, upscale: float) -> Image.Image:
    """Upscale a PIL image by a factor (>=1.0). Uses Lanczos resampling."""
    if upscale is None or upscale == 1.0:
        return image
    if upscale <= 0:
        raise ValueError(f"upscale must be > 0, got {upscale}")
    w = max(1, int(round(image.width * upscale)))
    h = max(1, int(round(image.height * upscale)))
    return image.resize((w, h), Image.LANCZOS)


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_metadata_json(metadata: dict, json_path: Path):
    """Save metadata dict to JSON next to outputs (pretty-printed)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=_json_safe)
    print(f"Saved metadata: {json_path}")


def config_to_metadata(gen_config) -> dict:
    """Convert GenerationConfig dataclass into JSON-serializable metadata."""
    if is_dataclass(gen_config):
        data = asdict(gen_config)
    else:
        data = dict(gen_config)
    # normalize Path-like values
    for k, v in list(data.items()):
        if isinstance(v, Path):
            data[k] = str(v)
    return data


def resolve_reference_image_path(reference_image: str) -> Path:
    """
    Resolve a reference image path.

    Accepts:
    - absolute path
    - path relative to current working dir
    - filename or relative path inside repo-root/reference_images/
    """
    p = Path(reference_image)
    if p.exists():
        return p

    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / REFERENCE_IMAGES_DIRNAME / reference_image
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Reference image not found: '{reference_image}'. "
        f"Tried '{p}' and '{candidate}'."
    )


def load_reference_image(reference_image: str, *, width: int | None = None, height: int | None = None) -> Image.Image:
    """Load reference image as RGB PIL.Image (optionally resized to width/height)."""
    path = resolve_reference_image_path(reference_image)
    img = Image.open(path).convert("RGB")
    if width and height:
        img = img.resize((width, height))
    return img


def load_init_image(init_image: str) -> Image.Image:
    """Load an existing image (init for img2img) as RGB PIL.Image."""
    path = Path(init_image).expanduser()
    img = Image.open(path).convert("RGB")
    return img
