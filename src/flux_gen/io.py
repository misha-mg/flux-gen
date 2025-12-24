"""Input/Output operations for FLUX generation."""

import os
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
