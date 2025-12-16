"""Input/Output operations for FLUX generation."""

import os
from pathlib import Path


def ensure_output_directory(out_dir: Path):
    """Create output directory if it doesn't exist."""
    os.makedirs(out_dir, exist_ok=True)


def save_generated_image(image, output_path: Path):
    """Save the generated image to the specified path."""
    image.save(output_path)
    print(f"Saved: {output_path}")
