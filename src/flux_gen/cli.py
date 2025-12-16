"""Command line interface for FLUX generation."""

import argparse
import os
from pathlib import Path

from .config import GenerationConfig


MODEL_ID = "black-forest-labs/FLUX.1-schnell"


def parse_args():
    """Parse command line arguments and return GenerationConfig."""
    parser = argparse.ArgumentParser(description="Generate images using FLUX model on Runpod")
    parser.add_argument(
        "--model_id",
        type=str,
        default=MODEL_ID,
        help=f"Model ID to use (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="cinematic portrait photo, soft natural light, 85mm lens, shallow depth of field, ultra realistic",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("src", "outputs"),
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Image height (default: 768)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Image width (default: 768)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for generation (default: 3.5)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps (default: 20)"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights file (.safetensors)"
    )
    parser.add_argument(
        "--lora_config_path",
        type=str,
        default=None,
        help="Path to LoRA config file (.json)"
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Scale factor for LoRA weights (default: 1.0)"
    )

    args = parser.parse_args()

    # Check PEFT availability if LoRA is requested
    if args.lora_path:
        try:
            import peft
        except ImportError:
            print("Warning: PEFT library is required for LoRA support but not installed.")
            print("Install it with: pip install peft>=0.7.0")
            print("Continuing without LoRA...")

    return GenerationConfig(
        model_id=args.model_id,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        out_dir=Path(args.out_dir),
        lora_path=args.lora_path,
        lora_config_path=args.lora_config_path,
        lora_scale=args.lora_scale,
    )
