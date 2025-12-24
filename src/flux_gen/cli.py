"""Command line interface for FLUX generation."""

import argparse
import os
from pathlib import Path

from .config import GenerationConfig, MODEL_ID


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
        "--reference_image",
        type=str,
        default=None,
        help="Reference image for conditioning (path or filename inside reference_images/). Enables IP-Adapter when set."
    )
    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default=1.0,
        help="IP-Adapter scale (default: 1.0). Only used when --reference_image is set."
    )
    parser.add_argument(
        "--ip_adapter_repo",
        type=str,
        default="XLabs-AI/flux-ip-adapter",
        help="Hugging Face repo_id for FLUX IP-Adapter (default: XLabs-AI/flux-ip-adapter). Only used when --reference_image is set."
    )
    parser.add_argument(
        "--ip_adapter_weight_name",
        type=str,
        default="ip_adapter.safetensors",
        help="IP-Adapter weights filename inside the repo (default: ip_adapter.safetensors). Only used when --reference_image is set."
    )
    parser.add_argument(
        "--ip_adapter_image_encoder",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Image encoder model for IP-Adapter (default: openai/clip-vit-large-patch14). Only used when --reference_image is set."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt to discourage undesired content (optional)"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights file (.safetensors)"
    )
    parser.add_argument(
        "--lora_paths",
        action="append",
        type=str,
        default=None,
        help="Path to LoRA weights file (.safetensors). Can be provided multiple times."
    )
    parser.add_argument(
        "--lora_config_path",
        type=str,
        default=None,
        help="Path to LoRA config file (.json)"
    )
    parser.add_argument(
        "--lora_config_paths",
        action="append",
        type=str,
        default=None,
        help="Path to LoRA config file (.json). Can be provided multiple times."
    )
    parser.add_argument(
        "--lora_scale",
        # keep single-value for backwards compatibility; also accept multiple via --lora_scales
        type=float,
        default=1.0,
        help="Scale factor for LoRA weights (default: 1.0)"
    )
    parser.add_argument(
        "--lora_scales",
        action="append",
        type=float,
        default=None,
        help="Scale factor for each LoRA when providing multiple --lora_paths"
    )
    parser.add_argument(
        "--lora_trigger_word",
        type=str,
        default=None,
        help="Trigger word for LoRA (automatically added to prompt start)"
    )
    parser.add_argument(
        "--lora_trigger_words",
        action="append",
        type=str,
        default=None,
        help="Trigger words for multiple LoRAs (provide once per --lora_paths)"
    )

    args = parser.parse_args()

    # Check PEFT availability if LoRA is requested
    if args.lora_path or args.lora_paths:
        try:
            import peft
        except ImportError:
            print("Warning: PEFT library is required for LoRA support but not installed.")
            print("Install it with: pip install peft>=0.7.0")
            print("Continuing without LoRA...")

    # Normalize single / multiple LoRA args into GenerationConfig fields
    normalized_lora_paths = args.lora_paths or ([args.lora_path] if args.lora_path else None)
    normalized_lora_scales = args.lora_scales or ([args.lora_scale] if args.lora_scale is not None else None)
    normalized_lora_triggers = args.lora_trigger_words or ([args.lora_trigger_word] if args.lora_trigger_word else None)

    return GenerationConfig(
        model_id=args.model_id,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        out_dir=Path(args.out_dir),
        reference_image=args.reference_image,
        ip_adapter_scale=args.ip_adapter_scale,
        ip_adapter_repo=args.ip_adapter_repo,
        ip_adapter_weight_name=args.ip_adapter_weight_name,
        ip_adapter_image_encoder=args.ip_adapter_image_encoder,
        negative_prompt=args.negative_prompt,
        lora_path=args.lora_path,
        lora_config_path=args.lora_config_path,
        lora_scale=args.lora_scale,
        lora_trigger_word=args.lora_trigger_word,
        lora_paths=normalized_lora_paths,
        lora_config_paths=args.lora_config_paths,
        lora_scales=normalized_lora_scales,
        lora_trigger_words=normalized_lora_triggers,
    )
