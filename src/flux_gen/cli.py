"""Command line interface for FLUX generation."""

import argparse
import os
from pathlib import Path

from .config import GenerationConfig, MODEL_ID


def _read_text_file(path: str) -> str:
    """Read a UTF-8 text file and return stripped content."""
    p = Path(path).expanduser()
    return p.read_text(encoding="utf-8").strip()


def _maybe_read_at_file(value: str | None) -> str | None:
    """
    Support curl-like '@file.txt' syntax for --prompt/--negative_prompt.
    If value starts with '@' and the referenced file exists, load its contents.
    """
    if not value or not isinstance(value, str):
        return value
    if value.startswith("@") and len(value) > 1:
        candidate = value[1:]
        try:
            p = Path(candidate).expanduser()
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
        except OSError:
            return value
    return value


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
        "--prompt_file",
        type=str,
        default=None,
        help="Path to a text file containing the prompt (overrides --prompt)."
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
        "--negative_prompt_file",
        type=str,
        default=None,
        help="Path to a text file containing the negative prompt (overrides --negative_prompt)."
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

    parser.add_argument(
        "--lora_fuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse LoRA into the model (default: True). Disable (--no-lora_fuse) if you want to change LoRA scales per-stage (e.g. with --refine)."
    )

    # Optional refine (img2img) stage
    parser.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable a second-pass refine stage (img2img) after base generation (default: False)."
    )
    parser.add_argument(
        "--refine_input_image",
        type=str,
        default=None,
        help="Run refine-only (img2img) on an existing image file. When set, base generation is skipped and output is saved as '<input>_refine.png' next to the input."
    )
    parser.add_argument(
        "--refine_strength",
        type=float,
        default=0.28,
        help="img2img strength for refine (default: 0.28). Lower = preserve base, higher = redraw."
    )
    parser.add_argument(
        "--refine_num_inference_steps",
        type=int,
        default=None,
        help="Number of steps for refine. Default: reuse --num_inference_steps."
    )
    parser.add_argument(
        "--refine_guidance_scale",
        type=float,
        default=None,
        help="Guidance scale for refine. Default: reuse --guidance_scale."
    )
    parser.add_argument(
        "--refine_upscale",
        type=float,
        default=1.0,
        help="Upscale factor applied to base image before refine (default: 1.0; try 1.5-2.0)."
    )
    parser.add_argument(
        "--refine_prompt",
        type=str,
        default=None,
        help="Prompt override for refine (default: reuse effective prompt). Supports '@file.txt' syntax."
    )
    parser.add_argument(
        "--refine_negative_prompt",
        type=str,
        default=None,
        help="Negative prompt override for refine (default: reuse --negative_prompt). Supports '@file.txt' syntax."
    )
    parser.add_argument(
        "--refine_ip_adapter_scale",
        type=float,
        default=None,
        help="IP-Adapter scale for refine (default: reuse --ip_adapter_scale)."
    )
    parser.add_argument(
        "--refine_lora_scales",
        action="append",
        type=float,
        default=None,
        help="Per-LoRA scales to apply for refine stage (only works when --no-lora_fuse is set). Provide once per --lora_paths."
    )

    args = parser.parse_args()

    # Load prompts from files if requested (or via '@file' syntax)
    if args.prompt_file:
        args.prompt = _read_text_file(args.prompt_file)
    else:
        args.prompt = _maybe_read_at_file(args.prompt)

    if args.negative_prompt_file:
        args.negative_prompt = _read_text_file(args.negative_prompt_file)
    else:
        args.negative_prompt = _maybe_read_at_file(args.negative_prompt)

    # Allow '@file' syntax for refine prompt overrides too
    args.refine_prompt = _maybe_read_at_file(args.refine_prompt)
    args.refine_negative_prompt = _maybe_read_at_file(args.refine_negative_prompt)

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
        lora_fuse=args.lora_fuse,
        refine=args.refine,
        refine_strength=args.refine_strength,
        refine_num_inference_steps=args.refine_num_inference_steps,
        refine_guidance_scale=args.refine_guidance_scale,
        refine_upscale=args.refine_upscale,
        refine_prompt=args.refine_prompt,
        refine_negative_prompt=args.refine_negative_prompt,
        refine_ip_adapter_scale=args.refine_ip_adapter_scale,
        refine_lora_scales=args.refine_lora_scales,
        refine_input_image=args.refine_input_image,
    )
