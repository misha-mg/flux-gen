"""
FLUX pipeline loading and management (optimized for RTX A6000 / A100).

Key principles:
- Explicit dtype (bf16 / fp16) to avoid FP32 OOM
- Full GPU placement for 48GB+ VRAM
- No CPU offload unless explicitly required
- Safe LoRA loading & fusion
"""

from __future__ import annotations

import torch

# Optional PEFT support for LoRA
try:
    import peft
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def load_flux_pipeline(gen_config, runtime_config):
    """
    Load FLUX pipeline optimized for large VRAM GPUs (RTX A6000, A100).

    Strategy:
    - bf16 on GPU
    - no CPU offload
    - safe LoRA fusion
    """

    from diffusers import FluxPipeline

    if not runtime_config.has_cuda:
        raise RuntimeError("CUDA is required to run FLUX models.")

    try:
        pipe = FluxPipeline.from_pretrained(
            gen_config.model_id,
            torch_dtype=torch.bfloat16,      # CRITICAL: avoid FP32
            device_map="balanced",           # balance across GPU devices
            token=runtime_config.hf_token,
        )
    except Exception as e:
        if "401" in str(e) or "authorization" in str(e).lower():
            raise RuntimeError(
                f"Failed to load model '{gen_config.model_id}'.\n"
                "The model may be private.\n"
                "Please set HuggingFace token:\n\n"
                "  export HF_TOKEN=your_huggingface_token\n\n"
                f"Original error: {e}"
            )
        raise

    # Enable memory-efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers memory-efficient attention enabled")
    except Exception:
        print("xFormers not available — using default attention")

    # Apply LoRA if provided
    if gen_config.lora_path or (hasattr(gen_config, "lora_paths") and gen_config.lora_paths):
        apply_lora_to_pipeline(pipe, gen_config)

    return pipe


def apply_lora_to_pipeline(pipe, gen_config):
    """
    Load and fuse LoRA weights into FLUX pipeline.
    """

    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "PEFT library is required for LoRA support.\n"
            "Install with:\n\n"
            "  pip install peft>=0.7.0\n"
        )

    try:
        # Support both single and multiple LoRAs.
        paths: list[str] = []
        if getattr(gen_config, "lora_paths", None):
            paths = gen_config.lora_paths or []
        elif gen_config.lora_path:
            paths = [gen_config.lora_path]

        if not paths:
            print("No LoRA paths provided; skipping LoRA application.")
            return

        # Determine scales per-LoRA (fall back to single lora_scale)
        scales: list[float] = []
        if getattr(gen_config, "lora_scales", None):
            scales = gen_config.lora_scales or []
        elif getattr(gen_config, "lora_scale", None) is not None:
            scales = [gen_config.lora_scale] * len(paths)
        else:
            scales = [1.0] * len(paths)

        # Load each LoRA under its own adapter name and fuse with corresponding scale
        adapter_names: list[str] = []
        for idx, path in enumerate(paths):
            adapter = f"custom_lora_{idx}"
            pipe.load_lora_weights(path, adapter_name=adapter)
            adapter_names.append(adapter)

            # Use corresponding scale if available, else 1.0
            scale = scales[idx] if idx < len(scales) else 1.0
            pipe.fuse_lora(adapter_names=[adapter], lora_scale=scale)

        print("LoRA(s) successfully fused:")
        for p, s in zip(paths, scales):
            print(f"  path: {p}  scale: {s}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to apply LoRA from '{gen_config.lora_path}'.\n"
            f"Error: {e}\n\n"
            "Ensure:\n"
            "- LoRA is compatible with FLUX\n"
            "- .safetensors file is valid\n"
            "- PEFT >= 0.7.0 is installed"
        )
