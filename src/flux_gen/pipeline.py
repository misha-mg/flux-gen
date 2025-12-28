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
from pathlib import Path

# Diffusers pipeline
import diffusers
from diffusers import FluxPipeline

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
        adapter_names = apply_lora_to_pipeline(pipe, gen_config)
        # stash adapter names for later stage-specific scaling (e.g. refine)
        try:
            setattr(pipe, "_flux_gen_lora_adapters", adapter_names)
        except Exception:
            pass

    # Optional: IP-Adapter (reference image conditioning)
    if getattr(gen_config, "reference_image", None):
        if not hasattr(pipe, "load_ip_adapter"):
            raise RuntimeError(
                "IP-Adapter was requested (--reference_image is set), but your installed diffusers "
                f"({getattr(diffusers, '__version__', 'unknown')}) does not support "
                "`FluxPipeline.load_ip_adapter`.\n\n"
                "Fix: upgrade diffusers in your environment, for example:\n"
                "  pip install -U 'diffusers>=0.35.0'\n\n"
                "Then re-run the command."
            )
        try:
            pipe.load_ip_adapter(
                gen_config.ip_adapter_repo,
                weight_name=gen_config.ip_adapter_weight_name,
                image_encoder_pretrained_model_name_or_path=gen_config.ip_adapter_image_encoder,
            )
            pipe.set_ip_adapter_scale(gen_config.ip_adapter_scale)
            print(
                "IP-Adapter enabled: "
                f"repo={gen_config.ip_adapter_repo} "
                f"weight={gen_config.ip_adapter_weight_name} "
                f"encoder={gen_config.ip_adapter_image_encoder} "
                f"scale={gen_config.ip_adapter_scale}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to enable IP-Adapter. Error: {e}")

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

        # Resolve local file paths to absolute paths when possible.
        # If a path doesn't exist locally, keep it as-is to allow diffusers to handle
        # Hugging Face repo IDs or other resolvable identifiers.
        resolved_paths: list[str] = []
        project_root = Path(__file__).resolve().parents[2]
        for path in paths:
            p = Path(path)
            if p.exists():
                resolved_paths.append(str(p))
                continue

            candidate = project_root / path
            if candidate.exists():
                resolved_paths.append(str(candidate))
                continue

            # keep original string (may be HF repo_id or a path that will exist in runtime container)
            resolved_paths.append(path)

        # Load each LoRA under its own adapter name
        adapter_names: list[str] = []
        for idx, path in enumerate(resolved_paths):
            adapter = f"custom_lora_{idx}"
            pipe.load_lora_weights(path, adapter_name=adapter)
            adapter_names.append(adapter)

        # If requested, fuse LoRA for speed (but then scales cannot be changed per-stage)
        if getattr(gen_config, "lora_fuse", True):
            for idx, adapter in enumerate(adapter_names):
                scale = scales[idx] if idx < len(scales) else 1.0
                pipe.fuse_lora(adapter_names=[adapter], lora_scale=scale)

        # Otherwise keep adapters, but activate them with the requested weights
        else:
            set_lora_scales(pipe, adapter_names, scales)

        print("LoRA(s) successfully loaded:")
        for p, s in zip(resolved_paths, scales):
            print(f"  path: {p}  scale: {s}")

        if getattr(gen_config, "lora_fuse", True):
            print("LoRA mode: fused (fast, fixed scale)")
        else:
            print("LoRA mode: adapters (per-stage scaling enabled)")

        return adapter_names

    except Exception as e:
        raise RuntimeError(
            f"Failed to apply LoRA.\n"
            f"Error: {e}\n\n"
            "Ensure:\n"
            "- LoRA is compatible with FLUX\n"
            "- .safetensors file is valid\n"
            "- PEFT >= 0.7.0 is installed"
        )


def set_lora_scales(pipe, adapter_names: list[str], scales: list[float] | None):
    """
    Apply per-adapter LoRA weights (only works when LoRAs are kept as adapters, not fused).

    Diffusers has used both `adapter_weights` and `weights` kwarg names across versions.
    We try both for compatibility.
    """
    if not adapter_names:
        return

    weights = scales or [1.0] * len(adapter_names)
    # pad/truncate to match adapter_names length
    if len(weights) < len(adapter_names):
        weights = weights + [1.0] * (len(adapter_names) - len(weights))
    elif len(weights) > len(adapter_names):
        weights = weights[: len(adapter_names)]

    if not hasattr(pipe, "set_adapters"):
        raise RuntimeError(
            "This diffusers pipeline does not support `set_adapters`, so per-stage LoRA scaling is unavailable. "
            "Use --lora_fuse (default) or upgrade diffusers."
        )

    try:
        pipe.set_adapters(adapter_names, adapter_weights=weights)
    except TypeError:
        pipe.set_adapters(adapter_names, weights=weights)


def create_img2img_from_pipe(t2i_pipe):
    """
    Create an Image2Image pipeline that shares weights with an existing Text2Image pipeline,
    avoiding double-loading into VRAM.
    """
    try:
        from diffusers import AutoPipelineForImage2Image
    except Exception as e:
        raise RuntimeError(
            "Refine (img2img) requires diffusers AutoPipelineForImage2Image. "
            "Please upgrade diffusers in your environment.\n\n"
            f"Original error: {e}"
        )

    try:
        return AutoPipelineForImage2Image.from_pipe(t2i_pipe)
    except Exception as e:
        raise RuntimeError(
            "Failed to create img2img pipeline from the loaded FLUX pipeline. "
            "Your diffusers version may not support img2img for this model.\n\n"
            f"Original error: {e}"
        )
