"""FLUX pipeline loading and management."""

# Import PEFT for LoRA support
try:
    import peft
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def load_flux_pipeline(gen_config, runtime_config):
    """Load and return FLUX pipeline with error handling."""
    from diffusers import FluxPipeline

    try:
        # For FLUX models, use CPU offload without device_map for better memory management
        # Let the pipeline use default dtype to avoid deprecation warnings
        pipe = FluxPipeline.from_pretrained(
            gen_config.model_id,
            low_cpu_mem_usage=True,
            token=runtime_config.hf_token,
        )
    except Exception as e:
        if "401" in str(e) or "authorization" in str(e).lower():
            raise RuntimeError(
                f"Failed to load model '{gen_config.model_id}'. "
                "This might be a private model. Please set HF_TOKEN environment variable:\n"
                "export HF_TOKEN=your_huggingface_token_here\n"
                f"Original error: {e}"
            )
        else:
            raise

    # Enable CPU offload for memory efficiency - this is crucial for large models like FLUX
    pipe.enable_model_cpu_offload()

    # Load and apply LoRA if specified
    if gen_config.lora_path:
        try:
            apply_lora_to_pipeline(pipe, gen_config)
        except RuntimeError as e:
            if "PEFT library is required" in str(e):
                print(f"Warning: {e}")
                print("Continuing without LoRA...")
            else:
                raise

    return pipe


def apply_lora_to_pipeline(pipe, gen_config):
    """Apply LoRA weights to the FLUX pipeline."""
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "PEFT library is required for LoRA support. Please install it with:\n"
            "pip install peft>=0.7.0"
        )

    try:
        # Load LoRA weights
        if gen_config.lora_config_path:
            # If config file is provided, use it for more control
            pipe.load_lora_weights(
                gen_config.lora_path,
                weight_name=None,  # Will be inferred from safetensors file
                adapter_name="custom_lora"
            )
        else:
            # Load from safetensors directly
            pipe.load_lora_weights(gen_config.lora_path, adapter_name="custom_lora")

        # Fuse LoRA weights into the model for better performance
        pipe.fuse_lora(adapter_names=["custom_lora"], lora_scale=gen_config.lora_scale)

        print(f"LoRA applied successfully: {gen_config.lora_path} (scale: {gen_config.lora_scale})")

    except Exception as e:
        error_msg = f"Failed to apply LoRA from '{gen_config.lora_path}': {e}\n"
        if "PEFT backend is required" in str(e):
            error_msg += "Make sure PEFT library is installed: pip install peft>=0.7.0\n"
        error_msg += "Make sure the LoRA file is valid and compatible with FLUX model."
        raise RuntimeError(error_msg)
