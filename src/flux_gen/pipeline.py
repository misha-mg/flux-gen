"""FLUX pipeline loading and management."""


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
        apply_lora_to_pipeline(pipe, gen_config)

    return pipe


def apply_lora_to_pipeline(pipe, gen_config):
    """Apply LoRA weights to the FLUX pipeline."""
    try:
        # Load LoRA weights
        if gen_config.lora_config_path:
            # If config file is provided, use it
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
        raise RuntimeError(
            f"Failed to apply LoRA from '{gen_config.lora_path}': {e}\n"
            "Make sure the LoRA file is valid and compatible with FLUX model."
        )
