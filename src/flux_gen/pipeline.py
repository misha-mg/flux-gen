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

    return pipe
