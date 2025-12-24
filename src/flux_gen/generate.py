"""Main generation orchestrator for FLUX images."""

from . import config, device, env, io, pipeline


def run_generation(gen_config: config.GenerationConfig):
    """Run the complete FLUX image generation pipeline."""
    # Apply environment settings
    env.apply_compatibility_settings()

    # Configure PyTorch thread settings for better CPU usage
    try:
        import torch
        import multiprocessing

        # Set PyTorch thread pool size based on CPU cores
        cpu_count = multiprocessing.cpu_count()
        torch_threads = max(1, min(4, cpu_count // 2))  # Use half the cores, max 4
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(torch_threads)

        print(f"PyTorch threads: {torch.get_num_threads()} intra-op, {torch.get_num_interop_threads()} inter-op")
    except ImportError:
        print("Warning: PyTorch not available for thread optimization")
    except Exception as e:
        print(f"Warning: Could not configure PyTorch threads: {e}")

    # Get runtime configuration
    runtime_config = config.RuntimeConfig.from_env()

    # Report device and token status
    device.detect_and_report_device(runtime_config)
    device.report_hf_token_status(runtime_config)

    # Ensure output directory exists
    io.ensure_output_directory(gen_config.out_dir)

    # Load pipeline
    pipe = pipeline.load_flux_pipeline(gen_config, runtime_config)

    # Run inference (use effective_prompt which includes LoRA trigger word if specified)
    effective_prompt = gen_config.effective_prompt
    if effective_prompt != gen_config.prompt:
        print(f"Using effective prompt with LoRA trigger: '{effective_prompt}'")

    pipe_kwargs = dict(
        prompt=effective_prompt,
        height=gen_config.height,
        width=gen_config.width,
        guidance_scale=gen_config.guidance_scale,
        num_inference_steps=gen_config.num_inference_steps,
    )

    # Optional: reference image conditioning via IP-Adapter
    if getattr(gen_config, "reference_image", None):
        ref_img = io.load_reference_image(
            gen_config.reference_image,
            width=gen_config.width,
            height=gen_config.height,
        )
        pipe_kwargs["ip_adapter_image"] = ref_img

    image = pipe(**pipe_kwargs).images[0]

    # Save result
    io.save_generated_image(image, gen_config.output_path)
