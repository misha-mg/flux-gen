"""Main generation orchestrator for FLUX images."""

from . import config, device, env, io, pipeline


def run_generation(gen_config: config.GenerationConfig):
    """Run the complete FLUX image generation pipeline."""
    # Apply environment settings
    env.apply_compatibility_settings()

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

    image = pipe(
        prompt=effective_prompt,
        height=gen_config.height,
        width=gen_config.width,
        guidance_scale=gen_config.guidance_scale,
        num_inference_steps=gen_config.num_inference_steps,
    ).images[0]

    # Save result
    io.save_generated_image(image, gen_config.output_path)
