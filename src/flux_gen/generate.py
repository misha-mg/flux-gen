"""Main generation orchestrator for FLUX images."""

from pathlib import Path

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

    # Refine-only mode: run img2img on an existing image file; save only "<input>_refine.png" (no JSON).
    if getattr(gen_config, "refine_input_image", None):
        init_img = io.load_init_image(gen_config.refine_input_image)
        refine_input = io.upscale_image(init_img, gen_config.refine_upscale)

        i2i = pipeline.create_img2img_from_pipe(pipe)

        # Stage-specific overrides
        refine_prompt = gen_config.refine_prompt or effective_prompt
        refine_negative = (
            gen_config.refine_negative_prompt
            if gen_config.refine_negative_prompt is not None
            else getattr(gen_config, "negative_prompt", None)
        )
        refine_steps = (
            gen_config.refine_num_inference_steps
            if gen_config.refine_num_inference_steps is not None
            else gen_config.num_inference_steps
        )
        refine_guidance = (
            gen_config.refine_guidance_scale
            if gen_config.refine_guidance_scale is not None
            else gen_config.guidance_scale
        )

        # Optional: lower reference conditioning during refine
        if getattr(gen_config, "reference_image", None) and gen_config.refine_ip_adapter_scale is not None:
            if hasattr(i2i, "set_ip_adapter_scale"):
                i2i.set_ip_adapter_scale(gen_config.refine_ip_adapter_scale)

        # Optional: per-stage LoRA scaling (only works when LoRA is not fused)
        if not getattr(gen_config, "lora_fuse", True) and getattr(gen_config, "refine_lora_scales", None):
            adapter_names = getattr(pipe, "_flux_gen_lora_adapters", None) or getattr(i2i, "_flux_gen_lora_adapters", None)
            if adapter_names:
                pipeline.set_lora_scales(i2i, adapter_names, gen_config.refine_lora_scales)

        refine_kwargs = dict(
            prompt=refine_prompt,
            image=refine_input,
            strength=gen_config.refine_strength,
            num_inference_steps=refine_steps,
            guidance_scale=refine_guidance,
        )
        if refine_negative:
            refine_kwargs["negative_prompt"] = refine_negative

        if getattr(gen_config, "reference_image", None):
            ref_img = io.load_reference_image(
                gen_config.reference_image,
                width=refine_input.width,
                height=refine_input.height,
            )
            refine_kwargs["ip_adapter_image"] = ref_img

        refined_image = i2i(**refine_kwargs).images[0]

        in_path = Path(gen_config.refine_input_image).expanduser()
        suffix = in_path.suffix if in_path.suffix else ".png"
        out_path = in_path.with_name(f"{in_path.stem}_refine{suffix}")
        io.save_generated_image(refined_image, out_path)
        return

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
    # Optional: negative prompt
    if getattr(gen_config, "negative_prompt", None):
        pipe_kwargs["negative_prompt"] = gen_config.negative_prompt

    base_image = pipe(**pipe_kwargs).images[0]

    # If refine is disabled, behave exactly like before
    if not getattr(gen_config, "refine", False):
        io.save_generated_image(base_image, gen_config.output_path)
        return

    # Refine enabled: save base output and metadata, then run img2img refine and save final output.
    out_path = gen_config.output_path
    base_path = out_path.with_name(f"{out_path.stem}_base{out_path.suffix}")
    base_meta_path = base_path.with_suffix(".json")

    io.save_generated_image(base_image, base_path)
    io.save_metadata_json(
        {
            "stage": "base",
            "output_image": str(base_path),
            "config": io.config_to_metadata(gen_config),
        },
        base_meta_path,
    )

    i2i = pipeline.create_img2img_from_pipe(pipe)

    # Stage-specific overrides
    refine_prompt = gen_config.refine_prompt or effective_prompt
    refine_negative = (
        gen_config.refine_negative_prompt
        if gen_config.refine_negative_prompt is not None
        else getattr(gen_config, "negative_prompt", None)
    )
    refine_steps = (
        gen_config.refine_num_inference_steps
        if gen_config.refine_num_inference_steps is not None
        else gen_config.num_inference_steps
    )
    refine_guidance = (
        gen_config.refine_guidance_scale
        if gen_config.refine_guidance_scale is not None
        else gen_config.guidance_scale
    )

    # Optional: lower reference conditioning during refine
    if getattr(gen_config, "reference_image", None) and gen_config.refine_ip_adapter_scale is not None:
        if hasattr(i2i, "set_ip_adapter_scale"):
            i2i.set_ip_adapter_scale(gen_config.refine_ip_adapter_scale)

    # Optional: per-stage LoRA scaling (only works when LoRA is not fused)
    if not getattr(gen_config, "lora_fuse", True) and getattr(gen_config, "refine_lora_scales", None):
        adapter_names = getattr(pipe, "_flux_gen_lora_adapters", None) or getattr(i2i, "_flux_gen_lora_adapters", None)
        if adapter_names:
            pipeline.set_lora_scales(i2i, adapter_names, gen_config.refine_lora_scales)

    # Optional: upscale base image before refine (gives the model "room" for microdetail)
    refine_input = io.upscale_image(base_image, gen_config.refine_upscale)

    refine_kwargs = dict(
        prompt=refine_prompt,
        image=refine_input,
        strength=gen_config.refine_strength,
        num_inference_steps=refine_steps,
        guidance_scale=refine_guidance,
    )
    if refine_negative:
        refine_kwargs["negative_prompt"] = refine_negative

    # If reference image is enabled, re-load it resized to current refine resolution
    if getattr(gen_config, "reference_image", None):
        ref_img = io.load_reference_image(
            gen_config.reference_image,
            width=refine_input.width,
            height=refine_input.height,
        )
        refine_kwargs["ip_adapter_image"] = ref_img

    refined_image = i2i(**refine_kwargs).images[0]

    refined_meta_path = out_path.with_suffix(".json")
    io.save_generated_image(refined_image, out_path)
    io.save_metadata_json(
        {
            "stage": "refine",
            "input_image": str(base_path),
            "output_image": str(out_path),
            "refine_input_size": [refine_input.width, refine_input.height],
            "config": io.config_to_metadata(gen_config),
            "refine_params": {
                "strength": gen_config.refine_strength,
                "num_inference_steps": refine_steps,
                "guidance_scale": refine_guidance,
                "ip_adapter_scale": (
                    gen_config.refine_ip_adapter_scale
                    if gen_config.refine_ip_adapter_scale is not None
                    else gen_config.ip_adapter_scale
                ),
                "lora_fuse": getattr(gen_config, "lora_fuse", True),
                "refine_lora_scales": getattr(gen_config, "refine_lora_scales", None),
            },
        },
        refined_meta_path,
    )
