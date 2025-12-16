import os
import argparse
import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

def main(args):
    # Device detection and setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")

        # Load to CPU first, then enable offload during inference for better VRAM management
        device_map = "cpu"
        torch_dtype = torch.float16
        use_cpu_offload = True
    else:
        print("Warning: CUDA GPU not available, falling back to CPU")
        print("Note: CPU inference will be very slow for FLUX models")
        device = torch.device("cpu")
        device_map = "cpu"
        torch_dtype = torch.float32  # Use float32 on CPU for better compatibility
        use_cpu_offload = False

    # Check for Hugging Face token if model might be private
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")
        print("If the model is private, set HF_TOKEN before running:")
        print("export HF_TOKEN=your_huggingface_token_here")

    # Keep outputs inside src/ to match repo structure.
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Avoid accidental fp32 loads (common cause of OOM on 24GB cards).
    # NOTE: Diffusers expects torch_dtype (NOT dtype).

    try:
        pipe = FluxPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            use_auth_token=hf_token,
        )
    except Exception as e:
        if "401" in str(e) or "authorization" in str(e).lower():
            raise RuntimeError(
                f"Failed to load model '{args.model_id}'. "
                "This might be a private model. Please set HF_TOKEN environment variable:\n"
                "export HF_TOKEN=your_huggingface_token_here\n"
                f"Original error: {e}"
            )
        else:
            raise

    # Enable CPU offload for memory efficiency on RTX 3090
    if use_cpu_offload:
        pipe.enable_model_cpu_offload()

    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    ).images[0]

    out_path = os.path.join(out_dir, "flux_schnell.png")
    image.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
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

    args = parser.parse_args()
    main(args)
