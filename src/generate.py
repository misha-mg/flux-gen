import os
import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    # Keep outputs inside src/ to match repo structure.
    out_dir = os.path.join("src", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Avoid accidental fp32 loads (common cause of OOM on 24GB cards).
    # NOTE: Diffusers expects torch_dtype (NOT dtype).
    torch_dtype = torch.float16

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        # On a single 24GB GPU, "balanced" can still push too much into VRAM during load.
        # Load to CPU first, then offload during inference.
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # 🔑 КРИТИЧНО: offload
    pipe.enable_model_cpu_offload()

    prompt = (
        "cinematic portrait photo, soft natural light, "
        "85mm lens, shallow depth of field, ultra realistic"
    )

    image = pipe(
        prompt=prompt,
        height=768,                     # 🔑 ЗМЕНШЕНО
        width=768,
        guidance_scale=3.5,
        num_inference_steps=20,
    ).images[0]

    out_path = os.path.join(out_dir, "flux_schnell.png")
    image.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
