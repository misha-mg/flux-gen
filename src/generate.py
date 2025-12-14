import os
import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    os.makedirs("outputs", exist_ok=True)

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        device_map="balanced",          # 🔑 КРИТИЧНО
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

    image.save("outputs/flux_schnell.png")
    print("Saved: outputs/flux_schnell.png")

if __name__ == "__main__":
    main()
