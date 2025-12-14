import os
import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-dev"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU не знайдено")

    os.makedirs("outputs", exist_ok=True)

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )

    # 🔑 КЛЮЧОВІ ОПТИМІЗАЦІЇ
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    prompt = (
        "cinematic portrait photo, soft natural light, "
        "85mm lens, shallow depth of field, ultra realistic"
    )

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.0,
        num_inference_steps=28,
    ).images[0]

    image.save("outputs/flux_output.png")
    print("Saved: outputs/flux_output.png")

if __name__ == "__main__":
    main()
