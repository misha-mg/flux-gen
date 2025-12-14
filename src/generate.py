import os
import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-dev"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FLUX потребує CUDA GPU. "
            "На macOS цей скрипт призначений для запуску на RunPod."
        )

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    ).to("cuda")

    image = pipe(
        prompt="cinematic portrait photo, ultra realistic, soft light",
        height=1024,
        width=1024,
        guidance_scale=4.0,
        num_inference_steps=28,
    ).images[0]

    os.makedirs("outputs", exist_ok=True)
    image.save("outputs/flux_output.png")

if __name__ == "__main__":
    main()
