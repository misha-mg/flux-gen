"""Configuration dataclasses for FLUX image generation."""

import os
from dataclasses import dataclass
from pathlib import Path


# Default model ID - can be overridden by MODEL_ID environment variable
MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")

# Default directory name (relative to repo root) for reference images
REFERENCE_IMAGES_DIRNAME = "reference_images"


@dataclass
class GenerationConfig:
    """Configuration for image generation parameters."""
    model_id: str
    prompt: str
    height: int
    width: int
    guidance_scale: float
    num_inference_steps: int
    out_dir: Path
    # Optional: reference image conditioning (IP-Adapter for FLUX)
    reference_image: str | None = None  # Path to image or filename inside reference_images/
    ip_adapter_repo: str = "XLabs-AI/flux-ip-adapter"
    ip_adapter_weight_name: str = "ip_adapter.safetensors"
    ip_adapter_image_encoder: str = "openai/clip-vit-large-patch14"
    ip_adapter_scale: float = 1.0
    # Optional negative prompt to discourage undesired content
    negative_prompt: str | None = None
    # Backwards-compatible single LoRA fields:
    lora_path: str | None = None  # Path to LoRA weights file (.safetensors)
    lora_config_path: str | None = None  # Path to LoRA config file (.json)
    lora_scale: float = 1.0  # Scale factor for single LoRA weights
    lora_trigger_word: str | None = None  # Trigger word for single LoRA (auto-added to prompt)

    # Support multiple LoRAs:
    lora_paths: list[str] | None = None  # Multiple LoRA weight paths (use CLI --lora_paths)
    lora_config_paths: list[str] | None = None
    lora_scales: list[float] | None = None
    lora_trigger_words: list[str] | None = None

    @property
    def output_path(self) -> Path:
        """Get the full path where the generated image will be saved."""
        # Generate filename based on model ID (e.g., "flux_dev.png" for FLUX.1-dev)
        # If a file with the base name already exists, append an incrementing
        # index suffix like `_0001` to avoid overwriting previous outputs.
        model_name = self.model_id.split('/')[-1].lower().replace('.', '_').replace('-', '_')
        base = model_name
        out_dir = self.out_dir
        # Ensure directory exists
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find existing PNG files that start with the base name
        existing_files = list(out_dir.glob(f"{base}*.png"))
        max_idx = 0
        for p in existing_files:
            stem = p.stem  # e.g., "flux_dev", "flux_dev_0001"
            if stem == base:
                idx = 0
            else:
                parts = stem.rsplit("_", 1)
                if len(parts) == 2 and parts[0] == base and parts[1].isdigit():
                    idx = int(parts[1])
                else:
                    # ignore unrelated filenames like "flux_dev_extra.png"
                    continue
            if idx > max_idx:
                max_idx = idx

        next_idx = max_idx + 1
        # If no existing files at all, use the base filename without suffix.
        if next_idx == 1 and not (out_dir / f"{base}.png").exists():
            return out_dir / f"{base}.png"

        # Otherwise use zero-padded 4-digit index suffix.
        filename = f"{base}_{next_idx:04d}.png"
        return out_dir / filename

    @property
    def effective_prompt(self) -> str:
        """Get the effective prompt with LoRA trigger word if specified."""
        triggers: list[str] = []

        # Single LoRA trigger (backwards compatible)
        if self.lora_trigger_word and (self.lora_path or (self.lora_paths and len(self.lora_paths) > 0)):
            triggers.append(self.lora_trigger_word)

        # Multiple LoRA triggers
        if self.lora_trigger_words and self.lora_paths:
            triggers.extend(self.lora_trigger_words)

        if triggers:
            return f"{', '.join(triggers)}, {self.prompt}"
        return self.prompt


@dataclass
class RuntimeConfig:
    """Configuration for runtime environment."""
    hf_token: str | None
    has_cuda: bool

    @classmethod
    def from_env(cls) -> 'RuntimeConfig':
        """Create RuntimeConfig from environment variables."""
        import os
        return cls(
            hf_token=os.getenv("HF_TOKEN"),
            has_cuda=cls._detect_cuda()
        )

    @staticmethod
    def _detect_cuda() -> bool:
        """Detect CUDA availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
