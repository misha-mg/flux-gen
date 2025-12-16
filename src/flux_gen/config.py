"""Configuration dataclasses for FLUX image generation."""

import os
from dataclasses import dataclass
from pathlib import Path


# Default model ID - can be overridden by MODEL_ID environment variable
MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")


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
    lora_path: str | None = None  # Path to LoRA weights file (.safetensors)
    lora_config_path: str | None = None  # Path to LoRA config file (.json)
    lora_scale: float = 1.0  # Scale factor for LoRA weights
    lora_trigger_word: str | None = None  # Trigger word for LoRA (auto-added to prompt)

    @property
    def output_path(self) -> Path:
        """Get the full path where the generated image will be saved."""
        # Generate filename based on model ID (e.g., "flux_dev.png" for FLUX.1-dev)
        model_name = self.model_id.split('/')[-1].lower().replace('.', '_').replace('-', '_')
        return self.out_dir / f"{model_name}.png"

    @property
    def effective_prompt(self) -> str:
        """Get the effective prompt with LoRA trigger word if specified."""
        if self.lora_trigger_word and self.lora_path:
            return f"{self.lora_trigger_word}, {self.prompt}"
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
