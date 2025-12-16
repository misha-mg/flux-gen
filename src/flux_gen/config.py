"""Configuration dataclasses for FLUX image generation."""

from dataclasses import dataclass
from pathlib import Path


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

    @property
    def output_path(self) -> Path:
        """Get the full path where the generated image will be saved."""
        return self.out_dir / "flux_schnell.png"


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
