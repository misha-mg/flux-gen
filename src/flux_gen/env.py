"""Environment setup for FLUX compatibility."""

import os


def apply_compatibility_settings():
    """Apply environment variables for FLUX model compatibility."""
    # Set environment variables for FLUX compatibility
    os.environ['DIFFUSERS_FORCE_ATTENTION_BACKEND'] = 'math'
    os.environ.setdefault('TORCH_USE_CUDA_DSA', '1')
