"""Environment setup for FLUX compatibility."""

import os
import multiprocessing


def optimize_cpu_settings():
    """Optimize CPU thread settings based on available cores."""
    try:
        cpu_count = multiprocessing.cpu_count()
        # Use 75% of available cores, but at least 2 and at most 8
        optimal_threads = max(2, min(8, int(cpu_count * 0.75)))

        os.environ.setdefault('OMP_NUM_THREADS', str(optimal_threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(optimal_threads))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(optimal_threads))
        os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(optimal_threads))

        print(f"CPU optimization: Using {optimal_threads} threads (out of {cpu_count} available cores)")

    except Exception as e:
        print(f"Warning: Could not optimize CPU settings: {e}")
        # Fallback to conservative defaults
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        os.environ.setdefault('MKL_NUM_THREADS', '4')


def apply_compatibility_settings():
    """Apply environment variables for FLUX model compatibility."""
    # Optimize CPU settings first
    optimize_cpu_settings()

    # Set environment variables for FLUX compatibility
    os.environ['DIFFUSERS_FORCE_ATTENTION_BACKEND'] = 'math'
    os.environ.setdefault('TORCH_USE_CUDA_DSA', '1')

    # Additional PyTorch settings for better memory management
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    # PyTorch compatibility shim:
    # Some newer diffusers versions call torch.nn.functional.scaled_dot_product_attention
    # with enable_gqa=..., which doesn't exist in older PyTorch builds.
    try:
        import inspect
        import torch

        sdp = torch.nn.functional.scaled_dot_product_attention
        try:
            params = inspect.signature(sdp).parameters
        except (TypeError, ValueError):
            params = {}

        if "enable_gqa" not in params:
            _orig_sdp = sdp

            def _sdp_compat(*args, **kwargs):
                kwargs.pop("enable_gqa", None)
                return _orig_sdp(*args, **kwargs)

            torch.nn.functional.scaled_dot_product_attention = _sdp_compat  # type: ignore[attr-defined]
            print("Applied PyTorch SDP compat shim (ignored enable_gqa).")
    except Exception as e:
        print(f"Warning: Could not apply SDP compat shim: {e}")
