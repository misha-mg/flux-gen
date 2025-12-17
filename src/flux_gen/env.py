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
