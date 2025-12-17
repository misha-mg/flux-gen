#!/usr/bin/env python3
"""
FLUX model unload utility for RunPod.

IMPORTANT:
- This script runs in the SAME process where the model was loaded.
- GPU memory is ONLY fully released when the process exits.
- CUDA cache cleanup is best-effort, not a guarantee.

Recommended usage on RunPod:
- Call this script or function
- Let the process exit
- Rely on RunPod to restart the worker / pod
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def log_vram(stage: str):
    """Log current VRAM usage (best-effort)."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print(f"[VRAM] {stage}: CUDA not available")
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(
        f"[VRAM] {stage}: "
        f"allocated={allocated:.2f}GB, "
        f"reserved={reserved:.2f}GB"
    )


def unload_model(
    model: Optional[object] = None,
    verbose: bool = False,
    exit_process: bool = False,
) -> None:
    """
    Best-effort unload of a model and optional process termination.

    Args:
        model: Model or pipeline object to dereference
        verbose: Print debug information
        exit_process: If True, terminate the Python process (RECOMMENDED on RunPod)
    """

    if verbose:
        print("Starting model unload sequence")

    if TORCH_AVAILABLE and torch.cuda.is_available():
        log_vram("before unload")

    # Remove Python references
    if model is not None:
        if verbose:
            print(f"Dereferencing model object: {type(model).__name__}")
        del model

    # Force garbage collection
    if verbose:
        print("Running Python garbage collection")
    gc.collect()

    # Clear CUDA cache (best-effort)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        if verbose:
            print("Clearing CUDA cache")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log_vram("after cuda cache clear")

    if exit_process:
        if verbose:
            print("Exiting process to fully release GPU memory")
        # Preferred for RunPod – clean exit
        sys.exit(0)

    if verbose:
        print(
            "Unload completed (best-effort). "
            "NOTE: Full VRAM release requires process termination."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Unload FLUX model (RunPod-safe)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed logs",
    )

    parser.add_argument(
        "--exit",
        action="store_true",
        help=(
            "Exit the Python process after cleanup. "
            "RECOMMENDED for RunPod to fully release GPU memory."
        ),
    )

    args = parser.parse_args()

    if args.verbose:
        print("FLUX Model Unloader (RunPod)")
        print("=" * 40)

        if TORCH_AVAILABLE:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("PyTorch not available")

        print()

    unload_model(
        model=None,          # model must be dereferenced by caller if needed
        verbose=args.verbose,
        exit_process=args.exit,
    )

    print("✓ Unload routine completed")


if __name__ == "__main__":
    main()
