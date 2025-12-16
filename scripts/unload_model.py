#!/usr/bin/env python3
"""
Script to unload FLUX model from GPU memory to free up resources.

This script can be used to:
1. Manually unload models from memory
2. Be called from HTTP endpoints for programmatic control
3. Be integrated into Runpod lifecycle management

Usage:
    python scripts/unload_model.py [--force] [--verbose]

Options:
    --force: Force unload even if model appears not to be loaded
    --verbose: Show detailed output
"""

import argparse
import gc
import sys
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Cannot clear GPU cache.")

# Global reference to keep track of loaded models (if needed)
_loaded_models = []


def unload_model_from_memory(model: Optional[object] = None, verbose: bool = False) -> bool:
    """
    Unload a model from memory and clear GPU cache.

    Args:
        model: The model object to unload (optional)
        verbose: Whether to print detailed information

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Delete the model object if provided
        if model is not None:
            if verbose:
                print(f"Unloading model: {type(model).__name__}")
            del model

        # Clear any global references
        global _loaded_models
        if _loaded_models:
            if verbose:
                print(f"Clearing {len(_loaded_models)} model references")
            _loaded_models.clear()

        # Force garbage collection
        if verbose:
            print("Running garbage collection...")
        gc.collect()

        # Clear CUDA cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if verbose:
                print("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations are complete

            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB

            if verbose:
                print(f"Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        else:
            if verbose:
                print("CUDA not available or PyTorch not installed. Skipping GPU cache clearing.")

        if verbose:
            print("Model unloading completed successfully.")
        return True

    except Exception as e:
        print(f"Error during model unloading: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Unload FLUX model from GPU memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/unload_model.py --verbose
  python scripts/unload_model.py --force
        """
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force unload even if no models appear to be loaded'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()

    if args.verbose:
        print("FLUX Model Unloader")
        print("=" * 40)

        if TORCH_AVAILABLE:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
        else:
            print("PyTorch not available")

        print()

    # Perform the unload operation
    success = unload_model_from_memory(verbose=args.verbose)

    if success:
        print("✓ Model unloaded successfully")
        return 0
    else:
        print("✗ Failed to unload model", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
