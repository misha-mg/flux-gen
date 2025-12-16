"""FLUX image generation CLI wrapper."""

from flux_gen.cli import parse_args
from flux_gen.generate import run_generation


def main():
    """Main entry point for FLUX image generation."""
    config = parse_args()
    run_generation(config)


if __name__ == "__main__":
    main()
