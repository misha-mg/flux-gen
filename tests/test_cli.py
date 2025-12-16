"""Tests for CLI argument parsing."""

import pytest
from pathlib import Path
from flux_gen.cli import parse_args, MODEL_ID


def test_parse_args_defaults():
    """Test that default arguments are parsed correctly."""
    # Mock sys.argv to simulate command line call
    import sys
    original_argv = sys.argv
    try:
        sys.argv = ['generate.py']  # Simulate running without arguments
        config = parse_args()

        assert config.model_id == MODEL_ID
        assert config.prompt == "cinematic portrait photo, soft natural light, 85mm lens, shallow depth of field, ultra realistic"
        assert config.out_dir == Path("src/outputs")
        assert config.height == 768
        assert config.width == 768
        assert config.guidance_scale == 3.5
        assert config.num_inference_steps == 20
        assert config.output_path == Path("src/outputs/flux_schnell.png")
    finally:
        sys.argv = original_argv


def test_parse_args_custom_values():
    """Test that custom arguments are parsed correctly."""
    import sys
    original_argv = sys.argv
    try:
        sys.argv = [
            'generate.py',
            '--model_id', 'custom/model',
            '--prompt', 'test prompt',
            '--out_dir', './custom_outputs',
            '--height', '512',
            '--width', '1024',
            '--guidance_scale', '2.0',
            '--num_inference_steps', '10'
        ]
        config = parse_args()

        assert config.model_id == 'custom/model'
        assert config.prompt == 'test prompt'
        assert config.out_dir == Path('./custom_outputs')
        assert config.height == 512
        assert config.width == 1024
        assert config.guidance_scale == 2.0
        assert config.num_inference_steps == 10
        assert config.output_path == Path('./custom_outputs/flux_schnell.png')
    finally:
        sys.argv = original_argv
