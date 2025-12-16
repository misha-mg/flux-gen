"""Tests for main generation orchestrator."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from flux_gen.config import GenerationConfig, RuntimeConfig
from flux_gen.generate import run_generation


def test_run_generation_full_cycle(tmp_path):
    """Test full generation cycle with mocked dependencies."""
    # Create config
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test prompt",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=tmp_path / "outputs"
    )

    # Mock all dependencies
    with patch('flux_gen.env.apply_compatibility_settings') as mock_env, \
         patch('flux_gen.config.RuntimeConfig.from_env', return_value=RuntimeConfig(hf_token="test", has_cuda=True)) as mock_runtime, \
         patch('flux_gen.device.detect_and_report_device') as mock_device_detect, \
         patch('flux_gen.device.report_hf_token_status') as mock_token_report, \
         patch('flux_gen.io.ensure_output_directory') as mock_ensure_dir, \
         patch('flux_gen.pipeline.load_flux_pipeline') as mock_load_pipe, \
         patch('flux_gen.io.save_generated_image') as mock_save:

        # Setup mock pipeline
        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_pipe.return_value.images = [mock_image]
        mock_load_pipe.return_value = mock_pipe

        # Run generation
        run_generation(gen_config)

        # Verify all steps were called
        mock_env.assert_called_once()
        mock_runtime.assert_called_once()
        mock_device_detect.assert_called_once()
        mock_token_report.assert_called_once()
        mock_ensure_dir.assert_called_once_with(gen_config.out_dir)
        mock_load_pipe.assert_called_once()
        mock_pipe.assert_called_once_with(
            prompt=gen_config.prompt,
            height=gen_config.height,
            width=gen_config.width,
            guidance_scale=gen_config.guidance_scale,
            num_inference_steps=gen_config.num_inference_steps,
        )
        mock_save.assert_called_once_with(mock_image, gen_config.output_path)
