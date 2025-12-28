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


def test_run_generation_with_negative_prompt(tmp_path):
    """Test generation when negative_prompt is provided."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test prompt",
        height=256,
        width=320,
        guidance_scale=2.0,
        num_inference_steps=5,
        out_dir=tmp_path / "outputs",
        negative_prompt="lowres, bad anatomy"
    )

    with patch('flux_gen.env.apply_compatibility_settings') as mock_env, \
         patch('flux_gen.config.RuntimeConfig.from_env', return_value=RuntimeConfig(hf_token="test", has_cuda=True)) as mock_runtime, \
         patch('flux_gen.device.detect_and_report_device') as mock_device_detect, \
         patch('flux_gen.device.report_hf_token_status') as mock_token_report, \
         patch('flux_gen.io.ensure_output_directory') as mock_ensure_dir, \
         patch('flux_gen.pipeline.load_flux_pipeline') as mock_load_pipe, \
         patch('flux_gen.io.save_generated_image') as mock_save:

        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_pipe.return_value.images = [mock_image]
        mock_load_pipe.return_value = mock_pipe

        run_generation(gen_config)

        mock_load_pipe.assert_called_once()
        # check that the pipeline was called with negative_prompt passed through
        mock_pipe.assert_called_once()
        _, kwargs = mock_pipe.call_args
        assert kwargs.get("negative_prompt") == "lowres, bad anatomy"
        mock_save.assert_called_once_with(mock_image, gen_config.output_path)


def test_run_generation_with_reference_image(tmp_path):
    """Test generation cycle when reference image is enabled (IP-Adapter image passed)."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test prompt",
        height=256,
        width=320,
        guidance_scale=2.0,
        num_inference_steps=5,
        out_dir=tmp_path / "outputs",
        reference_image="alina.png",
    )

    with patch('flux_gen.env.apply_compatibility_settings') as mock_env, \
         patch('flux_gen.config.RuntimeConfig.from_env', return_value=RuntimeConfig(hf_token="test", has_cuda=True)) as mock_runtime, \
         patch('flux_gen.device.detect_and_report_device') as mock_device_detect, \
         patch('flux_gen.device.report_hf_token_status') as mock_token_report, \
         patch('flux_gen.io.ensure_output_directory') as mock_ensure_dir, \
         patch('flux_gen.pipeline.load_flux_pipeline') as mock_load_pipe, \
         patch('flux_gen.io.load_reference_image') as mock_load_ref, \
         patch('flux_gen.io.save_generated_image') as mock_save:

        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_pipe.return_value.images = [mock_image]
        mock_load_pipe.return_value = mock_pipe

        mock_ref_img = MagicMock()
        mock_load_ref.return_value = mock_ref_img

        run_generation(gen_config)

        mock_load_ref.assert_called_once_with("alina.png", width=gen_config.width, height=gen_config.height)
        mock_pipe.assert_called_once()
        _, kwargs = mock_pipe.call_args
        assert kwargs["ip_adapter_image"] == mock_ref_img


def test_run_generation_with_refine_enabled(tmp_path):
    """When refine is enabled, we save base + refined images and write JSON metadata for each stage."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test prompt",
        height=256,
        width=320,
        guidance_scale=2.0,
        num_inference_steps=5,
        out_dir=tmp_path / "outputs",
        refine=True,
        refine_strength=0.25,
        refine_num_inference_steps=3,
        refine_guidance_scale=1.5,
        refine_upscale=1.0,
    )

    with patch('flux_gen.env.apply_compatibility_settings') as mock_env, \
         patch('flux_gen.config.RuntimeConfig.from_env', return_value=RuntimeConfig(hf_token="test", has_cuda=True)) as mock_runtime, \
         patch('flux_gen.device.detect_and_report_device') as mock_device_detect, \
         patch('flux_gen.device.report_hf_token_status') as mock_token_report, \
         patch('flux_gen.io.ensure_output_directory') as mock_ensure_dir, \
         patch('flux_gen.pipeline.load_flux_pipeline') as mock_load_pipe, \
         patch('flux_gen.pipeline.create_img2img_from_pipe') as mock_create_i2i, \
         patch('flux_gen.io.save_generated_image') as mock_save_img, \
         patch('flux_gen.io.save_metadata_json') as mock_save_meta:

        # base pipe returns base image
        mock_pipe = MagicMock()
        mock_base_img = MagicMock()
        mock_pipe.return_value.images = [mock_base_img]
        mock_load_pipe.return_value = mock_pipe

        # i2i pipe returns refined image
        mock_i2i = MagicMock()
        mock_refined_img = MagicMock()
        mock_i2i.return_value.images = [mock_refined_img]
        mock_create_i2i.return_value = mock_i2i

        run_generation(gen_config)

        # base called once
        mock_pipe.assert_called_once()
        # refine called once
        mock_create_i2i.assert_called_once_with(mock_pipe)
        mock_i2i.assert_called_once()

        # images saved twice (base + refined)
        assert mock_save_img.call_count == 2
        assert mock_save_meta.call_count == 2


def test_run_generation_refine_only_saves_single_image_no_json(tmp_path):
    """Refine-only mode should save only '<input>_refine.png' and never write JSON metadata."""
    input_path = tmp_path / "some_base.png"
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test prompt",
        height=256,
        width=320,
        guidance_scale=2.0,
        num_inference_steps=5,
        out_dir=tmp_path / "outputs",
        refine_input_image=str(input_path),
        refine_strength=0.25,
        refine_num_inference_steps=3,
        refine_guidance_scale=1.5,
        refine_upscale=1.0,
    )

    with patch('flux_gen.env.apply_compatibility_settings'), \
         patch('flux_gen.config.RuntimeConfig.from_env', return_value=RuntimeConfig(hf_token="test", has_cuda=True)), \
         patch('flux_gen.device.detect_and_report_device'), \
         patch('flux_gen.device.report_hf_token_status'), \
         patch('flux_gen.io.ensure_output_directory'), \
         patch('flux_gen.pipeline.load_flux_pipeline') as mock_load_pipe, \
         patch('flux_gen.pipeline.create_img2img_from_pipe') as mock_create_i2i, \
         patch('flux_gen.io.load_init_image') as mock_load_init, \
         patch('flux_gen.io.save_generated_image') as mock_save_img, \
         patch('flux_gen.io.save_metadata_json') as mock_save_meta:

        mock_pipe = MagicMock()
        mock_load_pipe.return_value = mock_pipe

        mock_init_img = MagicMock()
        mock_load_init.return_value = mock_init_img

        mock_i2i = MagicMock()
        mock_refined_img = MagicMock()
        mock_i2i.return_value.images = [mock_refined_img]
        mock_create_i2i.return_value = mock_i2i

        run_generation(gen_config)

        # base pipe should NOT be called for t2i generation
        mock_pipe.assert_not_called()
        # i2i should be called once
        mock_create_i2i.assert_called_once_with(mock_pipe)
        mock_i2i.assert_called_once()

        # Saved exactly once: "<input>_refine.png"
        assert mock_save_img.call_count == 1
        args, _ = mock_save_img.call_args
        assert str(args[1]).endswith("some_base_refine.png")

        # No JSON written
        mock_save_meta.assert_not_called()
