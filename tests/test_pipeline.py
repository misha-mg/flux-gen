"""Tests for pipeline loading and error handling."""

import pytest
from unittest.mock import patch, MagicMock
from flux_gen.config import GenerationConfig, RuntimeConfig
from flux_gen.pipeline import load_flux_pipeline


def test_load_flux_pipeline_success():
    """Test successful pipeline loading."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=None  # Not needed for this test
    )
    runtime_config = RuntimeConfig(hf_token="test_token", has_cuda=True)

    mock_pipeline = MagicMock()
    with patch('flux_gen.pipeline.FluxPipeline.from_pretrained', return_value=mock_pipeline):
        result = load_flux_pipeline(gen_config, runtime_config)

        assert result == mock_pipeline
        mock_pipeline.enable_model_cpu_offload.assert_called_once()


def test_load_flux_pipeline_401_error():
    """Test pipeline loading with 401 authorization error."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=None
    )
    runtime_config = RuntimeConfig(hf_token=None, has_cuda=True)

    with patch('flux_gen.pipeline.FluxPipeline.from_pretrained', side_effect=Exception("401 Unauthorized")):
        with pytest.raises(RuntimeError) as exc_info:
            load_flux_pipeline(gen_config, runtime_config)

        error_msg = str(exc_info.value)
        assert "Failed to load model 'test/model'" in error_msg
        assert "HF_TOKEN" in error_msg


def test_load_flux_pipeline_auth_error():
    """Test pipeline loading with authorization error."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=None
    )
    runtime_config = RuntimeConfig(hf_token=None, has_cuda=True)

    with patch('flux_gen.pipeline.FluxPipeline.from_pretrained', side_effect=Exception("authorization failed")):
        with pytest.raises(RuntimeError) as exc_info:
            load_flux_pipeline(gen_config, runtime_config)

        error_msg = str(exc_info.value)
        assert "Failed to load model 'test/model'" in error_msg
        assert "HF_TOKEN" in error_msg


def test_load_flux_pipeline_other_error():
    """Test pipeline loading with other errors (should re-raise)."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=None
    )
    runtime_config = RuntimeConfig(hf_token="test_token", has_cuda=True)

    original_error = Exception("Some other error")
    with patch('flux_gen.pipeline.FluxPipeline.from_pretrained', side_effect=original_error):
        with pytest.raises(Exception) as exc_info:
            load_flux_pipeline(gen_config, runtime_config)

        assert exc_info.value == original_error
