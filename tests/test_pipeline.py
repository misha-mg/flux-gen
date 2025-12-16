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


def test_apply_lora_to_pipeline_success():
    """Test successful LoRA application."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=None,
        lora_path="/path/to/lora.safetensors",
        lora_scale=0.8
    )

    mock_pipeline = MagicMock()

    with patch('flux_gen.pipeline.apply_lora_to_pipeline') as mock_apply_lora:
        from flux_gen.pipeline import apply_lora_to_pipeline
        apply_lora_to_pipeline(mock_pipeline, gen_config)

        mock_pipeline.load_lora_weights.assert_called_once_with(
            "/path/to/lora.safetensors",
            adapter_name="custom_lora"
        )
        mock_pipeline.fuse_lora.assert_called_once_with(
            adapter_names=["custom_lora"],
            lora_scale=0.8
        )


def test_apply_lora_to_pipeline_with_config():
    """Test LoRA application with config file."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=None,
        lora_path="/path/to/lora.safetensors",
        lora_config_path="/path/to/lora_config.json",
        lora_scale=1.0
    )

    mock_pipeline = MagicMock()

    with patch('flux_gen.pipeline.apply_lora_to_pipeline') as mock_apply_lora:
        from flux_gen.pipeline import apply_lora_to_pipeline
        apply_lora_to_pipeline(mock_pipeline, gen_config)

        mock_pipeline.load_lora_weights.assert_called_once_with(
            "/path/to/lora.safetensors",
            weight_name=None,
            adapter_name="custom_lora"
        )
        mock_pipeline.fuse_lora.assert_called_once_with(
            adapter_names=["custom_lora"],
            lora_scale=1.0
        )


def test_apply_lora_to_pipeline_failure():
    """Test LoRA application failure."""
    gen_config = GenerationConfig(
        model_id="test/model",
        prompt="test",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=None,
        lora_path="/invalid/path/lora.safetensors"
    )

    mock_pipeline = MagicMock()
    mock_pipeline.load_lora_weights.side_effect = Exception("LoRA load failed")

    from flux_gen.pipeline import apply_lora_to_pipeline

    with pytest.raises(RuntimeError) as exc_info:
        apply_lora_to_pipeline(mock_pipeline, gen_config)

    error_msg = str(exc_info.value)
    assert "Failed to apply LoRA" in error_msg
    assert "LoRA load failed" in error_msg
