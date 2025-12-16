"""Tests for configuration and runtime setup."""

import pytest
from unittest.mock import patch
from pathlib import Path
from flux_gen.config import GenerationConfig, RuntimeConfig


def test_generation_config_output_path():
    """Test that output_path property works correctly."""
    config = GenerationConfig(
        model_id="test/model",
        prompt="test prompt",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=Path("/tmp/test_outputs")
    )

    assert config.output_path == Path("/tmp/test_outputs/flux_schnell.png")


def test_generation_config_with_lora():
    """Test GenerationConfig with LoRA parameters."""
    config = GenerationConfig(
        model_id="test/model",
        prompt="test prompt",
        height=512,
        width=512,
        guidance_scale=2.0,
        num_inference_steps=10,
        out_dir=Path("/tmp/test_outputs"),
        lora_path="/path/to/lora.safetensors",
        lora_config_path="/path/to/lora_config.json",
        lora_scale=0.8
    )

    assert config.lora_path == "/path/to/lora.safetensors"
    assert config.lora_config_path == "/path/to/lora_config.json"
    assert config.lora_scale == 0.8


def test_runtime_config_from_env():
    """Test RuntimeConfig creation from environment."""
    with patch.dict('os.environ', {'HF_TOKEN': 'test_token'}, clear=True):
        with patch('flux_gen.config.RuntimeConfig._detect_cuda', return_value=True):
            config = RuntimeConfig.from_env()

            assert config.hf_token == 'test_token'
            assert config.has_cuda is True


def test_runtime_config_from_env_no_token():
    """Test RuntimeConfig creation when HF_TOKEN is not set."""
    with patch.dict('os.environ', {}, clear=True):
        with patch('flux_gen.config.RuntimeConfig._detect_cuda', return_value=False):
            config = RuntimeConfig.from_env()

            assert config.hf_token is None
            assert config.has_cuda is False


def test_runtime_config_detect_cuda():
    """Test CUDA detection."""
    # Test with torch available
    with patch.dict('sys.modules', {'torch': pytest.MagicMock()}):
        with patch('torch.cuda.is_available', return_value=True):
            assert RuntimeConfig._detect_cuda() is True

        with patch('torch.cuda.is_available', return_value=False):
            assert RuntimeConfig._detect_cuda() is False

    # Test with torch not available
    with patch.dict('sys.modules', {}, clear=True):
        with patch('builtins.__import__', side_effect=ImportError):
            assert RuntimeConfig._detect_cuda() is False
