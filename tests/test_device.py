"""Tests for device detection and reporting."""

import pytest
from unittest.mock import patch
from io import StringIO
from flux_gen.config import RuntimeConfig
from flux_gen.device import detect_and_report_device, report_hf_token_status


def test_detect_and_report_device_cuda():
    """Test device detection and reporting with CUDA available."""
    runtime_config = RuntimeConfig(hf_token="test", has_cuda=True)

    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        with patch('torch.device') as mock_device:
            with patch('torch.cuda.get_device_name', return_value="RTX 3090"):
                with patch('torch.version.cuda', "12.1"):
                    with patch('torch.__version__', "2.1.0"):
                        device = detect_and_report_device(runtime_config)

                        output = mock_stdout.getvalue()
                        assert "Using CUDA GPU: RTX 3090" in output
                        assert "CUDA version: 12.1" in output
                        assert "PyTorch version: 2.1.0" in output


def test_detect_and_report_device_cpu():
    """Test device detection and reporting with CPU only."""
    runtime_config = RuntimeConfig(hf_token="test", has_cuda=False)

    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        with patch('torch.device') as mock_device:
            device = detect_and_report_device(runtime_config)

            output = mock_stdout.getvalue()
            assert "Warning: CUDA GPU not available, falling back to CPU" in output
            assert "Note: CPU inference will be very slow for FLUX models" in output


def test_report_hf_token_status_with_token():
    """Test HF token reporting when token is set."""
    runtime_config = RuntimeConfig(hf_token="test_token", has_cuda=True)

    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        report_hf_token_status(runtime_config)

        output = mock_stdout.getvalue()
        assert output == ""  # Should not print anything when token is set


def test_report_hf_token_status_no_token():
    """Test HF token reporting when token is not set."""
    runtime_config = RuntimeConfig(hf_token=None, has_cuda=True)

    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        report_hf_token_status(runtime_config)

        output = mock_stdout.getvalue()
        assert "Warning: HF_TOKEN environment variable not set." in output
        assert "If the model is private, set HF_TOKEN before running:" in output
        assert "export HF_TOKEN=your_huggingface_token_here" in output
