"""Tests for I/O operations."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from flux_gen.io import ensure_output_directory, save_generated_image


def test_ensure_output_directory(tmp_path):
    """Test that output directory is created if it doesn't exist."""
    test_dir = tmp_path / "test_outputs"
    assert not test_dir.exists()

    ensure_output_directory(test_dir)

    assert test_dir.exists()
    assert test_dir.is_dir()


def test_ensure_output_directory_existing(tmp_path):
    """Test that existing directory is not affected."""
    test_dir = tmp_path / "existing_dir"
    test_dir.mkdir()
    (test_dir / "existing_file.txt").write_text("test")

    ensure_output_directory(test_dir)

    assert test_dir.exists()
    assert (test_dir / "existing_file.txt").exists()


def test_save_generated_image(tmp_path):
    """Test saving generated image."""
    output_path = tmp_path / "test_image.png"
    mock_image = MagicMock()

    with patch('sys.stdout', new_callable=lambda: MagicMock()) as mock_stdout:
        save_generated_image(mock_image, output_path)

        mock_image.save.assert_called_once_with(output_path)
        # Note: print is called but we can't easily test stdout capture with MagicMock
