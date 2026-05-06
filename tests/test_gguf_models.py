"""
Tests for Phase 3: GGUF model loading and RAG pipeline.

Strategy:
- test_gguf_models.py tests the model wrapper in isolation
- Tests use mocking for the Llama model so they run WITHOUT a real GGUF file
- Integration tests (marked with @pytest.mark.integration) require a real model
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.gguf_models import LocalGGUFModel, list_available_models


# ── Unit Tests (no real model needed) ───────────────────────────────


class TestLocalGGUFModel:
    """Test LocalGGUFModel with mocked Llama backend."""

    def test_model_file_not_found(self):
        """Should raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            LocalGGUFModel("nonexistent/model.gguf")

    @patch("src.gguf_models.Llama")
    def test_model_loads_successfully(self, mock_llama, tmp_path):
        """Should load model when file exists."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"fake model data")

        model = LocalGGUFModel(str(model_file))
        assert model.llm is not None
        assert model.model_name == "test"
        mock_llama.assert_called_once()

    @patch("src.gguf_models.Llama")
    def test_custom_parameters(self, mock_llama, tmp_path):
        """Should pass custom parameters to Llama."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"fake")

        LocalGGUFModel(str(model_file), n_ctx=4096, n_threads=8, verbose=True)

        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_ctx"] == 4096
        assert call_kwargs["n_threads"] == 8
        assert call_kwargs["verbose"] is True

    @patch("src.gguf_models.Llama")
    def test_auto_thread_count(self, mock_llama, tmp_path):
        """Should auto-detect thread count when not specified."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"fake")

        LocalGGUFModel(str(model_file))

        call_kwargs = mock_llama.call_args[1]
        expected_threads = max(1, (os.cpu_count() or 4) // 2)
        assert call_kwargs["n_threads"] == expected_threads

    @patch("src.gguf_models.Llama")
    def test_generate_returns_text(self, mock_llama, tmp_path):
        """Should return generated text from model."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"fake")

        # Mock the Llama __call__ response
        mock_instance = MagicMock()
        mock_instance.return_value = {
            "choices": [{"text": "  This is the answer.  "}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_llama.return_value = mock_instance

        model = LocalGGUFModel(str(model_file))
        result = model.generate("What is the policy?")

        assert result == "This is the answer."
        mock_instance.assert_called_once()

    @patch("src.gguf_models.Llama")
    def test_generate_with_metadata(self, mock_llama, tmp_path):
        """Should return text plus token usage metadata."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"fake")

        mock_instance = MagicMock()
        mock_instance.return_value = {
            "choices": [{"text": "Answer here"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }
        mock_llama.return_value = mock_instance

        model = LocalGGUFModel(str(model_file))
        result = model.generate_with_metadata("prompt")

        assert result["text"] == "Answer here"
        assert result["prompt_tokens"] == 50
        assert result["completion_tokens"] == 10
        assert result["total_tokens"] == 60

    @patch("src.gguf_models.Llama")
    def test_generate_custom_params(self, mock_llama, tmp_path):
        """Should pass temperature and max_tokens to Llama."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"fake")

        mock_instance = MagicMock()
        mock_instance.return_value = {
            "choices": [{"text": "ok"}],
            "usage": {},
        }
        mock_llama.return_value = mock_instance

        model = LocalGGUFModel(str(model_file))
        model.generate("test", max_tokens=100, temperature=0.7, top_p=0.95)

        call_kwargs = mock_instance.call_args[1]
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.95

    @patch("src.gguf_models.Llama")
    def test_model_name_property(self, mock_llama, tmp_path):
        """Should return filename without extension."""
        model_file = tmp_path / "phi-3-mini-q4_k_m.gguf"
        model_file.write_bytes(b"fake")

        model = LocalGGUFModel(str(model_file))
        assert model.model_name == "phi-3-mini-q4_k_m"


class TestListAvailableModels:
    """Test the model discovery utility."""

    def test_empty_directory(self, tmp_path):
        """Should return empty list when no models exist."""
        result = list_available_models(str(tmp_path))
        assert result == []

    def test_nonexistent_directory(self):
        """Should return empty list for missing directory."""
        result = list_available_models("nonexistent_dir_xyz")
        assert result == []

    def test_finds_gguf_files(self, tmp_path):
        """Should find .gguf files in directory."""
        (tmp_path / "model_a.gguf").write_bytes(b"a")
        (tmp_path / "model_b.gguf").write_bytes(b"b")
        (tmp_path / "notes.txt").write_bytes(b"not a model")

        result = list_available_models(str(tmp_path))
        assert len(result) == 2
        names = [p.name for p in result]
        assert "model_a.gguf" in names
        assert "model_b.gguf" in names
