"""
Tests for the RAG Pipeline (retrieval + generation).

Uses mocking for the GGUF model so tests run without a real model file.
Tests retrieval with a real (temporary) ChromaDB instance.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
import tempfile
import shutil

from src.rag_pipeline import RAGPipeline, PROMPT_TEMPLATE_VI, PROMPT_TEMPLATE_EN


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a fake GGUF model file."""
    model_file = tmp_path / "fake_model.gguf"
    model_file.write_bytes(b"fake gguf data")
    return str(model_file)


@pytest.fixture
def chroma_dir(tmp_path):
    """Create a temporary ChromaDB directory."""
    db_dir = tmp_path / "test_chroma"
    db_dir.mkdir()
    return str(db_dir)


@pytest.fixture
def sample_pdf_path():
    """Path to the test sample PDF."""
    path = Path("data/sample_handbook.pdf")
    if not path.exists():
        pytest.skip("Sample PDF not found at data/sample_handbook.pdf")
    return str(path)


# ── Pipeline Initialization Tests ───────────────────────────────────


class TestPipelineInit:
    """Test RAG pipeline initialization."""

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_pipeline_creates_all_components(self, mock_model_cls, mock_model_path, chroma_dir):
        """Pipeline should initialize embedder, vector store, and LLM."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)

        assert pipeline.embedder is not None
        assert pipeline.vector_store is not None
        assert pipeline.vector_store.collection is not None
        assert pipeline.llm is not None

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_pipeline_vietnamese_prompt(self, mock_model_cls, mock_model_path, chroma_dir):
        """Default language should use Vietnamese prompt."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir, language="vi")
        assert "tiếng Việt" in pipeline.prompt_template

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_pipeline_english_prompt(self, mock_model_cls, mock_model_path, chroma_dir):
        """English language should use English prompt."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir, language="en")
        assert "accurately and concisely" in pipeline.prompt_template


# ── Document Ingestion Tests ────────────────────────────────────────


class TestIngestion:
    """Test PDF ingestion into the vector store."""

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_ingest_pdf(self, mock_model_cls, mock_model_path, chroma_dir, sample_pdf_path):
        """Should ingest PDF and store chunks in ChromaDB."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)
        count = pipeline.ingest_pdf(sample_pdf_path)

        assert count > 0
        assert pipeline.vector_store.count() > 0

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_ingest_pdf_custom_chunk_size(self, mock_model_cls, mock_model_path, chroma_dir, sample_pdf_path):
        """Should respect custom chunk size."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)

        count_small = pipeline.ingest_pdf(sample_pdf_path, chunk_size=100, chunk_overlap=20)
        # More chunks with smaller chunk size
        assert count_small > 0

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_ingest_nonexistent_pdf(self, mock_model_cls, mock_model_path, chroma_dir):
        """Should raise error for missing PDF."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)

        with pytest.raises(FileNotFoundError):
            pipeline.ingest_pdf("nonexistent.pdf")


# ── Retrieval Tests ─────────────────────────────────────────────────


class TestRetrieval:
    """Test context retrieval from vector store."""

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_retrieve_after_ingest(self, mock_model_cls, mock_model_path, chroma_dir, sample_pdf_path):
        """Should retrieve relevant chunks after ingestion."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)
        pipeline.ingest_pdf(sample_pdf_path)

        chunks = pipeline.retrieve("employee handbook", top_k=2)

        assert len(chunks) > 0
        assert len(chunks) <= 2
        assert "text" in chunks[0]
        assert "metadata" in chunks[0]

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_retrieve_empty_store(self, mock_model_cls, mock_model_path, chroma_dir):
        """Should return empty list from empty store."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)
        chunks = pipeline.retrieve("anything")
        assert chunks == []


# ── Prompt Building Tests ───────────────────────────────────────────


class TestPromptBuilding:
    """Test prompt construction."""

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_build_prompt_includes_context(self, mock_model_cls, mock_model_path, chroma_dir):
        """Prompt should include chunk text and question."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)

        chunks = [
            {"text": "Vacation is 15 days per year.", "metadata": {"page_num": 3, "source": "handbook"}},
            {"text": "Sick leave is 10 days.", "metadata": {"page_num": 5, "source": "handbook"}},
        ]

        prompt = pipeline.build_prompt("How many vacation days?", chunks)

        assert "Vacation is 15 days per year." in prompt
        assert "Sick leave is 10 days." in prompt
        assert "How many vacation days?" in prompt
        assert "Trang 3" in prompt
        assert "Trang 5" in prompt

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_build_prompt_numbered_chunks(self, mock_model_cls, mock_model_path, chroma_dir):
        """Prompt should number the context chunks."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)

        chunks = [
            {"text": "Chunk one.", "metadata": {"page_num": 1, "source": "doc"}},
            {"text": "Chunk two.", "metadata": {"page_num": 2, "source": "doc"}},
        ]

        prompt = pipeline.build_prompt("test?", chunks)
        assert "[Đoạn 1" in prompt
        assert "[Đoạn 2" in prompt


# ── Full Answer Pipeline Tests ──────────────────────────────────────


class TestAnswer:
    """Test the full answer() flow with mocked LLM."""

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_answer_empty_store(self, mock_model_cls, mock_model_path, chroma_dir):
        """Should return helpful message when no documents ingested."""
        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)
        result = pipeline.answer("test question")

        assert "question" in result
        assert "answer" in result
        assert "Chưa có tài liệu" in result["answer"]
        assert result["sources"] == []

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_answer_with_context(self, mock_model_cls, mock_model_path, chroma_dir, sample_pdf_path):
        """Should call LLM and return structured result."""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_llm.generate_with_metadata.return_value = {
            "text": "Chính sách nghỉ phép là 15 ngày/năm.",
            "prompt_tokens": 100,
            "completion_tokens": 20,
        }
        mock_llm.model_name = "fake_model"
        mock_model_cls.return_value = mock_llm

        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)
        pipeline.ingest_pdf(sample_pdf_path)

        result = pipeline.answer("Chính sách nghỉ phép?")

        assert result["question"] == "Chính sách nghỉ phép?"
        assert "nghỉ phép" in result["answer"]
        assert len(result["sources"]) > 0
        assert "timing" in result
        assert result["timing"]["total_seconds"] > 0

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_answer_returns_timing(self, mock_model_cls, mock_model_path, chroma_dir, sample_pdf_path):
        """Should include timing breakdown in result."""
        mock_llm = MagicMock()
        mock_llm.generate_with_metadata.return_value = {
            "text": "Answer",
            "prompt_tokens": 50,
            "completion_tokens": 10,
        }
        mock_model_cls.return_value = mock_llm

        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)
        pipeline.ingest_pdf(sample_pdf_path)

        result = pipeline.answer("test")

        timing = result["timing"]
        assert "retrieval_seconds" in timing
        assert "generation_seconds" in timing
        assert "total_seconds" in timing

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_answer_returns_token_usage(self, mock_model_cls, mock_model_path, chroma_dir, sample_pdf_path):
        """Should include token usage in result."""
        mock_llm = MagicMock()
        mock_llm.generate_with_metadata.return_value = {
            "text": "Answer",
            "prompt_tokens": 80,
            "completion_tokens": 15,
        }
        mock_model_cls.return_value = mock_llm

        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)
        pipeline.ingest_pdf(sample_pdf_path)

        result = pipeline.answer("test")

        assert result["token_usage"]["prompt_tokens"] == 80
        assert result["token_usage"]["completion_tokens"] == 15


# ── Stats Tests ─────────────────────────────────────────────────────


class TestStats:
    """Test pipeline statistics."""

    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_get_stats(self, mock_model_cls, mock_model_path, chroma_dir):
        """Should return pipeline stats."""
        mock_llm = MagicMock()
        mock_llm.model_name = "test_model"
        mock_model_cls.return_value = mock_llm

        pipeline = RAGPipeline(mock_model_path, persist_dir=chroma_dir)
        stats = pipeline.get_stats()

        assert stats["model"] == "test_model"
        assert stats["documents_in_store"] == 0
        assert stats["embedding_dimension"] == 384
