"""Unit tests for chunking module (Task 3) — fully local, no API."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_extraction import PDFExtractor
from src.chunking import chunk_text, chunk_pages, experiment_chunk_sizes

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "sample_handbook.pdf"


class TestChunkText:
    """Tests for the chunk_text() function."""

    def test_splits_into_multiple_chunks(self):
        """Long text should be split into more than one chunk."""
        text = "This is a test sentence. " * 50  # ~1200 chars
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        """Chunks should not greatly exceed the target size."""
        text = "Hello world. " * 100
        chunks = chunk_text(text, chunk_size=150, chunk_overlap=20)
        # Allow 20% slack due to sentence boundary handling
        for c in chunks:
            assert len(c) <= 180, f"Chunk too large: {len(c)} chars"

    def test_short_text_stays_as_one_chunk(self):
        """Short text that fits in one chunk should not be split."""
        text = "Short text."
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_empty_text(self):
        """Empty string should return an empty list."""
        chunks = chunk_text("", chunk_size=300)
        assert chunks == []


class TestChunkPages:
    """Tests for chunk_pages() with real PDF data."""

    def test_returns_chunks_with_metadata(self):
        """Each chunk must have required metadata fields."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
        chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=30)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "page_num" in chunk
            assert "chunk_index" in chunk
            assert "chunk_size" in chunk

    def test_page_num_preserved(self):
        """Page numbers from the original PDF should be preserved."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
        chunks = chunk_pages(pages, chunk_size=300)
        page_nums = {c["page_num"] for c in chunks}
        assert 1 in page_nums  # PDF has at least page 1
        assert 2 in page_nums  # PDF has page 2

    def test_chunk_index_sequential(self):
        """chunk_index should start at 0 for each page."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
        chunks = chunk_pages(pages, chunk_size=200)
        page1_chunks = [c for c in chunks if c["page_num"] == 1]
        indices = [c["chunk_index"] for c in page1_chunks]
        assert indices == list(range(len(page1_chunks)))


class TestExperimentChunkSizes:
    """Tests for experiment_chunk_sizes()."""

    def test_all_sizes_present(self):
        """All requested sizes should appear in results dict."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
        results = experiment_chunk_sizes(pages, sizes=[300, 600])
        assert 300 in results
        assert 600 in results

    def test_smaller_size_yields_more_chunks(self):
        """Smaller chunk size should produce more (or equal) chunks."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
        results = experiment_chunk_sizes(pages, sizes=[300, 600, 900])
        assert results[300]["total_chunks"] >= results[600]["total_chunks"]
        assert results[600]["total_chunks"] >= results[900]["total_chunks"]

    def test_result_has_required_keys(self):
        """Each size result should contain expected keys."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
        results = experiment_chunk_sizes(pages, sizes=[600])
        info = results[600]
        assert "total_chunks" in info
        assert "avg_chunk_len" in info
        assert "sample_chunks" in info
        assert info["total_chunks"] > 0
        assert info["avg_chunk_len"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
