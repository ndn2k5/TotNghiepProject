"""Unit tests for embeddings module (Task 4) — fully local, no API."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import LocalEmbedder, VectorStoreManager
from src.pdf_extraction import PDFExtractor
from src.chunking import chunk_pages

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "sample_handbook.pdf"

# Shared embedder (loaded once per test session for speed)
@pytest.fixture(scope="session")
def embedder():
    return LocalEmbedder()


class TestLocalEmbedder:
    """Tests for LocalEmbedder — CPU-only, no API."""

    def test_model_loads(self, embedder):
        """Model should load without errors."""
        assert embedder.model is not None

    def test_dimension_is_384(self, embedder):
        """all-MiniLM-L6-v2 outputs 384-dim vectors."""
        assert embedder.dimension == 384

    def test_embed_single_text(self, embedder):
        """Single text should return one 384-dim vector."""
        vectors = embedder.embed(["Hello world"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 384

    def test_embed_multiple_texts(self, embedder):
        """Multiple texts should return matching number of vectors."""
        texts = ["Vacation policy.", "Sick leave rules.", "Overtime pay."]
        vectors = embedder.embed(texts)
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_embed_returns_floats(self, embedder):
        """Each element of the vector should be a float."""
        vectors = embedder.embed(["Test sentence."])
        assert all(isinstance(x, float) for x in vectors[0])

    def test_different_texts_produce_different_vectors(self, embedder):
        """Semantically different texts should have different embeddings."""
        v1 = embedder.embed(["Employee vacation policy"])[0]
        v2 = embedder.embed(["Database connection error"])[0]
        assert v1 != v2


class TestVectorStoreManager:
    """Tests for ChromaDB integration — persistent local DB."""

    def test_collection_created(self, embedder, tmp_path):
        """A collection should be created in the tmp directory."""
        vsm = VectorStoreManager(persist_dir=str(tmp_path / "chroma_test"))
        col = vsm.create_collection(name="test_col")
        assert col is not None
        assert vsm.count() == 0

    def test_add_and_count_chunks(self, embedder, tmp_path):
        """Adding chunks should increase the document count."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
        chunks = chunk_pages(pages, chunk_size=300)

        vsm = VectorStoreManager(persist_dir=str(tmp_path / "chroma_add"))
        vsm.create_collection(name="add_test")
        vsm.add_chunks(chunks, embedder)
        assert vsm.count() == len(chunks)

    def test_query_returns_results(self, embedder, tmp_path):
        """Query should return top-k results with required fields."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
        chunks = chunk_pages(pages, chunk_size=300)

        vsm = VectorStoreManager(persist_dir=str(tmp_path / "chroma_query"))
        vsm.create_collection(name="query_test")
        vsm.add_chunks(chunks, embedder)

        results = vsm.query("How many vacation days do employees get?", embedder, top_k=2)
        assert len(results) == 2
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "distance" in r

    def test_query_vacation_relevance(self, embedder, tmp_path):
        """Query about vacation should retrieve vacation-related text."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
        chunks = chunk_pages(pages, chunk_size=300)

        vsm = VectorStoreManager(persist_dir=str(tmp_path / "chroma_relevance"))
        vsm.create_collection(name="relevance_test")
        vsm.add_chunks(chunks, embedder)

        results = vsm.query("vacation days per year", embedder, top_k=1)
        top_text = results[0]["text"].lower()
        assert "vacation" in top_text or "leave" in top_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
