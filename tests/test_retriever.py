"""
Tests for retriever.py

Coverage:
  - RetrievalResult dataclass
  - Semantic retrieval (without LLM re-ranking)
  - Re-ranking score calculation
  - Batch retrieval
  - Error handling
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.retriever import Retriever, RetrievalResult, create_retriever
from src.embeddings import LocalEmbedder, VectorStoreManager


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""

    def test_creation(self):
        """Test RetrievalResult creation."""
        result = RetrievalResult(
            text="Sample chunk",
            metadata={"page_num": 1},
            distance=0.15,
        )
        assert result.text == "Sample chunk"
        assert result.metadata["page_num"] == 1
        assert result.distance == 0.15
        assert result.rerank_score is None

    def test_to_dict(self):
        """Test RetrievalResult.to_dict() conversion."""
        result = RetrievalResult(
            text="Test",
            metadata={"page": 1},
            distance=0.2,
            rerank_score=0.85,
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["text"] == "Test"
        assert result_dict["distance"] == 0.2
        assert result_dict["rerank_score"] == 0.85

    def test_get_combined_score_semantic_only(self):
        """Test combined score calculation without re-ranking."""
        result = RetrievalResult(
            text="Test",
            metadata={},
            distance=0.2,  # distance=0.2 → semantic_score=0.8
            rerank_score=None,
        )
        score = result.get_combined_score()
        assert abs(score - 0.8) < 0.01  # Should equal 1 - 0.2

    def test_get_combined_score_with_reranking(self):
        """Test combined score calculation with re-ranking."""
        result = RetrievalResult(
            text="Test",
            metadata={},
            distance=0.2,  # semantic_score = 0.8
            rerank_score=0.9,  # rerank_score = 0.9
        )
        score = result.get_combined_score()
        # Combined: 0.6 * 0.8 + 0.4 * 0.9 = 0.48 + 0.36 = 0.84
        assert abs(score - 0.84) < 0.01

    def test_get_combined_score_clamps(self):
        """Test that combined score is within [0, 1]."""
        result = RetrievalResult(
            text="Test",
            metadata={},
            distance=0.0,  # semantic_score = 1.0
            rerank_score=1.0,  # rerank_score = 1.0
        )
        score = result.get_combined_score()
        assert 0.0 <= score <= 1.0


class TestRetrieverInitialization:
    """Test Retriever initialization."""

    def test_init_without_reranking(self):
        """Test initialization without re-ranking model."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_reranking=False,
        )
        assert retriever.vector_store is mock_store
        assert retriever.embedder is mock_embedder
        assert retriever.use_reranking is False
        assert retriever.reranker is None

    def test_init_with_invalid_rerank_model(self):
        """Test initialization with non-existent re-rank model."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            rerank_model_path="/nonexistent/model.gguf",
            use_reranking=True,
        )
        # Should fall back: use_reranking False, reranker None
        assert retriever.use_reranking is False or retriever.reranker is None


class TestSemanticRetrieval:
    """Test semantic retrieval (without re-ranking)."""

    def test_retrieve_success(self):
        """Test successful retrieval."""
        # Mock vector store
        mock_store = Mock(spec=VectorStoreManager)
        mock_store.query.return_value = [
            {
                "text": "Vacation policy is 20 days per year",
                "metadata": {"page_num": 1},
                "distance": 0.1,
            },
            {
                "text": "Additional leave available",
                "metadata": {"page_num": 2},
                "distance": 0.25,
            },
        ]

        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_reranking=False,
        )

        results, elapsed = retriever.retrieve("How much vacation?")

        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].text == "Vacation policy is 20 days per year"
        assert elapsed >= 0.0

    def test_retrieve_sorted_by_distance(self):
        """Test that results are sorted by distance when no re-ranking."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_store.query.return_value = [
            {"text": "Far chunk", "metadata": {}, "distance": 0.5},
            {"text": "Close chunk", "metadata": {}, "distance": 0.1},
        ]

        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_reranking=False,
        )

        results, _ = retriever.retrieve("Test query")

        # Should be sorted by distance (lower first)
        assert results[0].distance <= results[1].distance

    def test_retrieve_with_benchmark(self):
        """Test retrieval with benchmarking enabled."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_store.query.return_value = []
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_reranking=False,
        )

        results, elapsed = retriever.retrieve("Test", benchmark=True)

        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    def test_retrieve_empty_results(self):
        """Test retrieval when no results found."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_store.query.return_value = []
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_reranking=False,
        )

        results, _ = retriever.retrieve("Obscure question")

        assert len(results) == 0
        assert isinstance(results, list)

    def test_retrieve_error_handling(self):
        """Test retrieval error handling."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_store.query.side_effect = Exception("Database error")
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_reranking=False,
        )

        results, elapsed = retriever.retrieve("Test query")

        # Should gracefully handle error
        assert results == []
        assert elapsed >= 0.0


class TestSourceFormatting:
    """Test source metadata formatting."""

    def test_format_source_with_page(self):
        """Test source formatting with page number."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )

        source = retriever._format_source({"page_num": 3, "source": "handbook"})
        assert "handbook" in source
        assert "3" in source

    def test_format_source_missing_page(self):
        """Test source formatting with missing page number."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )

        source = retriever._format_source({"source": "handbook"})
        assert "?" in source  # Default unknown page


class TestBatchRetrieval:
    """Test batch retrieval."""

    def test_batch_retrieve(self):
        """Test batch retrieval for multiple queries."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_store.query.return_value = [
            {"text": "Result", "metadata": {}, "distance": 0.1}
        ]
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_reranking=False,
        )

        queries = ["Query 1", "Query 2"]
        batch_results = retriever.batch_retrieve(queries)

        assert len(batch_results) == 2
        assert "Query 1" in batch_results
        assert "Query 2" in batch_results

        # Each query should have (results, elapsed_time) tuple
        for query in queries:
            results, elapsed = batch_results[query]
            assert isinstance(results, list)
            assert isinstance(elapsed, float)


class TestFactoryFunction:
    """Test create_retriever factory function."""

    def test_create_retriever_without_reranking(self):
        """Test factory function without re-ranking."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = create_retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )

        assert isinstance(retriever, Retriever)
        assert retriever.use_reranking is False

    def test_create_retriever_with_reranking(self):
        """Test factory function with re-ranking (model not available)."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = create_retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            rerank_model_path="/nonexistent/model.gguf",
        )

        assert isinstance(retriever, Retriever)
        # Re-ranking should be disabled if model not found


class TestRetrieverEdgeCases:
    """Test retriever edge cases."""

    def test_retrieve_with_top_k_zero(self):
        """Test retrieval with top_k=0."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_store.query.return_value = []
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )

        results, _ = retriever.retrieve("Test", top_k=0)
        assert results == []

    def test_retrieve_with_large_top_k(self):
        """Test retrieval with very large top_k."""
        mock_store = Mock(spec=VectorStoreManager)
        mock_store.query.return_value = [
            {"text": f"Chunk {i}", "metadata": {}, "distance": 0.1}
            for i in range(100)
        ]
        mock_embedder = Mock(spec=LocalEmbedder)

        retriever = Retriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )

        results, _ = retriever.retrieve("Test", top_k=100)
        assert len(results) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
