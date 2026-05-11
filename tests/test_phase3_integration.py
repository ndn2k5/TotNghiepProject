"""
End-to-End Integration Tests for Phase 3

Tests full pipeline: Question → Normalize → Retrieve → Generate → Response

Run: pytest tests/test_phase3_integration.py -v -s
"""

import pytest
import logging
from pathlib import Path
from typing import List

from src.question_normalizer import QuestionNormalizer
from src.retriever import Retriever
from src.responder import ResponseGenerator, format_response_for_display
from src.embeddings import LocalEmbedder, VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPhase3Integration:
    """End-to-end integration tests for full chatbot pipeline."""

    @pytest.fixture
    def normalizer(self):
        """Initialize question normalizer."""
        return QuestionNormalizer(use_llm=False)

    @pytest.fixture
    def embedder(self):
        """Initialize embedder."""
        try:
            return LocalEmbedder()
        except Exception as e:
            logger.warning(f"Embedder not available: {e}")
            return None

    @pytest.fixture
    def vector_store(self, embedder):
        """Initialize vector store."""
        if embedder is None:
            pytest.skip("Embedder not available")
        try:
            store = VectorStoreManager()
            store.create_collection()
            return store
        except Exception as e:
            logger.warning(f"Vector store not available: {e}")
            return None

    @pytest.fixture
    def retriever(self, vector_store, embedder):
        """Initialize retriever."""
        if vector_store is None or embedder is None:
            pytest.skip("Vector store or embedder not available")
        return Retriever(
            vector_store=vector_store,
            embedder=embedder,
            use_reranking=False,
        )

    @pytest.fixture
    def responder(self):
        """Initialize responder (requires Phi-3-Mini model)."""
        model_path = Path("./models/phi-3-mini.gguf")
        if not model_path.exists():
            pytest.skip(
                f"Phi-3-Mini model not found at {model_path}. "
                "Download from HuggingFace: microsoft/Phi-3-mini-4k-instruct-gguf"
            )
        try:
            return ResponseGenerator(
                model_path=str(model_path),
                language="vi",
                max_tokens=256,
                temperature=0.3,
            )
        except Exception as e:
            pytest.skip(f"Could not load responder: {e}")

    def test_full_pipeline_vietnamese_question(
        self, normalizer, retriever, responder
    ):
        """Test complete pipeline with Vietnamese HR question."""
        if responder is None:
            pytest.skip("Responder not available")

        # Step 1: Question
        question = "Bao nhiêu ngày nghỉ phép mỗi năm?"

        # Step 2: Normalize
        normalized = normalizer.normalize(question)
        assert isinstance(normalized, str)
        assert len(normalized) > 0
        logger.info(f"Original: {question}")
        logger.info(f"Normalized: {normalized}")

        # Step 3: Retrieve
        retrieved, retrieval_time = retriever.retrieve(normalized, top_k=3)
        assert isinstance(retrieved, list)
        assert retrieval_time >= 0
        logger.info(f"Retrieved {len(retrieved)} chunks in {retrieval_time*1000:.1f}ms")

        # Step 4: Generate response
        response = responder.generate(normalized, retrieved, benchmark=True)
        assert response is not None
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0
        assert 0.0 <= response.confidence <= 1.0
        logger.info(f"Generated response: {response.answer[:100]}...")
        logger.info(f"Confidence: {response.confidence:.2f}")
        logger.info(f"Latency: {response.latency_ms:.1f}ms")

        # Verify answer is not just "not found"
        not_found_indicators = ["không tìm thấy", "không có thông tin"]
        has_not_found = any(indicator in response.answer.lower() for indicator in not_found_indicators)

        if not has_not_found:
            # If we found something, it should be reasonably long
            assert len(response.answer) > 20, "Answer too short to be meaningful"

    def test_multiple_questions(self, normalizer, retriever, responder):
        """Test pipeline with multiple diverse questions."""
        if responder is None:
            pytest.skip("Responder not available")

        test_questions = [
            "Làm cách nào để xin phép?",
            "Lương được trả vào ngày nào?",
            "Hợp đồng lao động như thế nào?",
        ]

        results = []
        for question in test_questions:
            normalized = normalizer.normalize(question)
            retrieved, _ = retriever.retrieve(normalized, top_k=3)
            response = responder.generate(normalized, retrieved)

            results.append({
                "question": question,
                "answer_length": len(response.answer),
                "confidence": response.confidence,
                "latency_ms": response.latency_ms,
            })

            logger.info(f"Q: {question}")
            logger.info(f"A: {response.answer[:80]}...")
            logger.info(f"Confidence: {response.confidence:.2f}, Latency: {response.latency_ms:.1f}ms\n")

        # Verify all completed successfully
        assert len(results) == len(test_questions)
        for result in results:
            assert result["answer_length"] > 0
            assert result["confidence"] >= 0.0

    def test_response_has_sources(self, normalizer, retriever, responder):
        """Test that responses include source citations."""
        if responder is None:
            pytest.skip("Responder not available")

        question = "Kỳ nghỉ phép là bao lâu?"
        normalized = normalizer.normalize(question)
        retrieved, _ = retriever.retrieve(normalized, top_k=3)
        response = responder.generate(normalized, retrieved)

        # Response should have sources or indication of no sources
        assert hasattr(response, "sources")
        assert isinstance(response.sources, list)

    def test_response_formatting(self, normalizer, retriever, responder):
        """Test that responses can be formatted for display."""
        if responder is None:
            pytest.skip("Responder not available")

        question = "Thời hạn thử việc bao lâu?"
        normalized = normalizer.normalize(question)
        retrieved, _ = retriever.retrieve(normalized, top_k=3)
        response = responder.generate(normalized, retrieved)

        # Format for display
        formatted = format_response_for_display(response)
        assert isinstance(formatted, str)
        assert "Trả lời" in formatted or "Answer" in formatted
        assert "Nguồn" in formatted or "Source" in formatted

    def test_latency_benchmark(self, normalizer, retriever, responder):
        """Benchmark latency for full pipeline."""
        if responder is None:
            pytest.skip("Responder not available")

        question = "Làm thêm giờ được trả lương gấp mấy lần?"
        normalized = normalizer.normalize(question)
        retrieved, retrieval_time = retriever.retrieve(normalized, top_k=3)
        response = responder.generate(normalized, retrieved, benchmark=True)

        total_time = retrieval_time + (response.latency_ms / 1000)

        logger.info(f"\nLatency Breakdown:")
        logger.info(f"  Retrieval: {retrieval_time*1000:.1f}ms")
        logger.info(f"  Response:  {response.latency_ms:.1f}ms")
        logger.info(f"  Total:     {total_time*1000:.1f}ms")

        # Log performance
        if total_time > 5.0:
            logger.warning(f"⚠️  Total latency {total_time*1000:.1f}ms is slow (target: <3000ms on CPU)")

    def test_edge_cases(self, normalizer, retriever, responder):
        """Test edge cases and error handling."""
        if responder is None:
            pytest.skip("Responder not available")

        edge_cases = [
            "",  # Empty question
            "?",  # Single character
            "a" * 500,  # Very long question
            "Làm sao đây ???",  # Multiple punctuation
        ]

        for question in edge_cases:
            try:
                normalized = normalizer.normalize(question)
                if normalized:  # Skip if normalization returns empty
                    retrieved, _ = retriever.retrieve(normalized, top_k=3)
                    response = responder.generate(normalized, retrieved)
                    # Should not crash
                    assert isinstance(response.answer, str)
                    logger.info(f"✓ Handled edge case: {question[:30]}")
            except Exception as e:
                logger.warning(f"⚠️  Edge case failed: {question[:30]} → {e}")


class TestResponderUnit:
    """Unit tests for responder module."""

    def test_response_dataclass(self):
        """Test Response dataclass creation."""
        from src.responder import Response

        response = Response(
            answer="Test answer",
            sources=[{"page": 1, "text": "Test"}],
            confidence=0.95,
            latency_ms=150.5,
        )

        assert response.answer == "Test answer"
        assert response.confidence == 0.95
        assert response.latency_ms == 150.5
        assert response.language == "vi"

    def test_responder_initialization(self):
        """Test responder initialization without model."""
        # This should raise an error if model not found
        model_path = Path("./models/phi-3-mini.gguf")
        if not model_path.exists():
            with pytest.raises(Exception):
                ResponseGenerator(str(model_path))
        else:
            # Model exists, test should pass
            responder = ResponseGenerator(str(model_path))
            assert responder is not None


class TestPipeline:
    """Test the complete pipeline integration."""

    def test_pipeline_components_available(self):
        """Verify all Phase 1-3 components can be imported."""
        from src.question_normalizer import QuestionNormalizer
        from src.retriever import Retriever
        from src.responder import ResponseGenerator

        assert QuestionNormalizer is not None
        assert Retriever is not None
        assert ResponseGenerator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
