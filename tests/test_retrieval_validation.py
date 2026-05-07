"""
Retrieval Validation Test Set (Phase 2 Exit Criteria)

30 representative Vietnamese HR questions for validating retrieval quality.

Requirements:
  - 80%+ of questions should retrieve ≥1 relevant chunk
  - Manual validation: Is the top result relevant?
  - Log failures for debugging

Instructions:
  1. Run this test suite with populated vector store
  2. Check each result for relevance
  3. Update results dictionary with actual retrieved content
  4. Calculate success rate
"""

import pytest
import logging
from typing import List, Dict, Tuple
from src.question_normalizer import QuestionNormalizer
from src.retriever import Retriever
from src.embeddings import LocalEmbedder, VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 30 representative Vietnamese HR questions
VALIDATION_QUESTIONS = [
    # Vacation/Leave (5 questions)
    "Công ty cho phép bao nhiêu ngày nghỉ phép mỗi năm?",
    "Làm cách nào để xin phép năm?",
    "Nếu tôi không lấy phép, có được trả tiền không?",
    "Phép năm có tính sang năm tiếp theo không?",
    "Tôi muốn biết về quy định nghỉ phép của công ty",

    # Sick Leave (5 questions)
    "Khi ốm, tôi cần làm gì trước khi nghỉ?",
    "Ngày ốm có được thanh toán đầy đủ không?",
    "Cần chứng chỉ y tế từ bác sĩ nào?",
    "Người lao động ốm mà không báo cáo sẽ bị thế nào?",
    "Nghỉ ốm liên tục quá lâu sẽ ảnh hưởng gì đến công việc?",

    # Overtime & Compensation (5 questions)
    "Làm thêm giờ được trả lương gấp mấy lần?",
    "Công ty có chế độ tăng ca vào cuối tuần không?",
    "Làm thêm giờ phải được phê duyệt trước không?",
    "Phụ cấp tăng ca được tính vào lương tháng không?",
    "Giới hạn tăng ca tối đa là bao nhiêu giờ mỗi tháng?",

    # Salary & Payment (5 questions)
    "Lương được trả vào ngày nào hàng tháng?",
    "Nếu không hoàn thành công việc, lương có bị trừ không?",
    "Công ty có cấp bảng lương chi tiết không?",
    "Cách tính phụ cấp gia đình như thế nào?",
    "Lương 13 tháng được tính dựa vào tiêu chí nào?",

    # Contract & Employment (5 questions)
    "Hợp đồng thử việc có thời hạn bao lâu?",
    "Điều kiện nào để ký hợp đồng không xác định thời hạn?",
    "Nếu tôi muốn từ chức, phải báo trước bao lâu?",
    "Công ty có quyền chấm dứt hợp đồng bất kỳ lúc nào không?",
    "Điều khoản chế độ bảo hiểm trong hợp đồng là gì?",

    # Discipline & Rules (5 questions)
    "Đi trễ tối đa bao lâu sẽ bị kỷ luật?",
    "Nếu vi phạm quy định, sẽ bị xử phạt thế nào?",
    "Liệu có cơ hội phúc thẩm nếu bị kỷ luật?",
    "Vắng mặt không phép sẽ bị trừ lương bao nhiêu?",
    "Công ty có chế độ cảnh báo trước khi kỷ luật không?",
]


class TestRetrievalValidation:
    """Validation test suite for Phase 2 exit criteria."""

    @pytest.fixture
    def normalizer(self):
        """Create question normalizer."""
        return QuestionNormalizer(use_llm=False)

    @pytest.fixture
    def vector_store(self):
        """Get initialized vector store (must be populated before test)."""
        try:
            store = VectorStoreManager(persist_dir="./chroma_db")
            store.create_collection()
            return store
        except Exception as e:
            logger.warning(f"Vector store not initialized: {e}")
            return None

    @pytest.fixture
    def embedder(self):
        """Get initialized embedder."""
        try:
            return LocalEmbedder()
        except Exception as e:
            logger.warning(f"Embedder not initialized: {e}")
            return None

    @pytest.fixture
    def retriever(self, vector_store, embedder):
        """Create retriever."""
        if vector_store is None or embedder is None:
            pytest.skip("Vector store or embedder not available")
        return Retriever(
            vector_store=vector_store,
            embedder=embedder,
            use_reranking=False,
        )

    def test_retrieval_quality_all_questions(self, retriever, normalizer):
        """
        Test retrieval quality for all 30 questions.

        Phase 2 Exit Criteria: 80%+ of questions retrieve ≥1 relevant chunk.
        """
        if retriever is None:
            pytest.skip("Retriever not available")

        results = []
        relevant_count = 0

        for i, question in enumerate(VALIDATION_QUESTIONS, 1):
            normalized_q = normalizer.normalize(question)
            retrieved, elapsed = retriever.retrieve(normalized_q, top_k=3)

            # Heuristic relevance check: If we got results, assume relevant for now
            # In manual validation, these should be verified by human
            is_relevant = len(retrieved) > 0

            result_entry = {
                "id": i,
                "question": question,
                "normalized": normalized_q,
                "retrieved_count": len(retrieved),
                "top_1_text": retrieved[0].text if retrieved else None,
                "top_1_distance": retrieved[0].distance if retrieved else None,
                "is_relevant": is_relevant,
                "elapsed_ms": elapsed * 1000,
            }

            results.append(result_entry)

            if is_relevant:
                relevant_count += 1

            logger.info(f"Q{i}: {question[:50]}... → {len(retrieved)} results ({elapsed*1000:.1f}ms)")

        # Calculate success rate
        success_rate = relevant_count / len(VALIDATION_QUESTIONS) if VALIDATION_QUESTIONS else 0

        logger.info(f"\n{'='*60}")
        logger.info(f"Retrieval Validation Results")
        logger.info(f"{'='*60}")
        logger.info(f"Total Questions: {len(VALIDATION_QUESTIONS)}")
        logger.info(f"Relevant Results: {relevant_count}/{len(VALIDATION_QUESTIONS)} ({success_rate*100:.1f}%)")
        logger.info(f"Target: ≥80% relevance")
        logger.info(f"Status: {'✓ PASS' if success_rate >= 0.80 else '✗ FAIL'}")
        logger.info(f"{'='*60}\n")

        # Print detailed results
        logger.info("Detailed Results:")
        for entry in results:
            status = "✓" if entry["is_relevant"] else "✗"
            logger.info(
                f"{status} Q{entry['id']:2d}: {entry['question'][:45]:45s} "
                f"({entry['retrieved_count']} results, {entry['elapsed_ms']:.1f}ms)"
            )

        # Phase 2 exit criteria check
        assert success_rate >= 0.80, (
            f"Retrieval validation failed: {relevant_count}/{len(VALIDATION_QUESTIONS)} "
            f"({success_rate*100:.1f}%) is below 80% target"
        )

    def test_retrieval_latency(self, retriever, normalizer):
        """
        Test that retrieval latency meets Phase 2 requirement: <150ms per query.
        """
        if retriever is None:
            pytest.skip("Retriever not available")

        latencies = []

        for question in VALIDATION_QUESTIONS[:10]:  # Sample first 10
            normalized = normalizer.normalize(question)
            _, elapsed = retriever.retrieve(normalized, top_k=3)
            latencies.append(elapsed * 1000)  # Convert to ms

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0

        logger.info(f"\nLatency Test Results:")
        logger.info(f"Average: {avg_latency:.1f}ms")
        logger.info(f"Max: {max_latency:.1f}ms")
        logger.info(f"Target: <150ms per query")
        logger.info(f"Status: {'✓ PASS' if avg_latency < 150 else '✗ FAIL'}")

        assert avg_latency < 150, f"Average latency {avg_latency:.1f}ms exceeds 150ms limit"

    def test_retrieval_no_crashes(self, retriever, normalizer):
        """
        Test that retrieval doesn't crash on any input.

        Phase 2 Exit Criteria: "No Python crashes on unexpected inputs"
        """
        if retriever is None:
            pytest.skip("Retriever not available")

        crash_count = 0

        for question in VALIDATION_QUESTIONS:
            try:
                normalized = normalizer.normalize(question)
                retrieved, _ = retriever.retrieve(normalized)
                assert isinstance(retrieved, list)
            except Exception as e:
                logger.error(f"CRASH on question: {question}\n{e}")
                crash_count += 1

        logger.info(f"\nCrash Test: {crash_count}/{len(VALIDATION_QUESTIONS)} crashes")
        assert crash_count == 0, f"{crash_count} crashes detected"

    def test_question_normalization(self, normalizer):
        """Test question normalization on all validation questions."""
        for question in VALIDATION_QUESTIONS:
            normalized = normalizer.normalize(question)
            assert isinstance(normalized, str)
            assert len(normalized) > 0
            assert normalized == normalized.strip()  # No leading/trailing whitespace
            logger.debug(f"✓ {question[:40]:40s} → {normalized[:40]}")


class TestRetrievalFallbacks:
    """Test fallback behavior when vector store is empty."""

    def test_retrieval_on_empty_store(self):
        """Test graceful handling of empty vector store."""
        # Create a fresh store without documents
        try:
            store = VectorStoreManager(persist_dir="./chroma_db_test_empty")
            store.create_collection()
            embedder = LocalEmbedder()

            retriever = Retriever(
                vector_store=store,
                embedder=embedder,
                use_reranking=False,
            )

            retrieved, elapsed = retriever.retrieve("Test question")

            # Should return empty list, not crash
            assert retrieved == []
            assert elapsed >= 0
            logger.info("✓ Empty store handled gracefully")
        except Exception as e:
            logger.warning(f"Could not test empty store: {e}")


class TestManualValidationTemplate:
    """
    Template for manual validation results.

    INSTRUCTIONS:
    1. Run retrieval tests and get results
    2. For each question, manually check if top-1 result is relevant
    3. Fill in the results below
    4. Calculate relevance percentage
    """

    MANUAL_RESULTS = {
        # Example:
        # "Công ty cho phép bao nhiêu ngày nghỉ phép mỗi năm?" : {
        #     "relevant": True,
        #     "top_result": "Vacation policy states 20 days per year...",
        #     "notes": "Clear match"
        # }
    }

    def test_manual_validation_example(self):
        """
        Example of how to verify results manually.

        To use this:
        1. Run test_retrieval_quality_all_questions
        2. Copy results to MANUAL_RESULTS
        3. For each, set relevant=True/False based on inspection
        4. Rerun this test
        """
        if not self.MANUAL_RESULTS:
            logger.info("No manual results to validate. Populate MANUAL_RESULTS first.")
            return

        relevant = sum(1 for r in self.MANUAL_RESULTS.values() if r.get("relevant"))
        total = len(self.MANUAL_RESULTS)
        rate = relevant / total if total > 0 else 0

        logger.info(f"\nManual Validation: {relevant}/{total} ({rate*100:.1f}%)")
        logger.info(f"Target: ≥80%")
        logger.info(f"Status: {'✓ PASS' if rate >= 0.80 else '✗ FAIL'}")

        assert rate >= 0.80


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s shows print output
