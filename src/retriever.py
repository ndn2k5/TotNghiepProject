# -*- coding: utf-8 -*-
"""
Retriever: Semantic search + optional re-ranking for HR policy questions.

Responsibilities:
  1. Embed normalized questions using all-MiniLM-L6-v2
  2. Retrieve top-k relevant chunks from ChromaDB
  3. Optional: Re-rank results using Qwen model for relevance scoring
  4. Return ranked results with confidence scores

All processing is local — no external API calls.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from src.embeddings import LocalEmbedder, VectorStoreManager
from src.gguf_models import LocalGGUFModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result (chunk + metadata + score)."""

    text: str
    metadata: Dict
    distance: float
    rerank_score: Optional[float] = None
    source_info: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def get_combined_score(self) -> float:
        """
        Combine semantic distance + re-rank score for final ranking.

        Lower distance + higher rerank score = higher combined score.
        """
        # Normalize distance to [0, 1] (ChromaDB returns 0-1 range for cosine distance)
        semantic_score = 1 - self.distance  # Invert: higher is better

        if self.rerank_score is not None:
            # Weighted average: 60% semantic, 40% re-ranking
            combined = 0.6 * semantic_score + 0.4 * self.rerank_score
        else:
            combined = semantic_score

        return combined


class Retriever:
    """Semantic retrieval with optional re-ranking."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedder: LocalEmbedder,
        rerank_model_path: Optional[str] = None,
        use_reranking: bool = True,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: Initialized VectorStoreManager instance
            embedder: Initialized LocalEmbedder instance
            rerank_model_path: Path to Qwen GGUF model for re-ranking (optional)
            use_reranking: Whether to apply re-ranking if model is available
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.use_reranking = use_reranking and rerank_model_path is not None
        self.reranker = None

        if self.use_reranking:
            try:
                logger.info("Loading Qwen model for re-ranking...")
                self.reranker = LocalGGUFModel(rerank_model_path, n_ctx=512, verbose=False)
                logger.info("✓ Qwen re-ranking model loaded.")
            except Exception as e:
                logger.warning(f"Could not load re-ranking model: {e}. Re-ranking disabled.")
                self.use_reranking = False

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        rerank_top_k: Optional[int] = None,
        benchmark: bool = False,
    ) -> Tuple[List[RetrievalResult], float]:
        """
        Retrieve relevant chunks for a query with optional re-ranking.

        Args:
            query: Normalized question
            top_k: Number of top results to return from semantic search
            rerank_top_k: Number of results to apply re-ranking to
                          (defaults to top_k). Re-ranking is expensive; limit this.
            benchmark: If True, return elapsed time

        Returns:
            Tuple of (list of RetrievalResult objects, elapsed time in seconds)
        """
        start_time = time.time()

        # Step 1: Semantic search
        try:
            raw_results = self.vector_store.query(
                query_text=query,
                embedder=self.embedder,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return [], 0.0

        # Convert to RetrievalResult objects
        results = []
        for res in raw_results:
            result = RetrievalResult(
                text=res["text"],
                metadata=res["metadata"],
                distance=res["distance"],
                source_info=self._format_source(res["metadata"]),
            )
            results.append(result)

        # Step 2: Optional re-ranking
        if self.use_reranking and self.reranker:
            rerank_limit = rerank_top_k or top_k
            results = self._rerank_results(query, results[:rerank_limit])

        # Step 3: Sort by combined score (if re-ranked) or distance (if not)
        if self.use_reranking and any(r.rerank_score is not None for r in results):
            results.sort(key=lambda r: r.get_combined_score(), reverse=True)
        else:
            results.sort(key=lambda r: r.distance)

        elapsed = time.time() - start_time

        if benchmark:
            logger.info(
                f"Retrieval completed in {elapsed:.3f}s "
                f"({len(results)} results, re-ranking={'yes' if self.use_reranking else 'no'})"
            )

        return results, elapsed

    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Use Qwen model to score result relevance to the query.

        Args:
            query: User question
            results: Initial retrieval results

        Returns:
            Results with rerank_score populated
        """
        if not self.reranker or not results:
            return results

        reranked = []
        for i, result in enumerate(results):
            try:
                score = self._score_relevance(query, result.text)
                result.rerank_score = score
                reranked.append(result)
            except Exception as e:
                logger.debug(f"Re-ranking result {i} failed: {e}. Keeping original.")
                reranked.append(result)

        return reranked

    def _score_relevance(self, query: str, chunk: str) -> float:
        """
        Use LLM to score how relevant a chunk is to the query.

        Returns:
            Score in range [0, 1]
        """
        if not self.reranker:
            return 0.5

        prompt = f"""Đánh giá mức độ liên quan của đoạn văn bản sau đến câu hỏi. 
Trả lời chỉ với một con số từ 0 đến 10 (10 là rất liên quan, 0 là hoàn toàn không liên quan).

Câu hỏi: {query}

Đoạn văn bản: {chunk[:300]}

Mức độ liên quan (0-10):"""

        try:
            output = self.reranker.generate(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1,
            )
            # Extract the score
            score_text = output.strip().split()[0]
            score = float(score_text) / 10.0  # Normalize to [0, 1]
            return min(1.0, max(0.0, score))  # Clamp to [0, 1]
        except Exception as e:
            logger.debug(f"Score extraction failed: {e}")
            return 0.5  # Default neutral score

    def _format_source(self, metadata: Dict) -> str:
        """Format metadata into a human-readable source string."""
        page = metadata.get("page_num", "?")
        source = metadata.get("source", "handbook")
        return f"{source} (page {page})"

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 3,
        rerank_top_k: Optional[int] = None,
    ) -> Dict[str, Tuple[List[RetrievalResult], float]]:
        """
        Retrieve results for multiple queries.

        Args:
            queries: List of normalized questions
            top_k: Results per query
            rerank_top_k: Re-ranking limit per query

        Returns:
            Dict mapping query → (results, elapsed_time)
        """
        results = {}
        for query in queries:
            results[query], elapsed = self.retrieve(query, top_k, rerank_top_k, benchmark=True)
            results[query] = (results[query], elapsed)
        return results


def create_retriever(
    vector_store: VectorStoreManager,
    embedder: LocalEmbedder,
    rerank_model_path: Optional[str] = None,
) -> Retriever:
    """
    Factory function to create a retriever.

    Args:
        vector_store: Initialized vector store
        embedder: Initialized embedder
        rerank_model_path: Optional path to re-ranking model

    Returns:
        Retriever instance
    """
    return Retriever(
        vector_store=vector_store,
        embedder=embedder,
        rerank_model_path=rerank_model_path,
        use_reranking=bool(rerank_model_path),
    )


if __name__ == "__main__":
    # Smoke test: Verify RetrievalResult works
    test_result = RetrievalResult(
        text="Test chunk",
        metadata={"page_num": 1},
        distance=0.15,
        rerank_score=0.9,
    )
    print(f"RetrievalResult: {test_result.to_dict()}")
    print(f"Combined score: {test_result.get_combined_score():.3f}")
    print("✓ Retriever module initialized successfully.")
