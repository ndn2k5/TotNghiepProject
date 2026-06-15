# -*- coding: utf-8 -*-
"""
Hybrid BM25 + Vector retrieval with Reciprocal Rank Fusion (RRF).

Problem solved: pure vector search fails on short Vietnamese queries where
keyword overlap is more reliable than embedding similarity.

BM25 (keyword)  +  Vector (semantic)  →  RRF merge  →  better top-k

Usage:
    from src.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(vector_store, embedder)
    results = retriever.retrieve("nghỉ phép mỗi năm bao nhiêu ngày", top_k=5)
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def tokenize_vi(text: str) -> List[str]:
    """
    Simple Vietnamese tokenizer.
    Splits on whitespace + removes punctuation.
    Good enough for BM25; Vietnamese is space-delimited unlike Chinese/Japanese.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


class HybridRetriever:
    """
    Combines BM25 keyword search + vector semantic search via Reciprocal Rank Fusion.

    RRF formula: score(doc) = Σ 1 / (k + rank(doc))  where k=60 (standard default)
    Final rank = alpha * vector_rrf + (1-alpha) * bm25_rrf

    Falls back to pure vector search if rank_bm25 is not installed.
    """

    def __init__(self, vector_store, embedder, alpha: float = 0.5, rrf_k: int = 60):
        """
        Args:
            vector_store: VectorStoreManager instance
            embedder: LocalEmbedder instance
            alpha: Weight for vector vs BM25. 0.0=pure BM25, 1.0=pure vector, 0.5=equal
            rrf_k: RRF constant (60 is the standard value from the original paper)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha
        self.rrf_k = rrf_k

        self._bm25 = None
        self._corpus: Optional[List[tuple]] = None  # (id, text, metadata)

    # ── BM25 index management ────────────────────────────────────────

    def _build_bm25_index(self) -> None:
        """Load all ChromaDB chunks and build BM25 index. Cached after first call."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 not installed — falling back to vector-only search.\n"
                "Install with: pip install rank-bm25"
            )
            self._corpus = []
            return

        result = self.vector_store.collection.get(include=["documents", "metadatas"])
        docs = result.get("documents") or []
        ids = result.get("ids") or []
        metas = result.get("metadatas") or []

        if not docs:
            logger.warning("ChromaDB empty — BM25 index not built.")
            self._corpus = []
            return

        self._corpus = list(zip(ids, docs, metas))
        tokenized = [tokenize_vi(doc) for doc in docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index ready: {len(docs)} chunks indexed.")

    def invalidate(self) -> None:
        """Call after re-indexing ChromaDB to force BM25 rebuild on next query."""
        self._bm25 = None
        self._corpus = None

    # ── RRF helpers ──────────────────────────────────────────────────

    def _rrf(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank + 1)

    # ── Main retrieval ───────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Hybrid retrieval: BM25 + vector, merged via RRF.

        Args:
            query: User question (Vietnamese or English)
            top_k: Number of chunks to return

        Returns:
            List of dicts: {text, metadata, distance, rrf_score}
            Sorted by descending RRF score (best match first).
        """
        # Lazy index build
        if self._corpus is None:
            self._build_bm25_index()

        n = self.vector_store.count()
        if n == 0:
            return []

        fetch_k = min(top_k * 4, n)

        # ── Step 1: Vector search ────────────────────────────────────
        vector_results = self.vector_store.query(query, self.embedder, top_k=fetch_k)
        # Map text → (rank, result_dict)
        vector_rank_map: Dict[str, int] = {r["text"]: rank for rank, r in enumerate(vector_results)}

        # ── Step 2: BM25 search ──────────────────────────────────────
        bm25_rank_map: Dict[str, int] = {}
        if self._bm25 is not None and self._corpus:
            tokens = tokenize_vi(query)
            scores = self._bm25.get_scores(tokens)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, idx in enumerate(ranked_indices[:fetch_k]):
                _, text, _ = self._corpus[idx]
                bm25_rank_map[text] = rank

        # ── Step 3: RRF fusion ───────────────────────────────────────
        all_texts = set(vector_rank_map) | set(bm25_rank_map)

        rrf_scores: Dict[str, float] = {}
        for text in all_texts:
            v_rank = vector_rank_map.get(text, fetch_k)  # worst rank if not in vector results
            b_rank = bm25_rank_map.get(text, fetch_k)    # worst rank if not in BM25 results
            rrf_scores[text] = (
                self.alpha * self._rrf(v_rank) +
                (1 - self.alpha) * self._rrf(b_rank)
            )

        top_texts = sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True)[:top_k]

        # ── Step 4: Build result dicts ───────────────────────────────
        # Prefer vector result dicts (have distance); fall back to corpus metadata
        text_to_vresult = {r["text"]: r for r in vector_results}
        text_to_corpus = {text: (cid, text, meta) for cid, text, meta in (self._corpus or [])}

        results = []
        for text in top_texts:
            if text in text_to_vresult:
                entry = dict(text_to_vresult[text])
            elif text in text_to_corpus:
                _, t, meta = text_to_corpus[text]
                entry = {"text": t, "metadata": meta, "distance": 1.0}
            else:
                continue
            entry["rrf_score"] = round(rrf_scores[text], 6)
            results.append(entry)

        return results
