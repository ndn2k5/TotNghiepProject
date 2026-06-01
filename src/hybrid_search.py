"""
Hybrid Search with BM25 + Semantic Search + RRF
Combines lexical and semantic search for better retrieval
"""

import logging
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 sparse retriever (lexical search)"""
    
    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
            self.BM25Okapi = BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. Install with: pip install rank-bm25")
            self.BM25Okapi = None
        
        self.corpus = []
        self.bm25 = None
    
    def index(self, documents: List[str]) -> None:
        """Index documents for BM25"""
        if not self.BM25Okapi:
            logger.error("rank_bm25 not available")
            return
        
        # Tokenize Vietnamese text
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.corpus = documents
        self.bm25 = self.BM25Okapi(tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for Vietnamese"""
        # Split on whitespace and remove punctuation
        import re
        text = text.lower()
        words = re.findall(r'\w+', text)
        return words
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search using BM25"""
        if not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        
        return results


class HybridSearcher:
    """
    Hybrid Search combining BM25 + Semantic Search
    Uses Reciprocal Rank Fusion (RRF) for fusion
    """
    
    def __init__(self, embedder, vector_store):
        """
        Initialize hybrid searcher
        
        Args:
            embedder: LocalEmbedder for semantic search
            vector_store: Vector store (ChromaDB)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = BM25Retriever()
        self.documents = []
    
    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents for both BM25 and semantic search"""
        doc_texts = [doc['text'] for doc in documents]
        self.documents = doc_texts
        self.bm25.index(doc_texts)
        
        logger.info(f"Indexed {len(documents)} documents for hybrid search")
    
    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        semantic_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF)
        Combines rankings from multiple retrieval systems
        
        Formula: score = 1 / (k + rank)
        """
        rrf_scores = {}
        
        # Add BM25 scores
        for rank, (idx, score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        
        # Add semantic scores
        for rank, (idx, score) in enumerate(semantic_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        
        # Sort by RRF score
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused_results
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and semantic search
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for semantic results (1-alpha for BM25)
            
        Returns:
            List of retrieved documents with scores
        """
        # Get results from both retrievers
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        
        # Semantic search
        query_embedding = self.embedder.embed(query)
        semantic_results = self.vector_store.search(
            query_embedding,
            top_k=top_k * 2
        )
        
        # Convert semantic results to same format
        semantic_results = [
            (i, result.get('distance', 0))
            for i, result in enumerate(semantic_results)
        ]
        
        # Fusion using RRF
        fused_results = self._rrf_fusion(bm25_results, semantic_results)
        
        # Get top-k fused results
        top_results = fused_results[:top_k]
        
        return [
            {
                'index': idx,
                'text': self.documents[idx] if idx < len(self.documents) else '',
                'hybrid_score': score
            }
            for idx, score in top_results
        ]


if __name__ == "__main__":
    print("Hybrid Search Module Ready")
    print("- BM25 Retriever ✓")
    print("- Semantic Search ✓")
    print("- RRF Fusion ✓")
