"""
Simple Hybrid Search: BM25 + Semantic + Union Merge
No extra complexity, just effective retrieval.
"""

import logging
from typing import List, Dict
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class SimpleHybridRetriever:
    """
    Hybrid retriever combining:
    - BM25: Exact keyword matching (sparse)
    - Semantic: Embedding similarity (dense)
    - Merge: Simple union with deduplication
    
    No cross-encoder, no RRF complexity.
    """
    
    def __init__(self, vectorstore, chunks):
        """
        Initialize hybrid retriever
        
        Args:
            vectorstore: Chroma or similar vector store
            chunks: List of document chunks (LangChain Document objects)
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.chunk_texts = [chunk.page_content for chunk in chunks]
        
        # Prepare BM25 index
        tokenized_corpus = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(f"✓ HybridRetriever initialized with {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = 3, use_bm25: bool = True) -> List[Dict]:
        """
        Search using hybrid approach
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_bm25: If True, combine BM25 + semantic. If False, semantic only.
        
        Returns:
            List of documents with metadata
        """
        
        # Semantic search
        semantic_docs = self.vectorstore.similarity_search(query, k=top_k)
        
        if not use_bm25:
            # Only semantic
            return [
                {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'source': 'semantic'
                }
                for doc in semantic_docs
            ]
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top BM25 indices
        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k]
        
        bm25_docs = [
            {
                'text': self.chunks[i].page_content,
                'metadata': self.chunks[i].metadata,
                'source': 'bm25',
                'score': bm25_scores[i]
            }
            for i in bm25_indices
        ]
        
        semantic_dicts = [
            {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'source': 'semantic'
            }
            for doc in semantic_docs
        ]
        
        # Merge: Union with deduplication
        seen = set()
        merged = []
        
        # Add semantic results first (usually better quality)
        for doc in semantic_dicts:
            text_key = doc['text'][:100]  # Use first 100 chars as key
            if text_key not in seen:
                seen.add(text_key)
                merged.append(doc)
        
        # Add BM25 results (fill gaps)
        for doc in bm25_docs:
            text_key = doc['text'][:100]
            if text_key not in seen:
                seen.add(text_key)
                merged.append(doc)
        
        # Return top_k
        result = merged[:top_k]
        
        logger.debug(
            f"Hybrid search: {len(semantic_dicts)} semantic + "
            f"{len(bm25_docs)} bm25 → {len(result)} merged"
        )
        
        return result


if __name__ == "__main__":
    print("✓ SimpleHybridRetriever Ready")
    print("  - BM25 keyword matching")
    print("  - Semantic embedding search")
    print("  - Union merge (no complexity)")
