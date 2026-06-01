import logging
from typing import List, Dict
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class ContextReranker:
    """
    Reranks chunks retrieved from vector store using a CrossEncoder model.
    This acts as a "Context Compression" step to feed only the best chunks to the LLM.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading Reranker: {model_name} on {self.device}...")
        self.model = CrossEncoder(model_name, device=self.device)
        logger.info("✅ Reranker loaded successfully.")

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 2) -> List[Dict]:
        """
        Rerank a list of chunks based on a query.
        
        Args:
            query: User's question
            chunks: List of chunk dictionaries (must have "text" key)
            top_k: Number of top chunks to return
            
        Returns:
            List of the top_k best chunk dictionaries
        """
        if not chunks:
            return []
            
        # Prepare pairs for cross-encoder: [query, passage]
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add scores to chunks and keep track of original object
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
            
        # Sort chunks by score descending
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]
