"""
Local embedding using sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2).
Multilingual model supporting 50+ languages including Vietnamese — no API calls,
runs entirely on CPU or GPU with CUDA support.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Optional
import logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Wraps sentence-transformers for local multilingual text embedding (no API).

    Default model: paraphrase-multilingual-MiniLM-L12-v2
    - 50+ language support including Vietnamese
    - 384-dimensional output vectors
    - Fast CPU inference, GPU-accelerated when available
    """

    MODEL_DIMENSION = 384  # paraphrase-multilingual-MiniLM-L12-v2 output dimension

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: Optional[str] = None):
        """
        Load the embedding model (cached after first download).

        Args:
            model_name: HuggingFace model name. Default is paraphrase-multilingual-MiniLM-L12-v2
                        for cross-lingual Vietnamese retrieval.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        logger.info(f"Loading embedding model: {model_name} on {self.device.upper()} ...")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.MODEL_DIMENSION
        
        if self.device == "cuda":
            logger.info("✅ Model loaded successfully on GPU (CUDA)")
        else:
            logger.info("Model loaded successfully on CPU")

    def embed(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed
            show_progress: Show tqdm progress bar (useful for large batches)

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        vectors = self.model.encode(texts, show_progress_bar=show_progress)
        return vectors.tolist()


class VectorStoreManager:
    """Manages a ChromaDB persistent collection for semantic search."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Initialize ChromaDB client with a local persistent directory.

        Args:
            persist_dir: Directory where ChromaDB stores its data
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = None

    def create_collection(self, name: str = "handbook_chunks"):
        """Get or create a named collection."""
        self.collection = self.client.get_or_create_collection(name=name)
        logger.info(f"Collection '{name}' ready — {self.collection.count()} documents stored.")
        return self.collection

    def add_chunks(self, chunks: List[Dict], embedder: LocalEmbedder):
        """
        Embed and store text chunks in ChromaDB.

        Args:
            chunks: Output of chunking.chunk_pages()
            embedder: A LocalEmbedder instance
        """
        if not self.collection:
            self.create_collection()

        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts, show_progress=True)

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "page_num": c["page_num"],
                "chunk_index": c["chunk_index"],
                "source": c.get("source_file", "handbook")
            }
            for c in chunks
        ]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(chunks)} chunks to vector DB.")

    def query(self, query_text: str, embedder: LocalEmbedder, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most semantically similar chunks.

        Args:
            query_text: The user's question
            embedder: A LocalEmbedder instance
            top_k: Number of results to return

        Returns:
            List of dicts with text, metadata, and distance score
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection() first.")

        query_vector = embedder.embed([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        return formatted

    def count(self) -> int:
        """Return number of documents in the collection."""
        if not self.collection:
            return 0
        return self.collection.count()


if __name__ == "__main__":
    # Quick smoke test — no API required
    embedder = LocalEmbedder()
    test_texts = ["This is a test.", "Employee vacation policy."]
    embeddings = embedder.embed(test_texts)
    print(f"Embeddings: {len(embeddings)} vectors × {len(embeddings[0])} dims")
    print("✓ LocalEmbedder working correctly.")
