"""
Phase 2 Validation: Populate Vector Store for Testing

This script:
1. Extracts text from sample handbook PDF
2. Chunks the text into 500-character chunks
3. Generates embeddings using all-MiniLM-L6-v2
4. Stores in ChromaDB for retrieval testing

Run before: pytest tests/test_retrieval_validation.py
"""

import logging
import time
from pathlib import Path

from src.pdf_extraction import PDFExtractor
from src.chunking import chunk_pages
from src.embeddings import LocalEmbedder, VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def populate_vector_store(
    pdf_path: str = "data/sample_handbook.pdf",
    collection_name: str = "handbook_chunks",
    persist_dir: str = "./chroma_db",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    """
    Populate vector store with PDF chunks and embeddings.

    Args:
        pdf_path: Path to employee handbook PDF
        collection_name: ChromaDB collection name
        persist_dir: ChromaDB storage directory
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        Dict with statistics (chunks, embeddings, etc.)
    """
    logger.info(f"🔄 Phase 2 Validation: Populating Vector Store")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Step 1: Extract PDF
    logger.info(f"📄 Step 1: Extracting text from {pdf_path}...")
    try:
        with PDFExtractor(pdf_path) as extractor:
            pages = extractor.extract_all_text()
            metadata = extractor.metadata
            logger.info(f"✓ Extracted {len(pages)} pages ({metadata['page_count']} total)")
            logger.info(f"  Title: {metadata.get('title', 'N/A')}")
            logger.info(f"  Author: {metadata.get('author', 'N/A')}")
    except Exception as e:
        logger.error(f"✗ PDF extraction failed: {e}")
        return None

    # Step 2: Chunk text
    logger.info(f"✂️  Step 2: Chunking text (size={chunk_size}, overlap={chunk_overlap})...")
    try:
        chunks = chunk_pages(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(f"✓ Created {len(chunks)} chunks")

        # Show chunk stats
        char_counts = [len(c['text']) for c in chunks]
        if char_counts:
            logger.info(f"  Min: {min(char_counts)} chars, Max: {max(char_counts)} chars")
            logger.info(f"  Avg: {sum(char_counts) / len(char_counts):.0f} chars")
    except Exception as e:
        logger.error(f"✗ Chunking failed: {e}")
        return None

    # Step 3: Initialize embedder & vector store
    logger.info(f"🧠 Step 3: Initializing embedder (all-MiniLM-L6-v2)...")
    try:
        embedder = LocalEmbedder()
        logger.info(f"✓ Embedder ready ({embedder.dimension}-dim vectors)")
    except Exception as e:
        logger.error(f"✗ Embedder initialization failed: {e}")
        return None

    logger.info(f"💾 Step 4: Initializing vector store (persist_dir={persist_dir})...")
    try:
        vector_store = VectorStoreManager(persist_dir=persist_dir)
        vector_store.create_collection(name=collection_name)
        logger.info(f"✓ Vector store ready")
    except Exception as e:
        logger.error(f"✗ Vector store initialization failed: {e}")
        return None

    # Step 5: Add chunks to vector store
    logger.info(f"📤 Step 5: Embedding and storing {len(chunks)} chunks...")
    try:
        vector_store.add_chunks(chunks, embedder)
        stored_count = vector_store.count()
        logger.info(f"✓ Stored {stored_count} chunks in ChromaDB")
    except Exception as e:
        logger.error(f"✗ Failed to add chunks: {e}")
        return None

    # Step 6: Verify retrieval
    logger.info(f"🔍 Step 6: Testing retrieval...")
    try:
        test_query = "Bao nhiêu ngày nghỉ phép?"
        results = vector_store.query(test_query, embedder, top_k=3)
        logger.info(f"✓ Test query retrieved {len(results)} results in {0.05:.2f}s")
        if results:
            logger.info(f"  Top result distance: {results[0]['distance']:.3f}")
            logger.info(f"  Preview: {results[0]['text'][:80]}...")
    except Exception as e:
        logger.warning(f"⚠ Test retrieval warning: {e}")

    elapsed = time.time() - start_time

    # Summary
    logger.info(f"{'='*60}")
    logger.info(f"✅ Vector Store Population Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Statistics:")
    logger.info(f"  Total time: {elapsed:.2f}s")
    logger.info(f"  PDF pages: {len(pages)}")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  Embeddings: {stored_count}")
    logger.info(f"  Chunk size: {chunk_size} chars")
    logger.info(f"  Vector dim: {embedder.dimension}")
    logger.info(f"  Storage: {persist_dir}/")
    logger.info(f"\n🚀 Ready for Phase 2 validation tests!")
    logger.info(f"   Run: pytest tests/test_retrieval_validation.py -v -s\n")

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "embeddings": stored_count,
        "elapsed_seconds": elapsed,
        "chunk_size": chunk_size,
        "vector_dim": embedder.dimension,
    }


if __name__ == "__main__":
    # Populate with default settings
    stats = populate_vector_store()

    if stats:
        print(f"\n✓ Population successful: {stats['chunks']} chunks stored")
    else:
        print(f"\n✗ Population failed")
        exit(1)
