# -*- coding: utf-8 -*-
"""
Rebuild the ChromaDB knowledge base with multilingual embeddings.

Steps performed:
  1. Delete and recreate the 'handbook_chunks' collection (clean slate).
  2. Ingest all PDFs from data/raw/ and data/sample_handbook.pdf
     using src/pdf_extraction.py + src/chunking.py.
  3. Ingest all .txt files from data/viet_labor_docs/
     (paragraph-based chunking, ~300 chars per chunk).
  4. Embed everything with the new multilingual model via LocalEmbedder()
     (picks up paraphrase-multilingual-MiniLM-L12-v2 by default).
  5. Print chunk counts per source.

Usage:
    python scripts/reindex_chromadb.py

Requirements:
    - sentence-transformers, chromadb installed
    - data/viet_labor_docs/ populated (run download_viet_labor_docs.py first)
"""

import sys
import logging
from pathlib import Path

# Allow imports from repo root src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import LocalEmbedder, VectorStoreManager
from src.pdf_extraction import PDFExtractor
from src.chunking import chunk_pages

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION_NAME = "handbook_chunks"

PDF_RAW_DIR = REPO_ROOT / "data" / "raw"
SAMPLE_PDF = REPO_ROOT / "data" / "sample_handbook.pdf"
VIET_DOCS_DIR = REPO_ROOT / "data" / "viet_labor_docs"

PDF_CHUNK_SIZE = 600      # characters per chunk for PDF sources
PDF_CHUNK_OVERLAP = 100
VIET_CHUNK_SIZE = 300     # smaller chunks for Vietnamese policy paragraphs
VIET_CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_pdfs() -> list:
    """Return list of PDF paths from data/raw/ and data/sample_handbook.pdf."""
    pdf_paths = []

    if PDF_RAW_DIR.is_dir():
        pdfs = list(PDF_RAW_DIR.rglob("*.pdf"))
        pdf_paths.extend(pdfs)
        logger.info(f"Found {len(pdfs)} PDF(s) in {PDF_RAW_DIR}")
    else:
        logger.warning(f"data/raw/ directory not found: {PDF_RAW_DIR}")

    if SAMPLE_PDF.exists():
        pdf_paths.append(SAMPLE_PDF)
        logger.info(f"Found sample PDF: {SAMPLE_PDF.name}")
    else:
        logger.warning(f"Sample PDF not found: {SAMPLE_PDF}")

    return pdf_paths


def ingest_pdfs(pdf_paths: list) -> list:
    """Extract text from PDFs and return chunked dicts with source metadata."""
    all_chunks = []

    for pdf_path in pdf_paths:
        logger.info(f"  Processing PDF: {pdf_path.name} ...")
        try:
            with PDFExtractor(str(pdf_path)) as extractor:
                pages = extractor.extract_all_text()
                # Attach source filename to each page before chunking
                for page in pages:
                    page["source_file"] = pdf_path.name

            chunks = chunk_pages(
                pages,
                chunk_size=PDF_CHUNK_SIZE,
                chunk_overlap=PDF_CHUNK_OVERLAP,
            )
            logger.info(f"    {len(chunks)} chunks from {pdf_path.name}")
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"    Failed to process {pdf_path.name}: {e}")

    return all_chunks


def chunk_txt_by_paragraph(text: str, max_chars: int = 300, overlap: int = 50) -> list:
    """
    Split Vietnamese text into chunks by paragraph boundaries.

    Paragraphs are separated by blank lines or newlines. Paragraphs longer
    than max_chars are split further at sentence boundaries ('.' / '!' / '?').
    """
    # Split on double-newlines first (clean paragraph break)
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in raw_paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # Split long paragraphs by sentence-ending punctuation
            import re
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 <= max_chars:
                    current = (current + " " + sentence).strip() if current else sentence
                else:
                    if current:
                        chunks.append(current)
                    # Start new chunk with overlap from end of previous
                    if overlap > 0 and current:
                        overlap_text = current[-overlap:].strip()
                        current = (overlap_text + " " + sentence).strip()
                    else:
                        current = sentence
            if current:
                chunks.append(current)

    return [c for c in chunks if len(c) > 20]  # drop very short fragments


def ingest_viet_docs(viet_dir: Path) -> list:
    """Read .txt files from data/viet_labor_docs/ and chunk by paragraph."""
    all_chunks = []

    if not viet_dir.is_dir():
        logger.warning(f"Vietnamese docs directory not found: {viet_dir}")
        logger.warning("Run scripts/download_viet_labor_docs.py first.")
        return all_chunks

    txt_files = sorted(viet_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(f"No .txt files found in {viet_dir}")
        return all_chunks

    logger.info(f"Found {len(txt_files)} Vietnamese .txt file(s) in {viet_dir}")

    for txt_path in txt_files:
        try:
            text = txt_path.read_text(encoding="utf-8")
            paragraphs = chunk_txt_by_paragraph(
                text, max_chars=VIET_CHUNK_SIZE, overlap=VIET_CHUNK_OVERLAP
            )
            for idx, para in enumerate(paragraphs):
                all_chunks.append({
                    "text": para,
                    "page_num": 0,            # .txt files have no page number
                    "chunk_index": idx,
                    "chunk_size": len(para),
                    "source_file": txt_path.name,
                })
            logger.info(f"  {txt_path.name}: {len(paragraphs)} chunks")
        except Exception as e:
            logger.error(f"  Failed to read {txt_path.name}: {e}")

    return all_chunks


def reset_collection(manager: VectorStoreManager, collection_name: str) -> None:
    """Delete and recreate the ChromaDB collection for a clean rebuild."""
    try:
        manager.client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: '{collection_name}'")
    except Exception:
        logger.info(f"Collection '{collection_name}' did not exist — creating fresh.")
    manager.create_collection(collection_name)


def add_chunks_to_db(
    manager: VectorStoreManager,
    embedder: LocalEmbedder,
    chunks: list,
    source_label: str,
    id_offset: int = 0,
) -> int:
    """
    Embed and store a list of chunks, returning the number stored.

    Uses a unique id_offset to avoid ID collisions when adding multiple
    source batches to the same collection.
    """
    if not chunks:
        logger.warning(f"  No chunks to add for source: {source_label}")
        return 0

    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "page_num": c["page_num"],
            "chunk_index": c["chunk_index"],
            "source": c.get("source_file", source_label),
            "source_type": source_label,
        }
        for c in chunks
    ]
    ids = [f"chunk_{id_offset + i}" for i in range(len(chunks))]

    logger.info(f"  Embedding {len(chunks)} chunks from '{source_label}' ...")
    embeddings = embedder.embed(texts, show_progress=True)

    manager.collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )
    logger.info(f"  Stored {len(chunks)} chunks ({source_label}).")
    return len(chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("ChromaDB re-index (multilingual embeddings)")
    logger.info("=" * 60)

    # 1. Initialise embedder and vector store
    logger.info("\n[1/5] Loading multilingual embedder ...")
    embedder = LocalEmbedder()  # picks up paraphrase-multilingual-MiniLM-L12-v2

    logger.info("\n[2/5] Connecting to ChromaDB ...")
    manager = VectorStoreManager(persist_dir=str(CHROMA_DIR))

    # 2. Clear existing collection
    logger.info(f"\n[3/5] Resetting collection '{COLLECTION_NAME}' ...")
    reset_collection(manager, COLLECTION_NAME)

    # 3. Ingest PDFs
    logger.info("\n[4/5] Ingesting PDF sources ...")
    pdf_paths = collect_pdfs()
    pdf_chunks = ingest_pdfs(pdf_paths)
    id_offset = 0

    pdf_stored = add_chunks_to_db(
        manager, embedder, pdf_chunks, "pdf_handbooks", id_offset=id_offset
    )
    id_offset += pdf_stored

    # 4. Ingest Vietnamese .txt documents
    logger.info("\n[5/5] Ingesting Vietnamese labor docs ...")
    viet_chunks = ingest_viet_docs(VIET_DOCS_DIR)
    viet_stored = add_chunks_to_db(
        manager, embedder, viet_chunks, "viet_labor_docs", id_offset=id_offset
    )
    id_offset += viet_stored

    # 5. Summary
    total = manager.count()
    logger.info("\n" + "=" * 60)
    logger.info("Re-index complete. Chunk counts per source:")
    logger.info(f"  PDF handbooks      : {pdf_stored:>5} chunks")
    logger.info(f"  Vietnamese docs    : {viet_stored:>5} chunks")
    logger.info(f"  Total in ChromaDB  : {total:>5} chunks")
    logger.info("=" * 60)

    if total == 0:
        logger.warning("WARNING: Zero chunks stored. Check that input files exist.")
    elif total < 100:
        logger.warning(
            f"WARNING: Only {total} chunks stored — fewer than expected. "
            "Run download_viet_labor_docs.py and re-check data/raw/."
        )
    else:
        logger.info("Knowledge base ready for multilingual Vietnamese RAG queries.")


if __name__ == "__main__":
    main()
