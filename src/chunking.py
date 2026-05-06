"""
Text chunking strategies using LangChain's RecursiveCharacterTextSplitter.
Supports multiple chunk sizes for experimentation.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into chunks using recursive splitting.

    Args:
        text: Input text
        chunk_size: Target characters per chunk
        chunk_overlap: Overlap characters between chunks

    Returns:
        List of chunk strings
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split {len(text)} chars into {len(chunks)} chunks (size={chunk_size})")
    return chunks


def chunk_pages(pages_data: List[Dict], chunk_size: int = 600, chunk_overlap: int = 100) -> List[Dict]:
    """
    Chunk each page's text and attach metadata.

    Args:
        pages_data: List from PDFExtractor.extract_all_text()
        chunk_size: Target characters per chunk
        chunk_overlap: Overlap characters between chunks

    Returns:
        List of dicts: {"text": str, "page_num": int, "chunk_index": int, "chunk_size": int}
    """
    all_chunks = []
    for page in pages_data:
        text = page["text"]
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "page_num": page["page_num"],
                "chunk_index": idx,
                "chunk_size": len(chunk),
                "source_file": page.get("source_file", "unknown")
            })
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def experiment_chunk_sizes(pages_data: List[Dict], sizes: List[int] = None) -> Dict:
    """
    Run chunking experiments with different sizes.

    Args:
        pages_data: List from PDFExtractor.extract_all_text()
        sizes: List of chunk sizes to test

    Returns:
        Dict mapping size -> experiment info
    """
    if sizes is None:
        sizes = [300, 600, 900]

    results = {}
    for size in sizes:
        overlap = size // 6  # ~16% overlap
        chunks = chunk_pages(pages_data, chunk_size=size, chunk_overlap=overlap)
        results[size] = {
            "total_chunks": len(chunks),
            "avg_chunk_len": (
                sum(c["chunk_size"] for c in chunks) / len(chunks) if chunks else 0
            ),
            "sample_chunks": chunks[:3]  # first 3 chunks as example
        }
    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.pdf_extraction import PDFExtractor

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_handbook.pdf"

    with PDFExtractor(pdf_path) as extractor:
        pages = extractor.extract_all_text()

        print(f"\nTesting chunk sizes on: {pdf_path}")
        results = experiment_chunk_sizes(pages)
        for size, info in results.items():
            print(f"  Size {size:>4}: {info['total_chunks']} chunks, avg length {info['avg_chunk_len']:.0f} chars")
