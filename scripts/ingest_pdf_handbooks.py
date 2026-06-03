"""
Extract text from Vietnamese HR PDFs in data/raw/pdf/ and chunk into data/raw_chunks_viet.jsonl.

Usage:
    python scripts/ingest_pdf_handbooks.py
    python scripts/ingest_pdf_handbooks.py --pdf-dir data/raw/pdf --output data/raw_chunks_viet.jsonl
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.pdf_extraction import PDFExtractor
from src.chunking import chunk_pages

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ingest_pdf(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> list[dict]:
    try:
        with PDFExtractor(str(pdf_path)) as ex:
            pages = ex.extract_all_text()
    except Exception as e:
        print(f"  ERROR extracting {pdf_path.name}: {e}")
        return []

    if not pages:
        print(f"  skip {pdf_path.name}: no text extracted (scanned/image PDF?)")
        return []

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for c in chunks:
        c["source"] = pdf_path.stem
        c["filename"] = pdf_path.name

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="data/raw/pdf", help="Folder with PDF files")
    parser.add_argument("--output", default="data/raw_chunks_viet.jsonl", help="Output JSONL")
    parser.add_argument("--append", action="store_true", help="Append to existing output instead of overwrite")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")

    all_chunks = []
    for pdf in pdfs:
        print(f"  Processing: {pdf.name}")
        chunks = ingest_pdf(pdf, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"    -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    mode = "a" if args.append else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for c in all_chunks:
            json.dump(c, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nTotal: {len(all_chunks)} chunks → {output_path}")
    print(f"Run QA generation:")
    print(f"  python scripts/generate_qa.py \\")
    print(f"    --vllm-url https://api.groq.com/openai \\")
    print(f"    --api-key gsk_YOUR_KEY \\")
    print(f"    --model llama-3.3-70b-versatile \\")
    print(f"    --input {output_path}")


if __name__ == "__main__":
    main()
