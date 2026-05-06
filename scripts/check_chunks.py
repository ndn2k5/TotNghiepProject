"""Quick script to display chunk size experiment results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_extraction import PDFExtractor
from src.chunking import experiment_chunk_sizes

pdf_path = "data/sample_handbook.pdf"

with PDFExtractor(pdf_path) as e:
    pages = e.extract_all_text()
    results = experiment_chunk_sizes(pages)

print()
print("=== Chunk Size Experiment Results ===")
for size, info in results.items():
    avg = info["avg_chunk_len"]
    total = info["total_chunks"]
    print(f"  {size:>4} chars -> {total} chunks, avg {avg:.0f} chars/chunk")
print()
print("Best for RAG: 300-600 chars (balance between context and precision)")
