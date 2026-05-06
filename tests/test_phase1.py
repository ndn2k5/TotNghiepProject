# test_phase1.py
from src.pdf_extraction import PDFExtractor

with PDFExtractor("data/sample_handbook.pdf") as e:
    pages = e.extract_all_text()
    print(f"Pages: {len(pages)}")