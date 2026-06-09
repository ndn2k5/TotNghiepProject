# -*- coding: utf-8 -*-
"""
PDF Extraction Module using PyMuPDF (fitz)
Extracts text, metadata, and page numbers from employee handbook PDFs.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text and metadata from PDF files."""
    
    def __init__(self, file_path: str):
        """
        Initialize PDF extractor.
        
        Args:
            file_path: Path to PDF file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        self.doc = fitz.open(self.file_path)
        self.metadata = self._extract_metadata()
        
    def _extract_metadata(self) -> Dict:
        """Extract PDF metadata."""
        meta = self.doc.metadata
        return {
            "title": meta.get("title", self.file_path.stem),
            "author": meta.get("author", "Unknown"),
            "subject": meta.get("subject", ""),
            "keywords": meta.get("keywords", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "page_count": len(self.doc),
            "file_name": self.file_path.name
        }
    
    def extract_page_text(self, page_num: int) -> Optional[str]:
        """
        Extract text from a specific page (0-indexed).
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            Text content or None if page invalid
        """
        if page_num < 0 or page_num >= len(self.doc):
            logger.warning(f"Page {page_num} out of range (0-{len(self.doc)-1})")
            return None
        
        page = self.doc[page_num]
        text = page.get_text("text")
        return text.strip()
    
    def extract_all_text(self) -> List[Dict]:
        """
        Extract text from all pages with metadata.
        
        Returns:
            List of dicts: {"page_num": int, "text": str, "char_count": int}
        """
        pages_data = []
        for i in range(len(self.doc)):
            text = self.extract_page_text(i)
            if text:
                pages_data.append({
                    "page_num": i + 1,  # 1-indexed for users
                    "text": text,
                    "char_count": len(text)
                })
            else:
                logger.warning(f"Page {i+1} has no extractable text")
        
        logger.info(f"Extracted {len(pages_data)} pages with text content")
        return pages_data
    
    def extract_sections(self, heading_pattern: str = r'^(Chapter|Section|Article|Phần|Chương|Điều)\s+\d+') -> List[Dict]:
        """
        Attempt to split document into sections based on heading patterns.
        
        Args:
            heading_pattern: Regex pattern for detecting section headings
            
        Returns:
            List of sections: {"heading": str, "content": str, "start_page": int}
        """
        sections = []
        current_heading = "Introduction"
        current_content = []
        current_start_page = 1
        
        for page_data in self.extract_all_text():
            lines = page_data["text"].split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line matches heading pattern
                if re.match(heading_pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_content:
                        sections.append({
                            "heading": current_heading,
                            "content": " ".join(current_content),
                            "start_page": current_start_page,
                            "end_page": page_data["page_num"]
                        })
                    # Start new section
                    current_heading = line
                    current_content = []
                    current_start_page = page_data["page_num"]
                else:
                    current_content.append(line)
        
        # Add last section
        if current_content:
            sections.append({
                "heading": current_heading,
                "content": " ".join(current_content),
                "start_page": current_start_page,
                "end_page": len(self.doc)
            })
        
        logger.info(f"Extracted {len(sections)} sections")
        return sections
    
    def close(self):
        """Close the PDF document."""
        self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def quick_extract(file_path: str) -> str:
    """
    Quick function to extract all text from a PDF.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Full text concatenation
    """
    with PDFExtractor(file_path) as extractor:
        pages = extractor.extract_all_text()
        full_text = "\n\n".join([p["text"] for p in pages])
    return full_text


if __name__ == "__main__":
    # Quick test if script run directly
    import sys
    if len(sys.argv) > 1:
        text = quick_extract(sys.argv[1])
        print(f"Extracted {len(text)} characters")
        print("\nFirst 500 chars:\n", text[:500])
    else:
        print("Usage: python pdf_extraction.py <path_to_pdf>")
