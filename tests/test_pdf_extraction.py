"""Unit tests for PDF extraction module."""

import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_extraction import PDFExtractor, quick_extract

# Path to sample PDF
SAMPLE_PDF = Path(__file__).parent.parent / "data" / "sample_handbook.pdf"


class TestPDFExtractor:
    """Test PDFExtractor class."""
    
    def test_pdf_extractor_init(self):
        """Test PDFExtractor initialization."""
        extractor = PDFExtractor(str(SAMPLE_PDF))
        assert extractor.doc is not None
        assert extractor.metadata["page_count"] == 2
        assert "sample_handbook.pdf" in extractor.metadata["file_name"]
        extractor.close()
    
    def test_extract_page_text(self):
        """Test extracting text from a single page."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            page1_text = extractor.extract_page_text(0)
            assert page1_text is not None
            assert "Vacation Policy" in page1_text or "vacation" in page1_text.lower()
            
            page2_text = extractor.extract_page_text(1)
            assert page2_text is not None
            assert "Sick Leave" in page2_text or "sick" in page2_text.lower()
    
    def test_extract_page_out_of_range(self):
        """Test extracting invalid page returns None."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            page_invalid = extractor.extract_page_text(999)
            assert page_invalid is None
    
    def test_extract_all_text(self):
        """Test extracting all pages."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
            assert len(pages) == 2
            assert pages[0]["page_num"] == 1
            assert pages[1]["page_num"] == 2
            assert pages[0]["char_count"] > 0
            assert pages[1]["char_count"] > 0
    
    def test_quick_extract(self):
        """Test quick extraction helper."""
        text = quick_extract(str(SAMPLE_PDF))
        assert len(text) > 0
        assert "Vacation" in text or "vacation" in text.lower()
        assert "Sick" in text or "sick" in text.lower()
    
    def test_extract_sections(self):
        """Test section extraction."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            sections = extractor.extract_sections(r'^(Chapter|Chương|Phần)\s+\d+')
            # Should extract at least some sections
            assert len(sections) >= 1
    
    def test_metadata_extraction(self):
        """Test metadata extraction."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            meta = extractor.metadata
            assert meta["page_count"] == 2
            assert "file_name" in meta
            assert "title" in meta
            assert "author" in meta


class TestQuickExtract:
    """Test quick_extract function."""
    
    def test_quick_extract_returns_string(self):
        """Test quick_extract returns a string."""
        result = quick_extract(str(SAMPLE_PDF))
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_quick_extract_file_not_found(self):
        """Test quick_extract raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            quick_extract("/nonexistent/file.pdf")


class TestPDFExtractorContextManager:
    """Test PDFExtractor context manager."""
    
    def test_context_manager(self):
        """Test PDFExtractor works as context manager."""
        with PDFExtractor(str(SAMPLE_PDF)) as extractor:
            pages = extractor.extract_all_text()
            assert len(pages) > 0
        # After exiting context, doc should be closed
        assert extractor.doc.is_closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
