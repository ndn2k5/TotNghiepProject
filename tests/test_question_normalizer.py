"""
Tests for question_normalizer.py

Coverage:
  - Heuristic normalization (diacritics, whitespace, abbreviations)
  - HR domain keyword detection
  - Question validation
  - Query variant generation (heuristic)
"""

import pytest
from src.question_normalizer import QuestionNormalizer, normalize_question, HR_KEYWORDS


class TestHeuristicNormalization:
    """Test heuristic (rule-based) Vietnamese normalization."""

    def test_lowercase_conversion(self):
        """Test that uppercase is converted to lowercase."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalizer.normalize("NGHỈ PHÉP CHO CÔNG NHÂN")
        assert result == result.lower()

    def test_whitespace_normalization(self):
        """Test that multiple spaces are collapsed."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalizer.normalize("Câu   hỏi   về    lương")
        assert "  " not in result  # No double spaces
        assert result == "câu hỏi về lương"

    def test_leading_trailing_whitespace(self):
        """Test that leading/trailing spaces are removed."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalizer.normalize("  Hợp đồng lao động  ")
        assert result == result.strip()
        assert result == "hợp đồng lao động"

    def test_diacritic_normalization(self):
        """Test Vietnamese diacritic normalization."""
        normalizer = QuestionNormalizer(use_llm=False)
        # These should be normalized consistently
        result = normalizer.normalize("Tôi muốn hỏi về lương thưởng")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_input(self):
        """Test handling of empty input."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalizer.normalize("")
        assert result == ""

    def test_none_input(self):
        """Test handling of None input."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalizer.normalize(None)
        assert result == ""

    def test_abbreviation_expansion(self):
        """Test common Vietnamese HR abbreviations are expanded."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalizer.normalize("lv được tăng ca không?")
        # 'lv' should be expanded to 'lao động' or similar
        assert "lao" in result  # Partial match for expansion


class TestKeywordExtraction:
    """Test HR keyword extraction."""

    def test_vacation_keywords(self):
        """Test detection of vacation-related keywords."""
        normalizer = QuestionNormalizer(use_llm=False)
        keywords = normalizer.extract_keywords("Tôi muốn hỏi về nghỉ phép")
        assert "vacation" in keywords
        assert "nghỉ phép" in keywords.get("vacation", [])

    def test_sick_leave_keywords(self):
        """Test detection of sick leave keywords."""
        normalizer = QuestionNormalizer(use_llm=False)
        keywords = normalizer.extract_keywords("Khi nào được nghỉ ốm?")
        assert "sick_leave" in keywords

    def test_overtime_keywords(self):
        """Test detection of overtime keywords."""
        normalizer = QuestionNormalizer(use_llm=False)
        keywords = normalizer.extract_keywords("Làm thêm giờ có được hưởng trợ cấp không?")
        assert "overtime" in keywords

    def test_multiple_keywords(self):
        """Test detection of multiple keyword categories."""
        normalizer = QuestionNormalizer(use_llm=False)
        keywords = normalizer.extract_keywords("Hợp đồng lao động và lương thế nào?")
        assert len(keywords) >= 1

    def test_no_keywords(self):
        """Test question with no HR keywords."""
        normalizer = QuestionNormalizer(use_llm=False)
        keywords = normalizer.extract_keywords("Thời tiết hôm nay như thế nào?")
        assert "general_hr" not in keywords  # Not an HR question


class TestHRDomainCheck:
    """Test HR domain validation."""

    def test_valid_hr_question(self):
        """Test that clear HR questions are recognized."""
        normalizer = QuestionNormalizer(use_llm=False)
        assert normalizer.is_hr_question("Tôi muốn hỏi về nghỉ phép")
        assert normalizer.is_hr_question("Lương tháng này bao nhiêu?")
        assert normalizer.is_hr_question("Hợp đồng hết hạn lúc nào?")

    def test_invalid_hr_question(self):
        """Test that non-HR questions are rejected."""
        normalizer = QuestionNormalizer(use_llm=False)
        assert not normalizer.is_hr_question("Trời mưa hôm nay")
        assert not normalizer.is_hr_question("Thời tiết bên ngoài")
        assert not normalizer.is_hr_question("Hôm nay là thứ mấy?")


class TestQueryVariantGeneration:
    """Test query variant generation (heuristic only)."""

    def test_variant_generation_returns_list(self):
        """Test that variant generation returns a list."""
        normalizer = QuestionNormalizer(use_llm=False)
        variants = normalizer.generate_query_variants("Hỏi về lương", num_variants=0)
        assert isinstance(variants, list)
        assert len(variants) >= 1  # At least original

    def test_variant_includes_original(self):
        """Test that original question is always included."""
        normalizer = QuestionNormalizer(use_llm=False)
        original = "Tôi muốn biết về phép năm"
        variants = normalizer.generate_query_variants(original, num_variants=2)
        assert original in variants

    def test_variant_count(self):
        """Test that variant count respects the limit."""
        normalizer = QuestionNormalizer(use_llm=False)
        variants = normalizer.generate_query_variants("Câu hỏi test", num_variants=2)
        # Should have original + up to 2 variants
        assert len(variants) <= 3


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_normalize_question_without_normalizer(self):
        """Test normalize_question creates default normalizer."""
        result = normalize_question("Hỏi   về   lương")
        assert isinstance(result, str)
        assert "  " not in result

    def test_normalize_question_with_normalizer(self):
        """Test normalize_question with provided normalizer."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalize_question("  Tôi muốn hỏi  ", normalizer)
        assert result == result.strip()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_question(self):
        """Test handling of very long input."""
        normalizer = QuestionNormalizer(use_llm=False)
        long_question = "Hỏi " * 500  # Very long
        result = normalizer.normalize(long_question)
        assert isinstance(result, str)

    def test_special_characters(self):
        """Test handling of special characters."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalizer.normalize("Câu hỏi: @#$% về lương?")
        assert isinstance(result, str)

    def test_mixed_vietnamese_english(self):
        """Test handling of mixed Vietnamese/English."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalizer.normalize("Salary và lương bằng nhau?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_numeric_input(self):
        """Test handling of numeric content."""
        normalizer = QuestionNormalizer(use_llm=False)
        result = normalizer.normalize("Lương 12345 đó phải không?")
        assert "12345" in result


class TestNormalizerInitialization:
    """Test QuestionNormalizer initialization."""

    def test_init_without_model(self):
        """Test initialization without LLM model."""
        normalizer = QuestionNormalizer(use_llm=False)
        assert normalizer.use_llm is False
        assert normalizer.llm is None

    def test_init_with_nonexistent_model(self):
        """Test initialization with non-existent model path."""
        normalizer = QuestionNormalizer(
            model_path="/nonexistent/path/model.gguf",
            use_llm=True
        )
        # Should fall back to heuristic mode
        assert normalizer.use_llm is False or normalizer.llm is None


class TestKeywordDictionary:
    """Test HR_KEYWORDS dictionary."""

    def test_keywords_exist(self):
        """Test that HR_KEYWORDS is populated."""
        assert len(HR_KEYWORDS) > 0

    def test_keywords_structure(self):
        """Test that HR_KEYWORDS has expected structure."""
        for category, keywords in HR_KEYWORDS.items():
            assert isinstance(category, str)
            assert isinstance(keywords, list)
            assert all(isinstance(kw, str) for kw in keywords)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
