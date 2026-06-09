# -*- coding: utf-8 -*-
"""
Question Normalizer: Process Vietnamese HR questions using local Qwen model.

Responsibilities:
  1. Normalize Vietnamese text (handle diacritics, colloquial phrasings)
  2. Extract keywords for semantic search
  3. Validate question relevance to HR domain
  4. Optional: Generate query variants for better retrieval

All processing is local — no external API calls.
"""

import logging
import re
import unicodedata
from typing import List, Dict, Optional
from pathlib import Path

from src.gguf_models import LocalGGUFModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vietnamese HR domain keywords
HR_KEYWORDS = {
    "vacation": ["nghỉ phép", "kỳ nghỉ", "ngày nghỉ", "phép"],
    "sick_leave": ["ngày ốm", "nghỉ ốm", "chứng chỉ y tế", "sick"],
    "overtime": ["làm thêm giờ", "tăng ca", "overtime", "phụ cấp"],
    "salary": ["lương", "trả lương", "bảng lương", "salary"],
    "benefits": ["bảo hiểm", "phúc lợi", "trợ cấp", "benefit"],
    "contract": ["hợp đồng", "ký hợp đồng", "điều khoản", "contract"],
    "attendance": ["chấm công", "điểm danh", "vắng mặt", "attendance"],
    "discipline": ["kỷ luật", "vi phạm", "hình phạt", "discipline"],
    "promotion": ["thăng chức", "nâng bậc", "promotion"],
    "resignation": ["từ chức", "thôi việc", "resignation"],
}

HR_DOMAIN_REGEX = r"(tuyển|nhân sự|lương|phép|ốm|tăng ca|bảo hiểm|hợp đồng|kỷ luật|thăng chức)"


class QuestionNormalizer:
    """Normalizes Vietnamese HR questions and extracts search keywords."""

    def __init__(self, model_path: Optional[str] = None, use_llm: bool = True):
        """
        Initialize the question normalizer.

        Args:
            model_path: Path to Qwen GGUF model. If None, use heuristic normalization.
            use_llm: If True and model_path provided, use LLM for normalization.
                     Otherwise use heuristic rules.
        """
        self.use_llm = use_llm and model_path is not None
        self.llm = None

        if self.use_llm:
            try:
                logger.info("Loading Qwen model for question normalization...")
                self.llm = LocalGGUFModel(model_path, n_ctx=512, verbose=False)
                logger.info("✓ Qwen model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load Qwen model: {e}. Falling back to heuristics.")
                self.use_llm = False

    def normalize(self, question: str) -> str:
        """
        Normalize a Vietnamese question for better retrieval.

        Args:
            question: Raw Vietnamese question

        Returns:
            Normalized question string
        """
        if not question or not isinstance(question, str):
            return ""

        # Step 1: Heuristic normalization (always applied)
        normalized = self._heuristic_normalize(question)

        # Step 2: LLM-based normalization (if available)
        if self.use_llm and self.llm:
            try:
                normalized = self._llm_normalize(normalized)
            except Exception as e:
                logger.debug(f"LLM normalization failed: {e}. Using heuristic result.")

        return normalized.strip()

    def _heuristic_normalize(self, question: str) -> str:
        """
        Apply rule-based Vietnamese normalization.

        Handles:
          - Diacritic consistency
          - Whitespace normalization
          - Common abbreviations
          - Case normalization
        """
        # Convert to NFKC form (normalize Vietnamese diacritics)
        text = unicodedata.normalize("NFKC", question)

        # Lowercase (Vietnamese text is typically lowercase)
        text = text.lower()

        # Multiple spaces → single space
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Common abbreviations (Vietnamese HR context)
        abbreviations = {
            r"\bkh\b": "không",
            r"\bđc\b": "được",
            r"\bcó\b": "có",
            r"\blv\b": "lao động",
            r"\bns\b": "nhân sự",
            r"\bpc\b": "phụ cấp",
        }
        for abbr, expansion in abbreviations.items():
            text = re.sub(abbr, expansion, text)

        return text

    def _llm_normalize(self, question: str) -> str:
        """
        Use Qwen model for LLM-based normalization.

        Prompt Qwen to:
          - Clarify colloquial phrasings
          - Extract the core query intent
          - Expand abbreviations
        """
        if not self.llm:
            return question

        prompt = f"""Hãy chuẩn hóa câu hỏi tiếng Việt sau để dễ tìm kiếm trong tài liệu HR. 
Giữ nguyên ý nghĩa nhưng làm cho rõ ràng hơn. Chỉ trả lời bằng câu hỏi đã chuẩn hóa.

Câu hỏi gốc: {question}

Câu hỏi đã chuẩn hóa:"""

        try:
            output = self.llm.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3,
            )
            # Extract the normalized question from LLM output
            normalized = output.strip()
            if "\n" in normalized:
                normalized = normalized.split("\n")[0]
            return normalized[:200]  # Safety cap
        except Exception as e:
            logger.debug(f"LLM generation failed: {e}")
            return question

    def extract_keywords(self, question: str) -> Dict[str, List[str]]:
        """
        Extract HR-relevant keywords from a question.

        Args:
            question: A Vietnamese question

        Returns:
            Dict with category → keyword list
        """
        question_lower = question.lower()
        extracted = {}

        # Match HR domain keywords
        for category, keywords in HR_KEYWORDS.items():
            found = [kw for kw in keywords if kw in question_lower]
            if found:
                extracted[category] = found

        # If no explicit keywords match, check domain regex
        if not extracted:
            if re.search(HR_DOMAIN_REGEX, question_lower):
                extracted["general_hr"] = ["hr_related"]

        return extracted

    def is_hr_question(self, question: str) -> bool:
        """
        Heuristic: Check if question is HR-related.

        Returns True if question contains HR keywords or matches domain regex.
        """
        question_lower = question.lower()

        # Check against HR keywords
        for keywords in HR_KEYWORDS.values():
            if any(kw in question_lower for kw in keywords):
                return True

        # Check domain regex
        if re.search(HR_DOMAIN_REGEX, question_lower):
            return True

        return False

    def generate_query_variants(
        self, question: str, num_variants: int = 2
    ) -> List[str]:
        """
        Generate alternative phrasings of a question for better retrieval.

        Args:
            question: Original question
            num_variants: Number of variants to generate

        Returns:
            List of question variants (including original)
        """
        variants = [question]  # Include original

        if not self.llm or num_variants < 1:
            return variants

        prompt = f"""Hãy tạo {num_variants} cách khác để hỏi câu sau, vẫn giữ nguyên ý nghĩa:
Câu gốc: {question}

Liệt kê các cách hỏi khác (mỗi dòng một câu):"""

        try:
            output = self.llm.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.5,
            )
            lines = output.strip().split("\n")
            for line in lines[:num_variants]:
                clean = line.strip().lstrip("0123456789.-) ").strip()
                if clean and clean.lower() != question.lower():
                    variants.append(clean)
        except Exception as e:
            logger.debug(f"Query variant generation failed: {e}")

        return variants[:num_variants + 1]  # Original + variants


def normalize_question(question: str, normalizer: Optional[QuestionNormalizer] = None) -> str:
    """
    Helper function: Normalize a single question.

    Args:
        question: Vietnamese question
        normalizer: QuestionNormalizer instance (creates default if None)

    Returns:
        Normalized question
    """
    if normalizer is None:
        normalizer = QuestionNormalizer(use_llm=False)  # Heuristic only
    return normalizer.normalize(question)


if __name__ == "__main__":
    # Smoke test without LLM
    normalizer = QuestionNormalizer(use_llm=False)

    test_questions = [
        "Tôi muốn hỏi về nghỉ phép công nhân?",
        "Làm thêm giờ được trả lương gấp mấy lần?",
        "Hợp đồng lao động có thời hạn không?",
    ]

    for q in test_questions:
        normalized = normalizer.normalize(q)
        keywords = normalizer.extract_keywords(q)
        is_hr = normalizer.is_hr_question(q)
        print(f"Original: {q}")
        print(f"Normalized: {normalized}")
        print(f"Keywords: {keywords}")
        print(f"Is HR Question: {is_hr}")
        print()

    print("✓ QuestionNormalizer smoke test passed (heuristic mode).")
