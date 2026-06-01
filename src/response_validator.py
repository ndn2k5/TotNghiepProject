"""
Response Quality Validator & Improver
Ensures responses are accurate, complete, and grounded in source documents
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Response quality assessment"""
    completeness: float  # Does it answer the question fully?
    grounding: float     # Is it based on provided context?
    clarity: float       # Is it clear and well-formatted?
    length: float        # Is it appropriately sized?
    language: float      # Is it proper Vietnamese?
    overall: float       # Weighted average
    issues: List[str]    # List of detected issues
    suggestions: List[str]  # How to improve


class ResponseValidator:
    """Validates response quality against criteria"""
    
    def __init__(self):
        self.min_acceptable_score = 0.6
    
    def check_completeness(self, question: str, response: str) -> Tuple[float, List[str]]:
        """
        Check if response fully answers the question.
        Returns score 0-1 and list of issues.
        """
        issues = []
        score = 1.0
        
        question_lower = question.lower()
        response_lower = response.lower()
        
        # Check for non-answer patterns
        non_answers = [
            "không tìm thấy",
            "không có thông tin",
            "không rõ",
            "không biết",
            "không được đề cập",
        ]
        
        has_non_answer = any(pattern in response_lower for pattern in non_answers)
        if has_non_answer and len(response) < 100:
            score -= 0.3
            issues.append("Response appears to be 'not found' - should search better")
        
        # Check for question-specific words
        if len(response) > 50:  # Only check for substantial responses
            # Extract key question terms
            question_terms = re.findall(r'\b\w{4,}\b', question_lower)
            if question_terms:
                response_terms = set(re.findall(r'\b\w{4,}\b', response_lower))
                coverage = len(response_terms.intersection(question_terms)) / len(question_terms)
                
                if coverage < 0.3:
                    score -= 0.2
                    issues.append("Response doesn't address key aspects of question")
        
        return max(0.0, min(1.0, score)), issues
    
    def check_grounding(self, response: str, context: str) -> Tuple[float, List[str]]:
        """
        Check if response is grounded in provided context.
        Look for hallucinations or unsupported claims.
        """
        issues = []
        score = 1.0
        
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Extract phrases from response
        phrases = re.findall(r'[^.!?]*[.!?]', response)
        
        grounded_phrases = 0
        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            if len(phrase_lower) > 10:
                # Check if phrase or similar concepts appear in context
                words = re.findall(r'\w{3,}', phrase_lower)
                word_overlap = sum(1 for w in words if w in context_lower)
                
                if word_overlap > 0:
                    grounded_phrases += 1
        
        grounding_ratio = grounded_phrases / len(phrases) if phrases else 1.0
        
        if grounding_ratio < 0.5:
            score -= 0.4
            issues.append("Some claims may not be grounded in source documents")
        elif grounding_ratio < 0.7:
            score -= 0.2
            issues.append("Some parts lack clear grounding in sources")
        
        return max(0.0, min(1.0, score)), issues
    
    def check_clarity(self, response: str) -> Tuple[float, List[str]]:
        """
        Check if response is clear and well-structured.
        """
        issues = []
        score = 1.0
        
        # Check for structure (paragraphs, bullet points)
        has_structure = '\n' in response or '•' in response or '-' in response
        if not has_structure and len(response) > 200:
            score -= 0.1
            issues.append("Could benefit from better formatting/structure")
        
        # Check sentence length (too long = hard to read)
        sentences = re.split(r'[.!?]', response)
        long_sentences = [s for s in sentences if len(s) > 100]
        
        if long_sentences and len(long_sentences) / len(sentences) > 0.3:
            score -= 0.15
            issues.append("Some sentences are too long - try breaking them up")
        
        # Check for ambiguous pronouns
        pronouns = ['nó', 'nó', 'chúng nó', 'cái']
        for pronoun in pronouns:
            if pronoun in response.lower():
                score -= 0.05
                issues.append(f"Pronoun '{pronoun}' could be ambiguous")
        
        return max(0.0, min(1.0, score)), issues
    
    def check_length(self, response: str) -> Tuple[float, List[str]]:
        """
        Check if response length is appropriate.
        Too short = incomplete, too long = unnecessary detail.
        """
        issues = []
        score = 1.0
        
        word_count = len(response.split())
        
        if word_count < 20:
            score -= 0.4
            issues.append(f"Response too short ({word_count} words) - expand with more detail")
        elif word_count < 50:
            score -= 0.2
            issues.append(f"Response could be more detailed ({word_count} words)")
        elif word_count > 300:
            score -= 0.15
            issues.append(f"Response is long ({word_count} words) - can trim unnecessary parts")
        else:
            score = 1.0  # Ideal range: 50-300 words
        
        return max(0.0, min(1.0, score)), issues
    
    def check_vietnamese(self, response: str) -> Tuple[float, List[str]]:
        """
        Check if response is proper Vietnamese.
        """
        issues = []
        score = 1.0
        
        # Check for excessive English
        english_words = len(re.findall(r'\b[a-z]+\b', response.lower()))
        total_words = len(response.split())
        english_ratio = english_words / total_words if total_words > 0 else 0
        
        if english_ratio > 0.3:
            score -= 0.2
            issues.append("Too much English text - prefer Vietnamese")
        
        # Check for common mistakes
        mistakes = {
            r'\blà\s+của\b': 'Redundant "là của" - use just "của"',
            r'\bNhưng\s+tuy\b': 'Redundant "Nhưng tuy" - use just "Tuy"',
            r'\bđể\s+để\b': 'Repeated "để"',
        }
        
        for pattern, msg in mistakes.items():
            if re.search(pattern, response):
                score -= 0.1
                issues.append(msg)
        
        return max(0.0, min(1.0, score)), issues
    
    def assess_overall(self, question: str, response: str, context: str) -> QualityScore:
        """
        Overall quality assessment combining all checks.
        """
        # Run all checks
        completeness_score, completeness_issues = self.check_completeness(question, response)
        grounding_score, grounding_issues = self.check_grounding(response, context)
        clarity_score, clarity_issues = self.check_clarity(response)
        length_score, length_issues = self.check_length(response)
        language_score, language_issues = self.check_vietnamese(response)
        
        # Weighted average
        overall = (
            completeness_score * 0.25 +
            grounding_score * 0.30 +
            clarity_score * 0.20 +
            length_score * 0.15 +
            language_score * 0.10
        )
        
        # Combine all issues and suggestions
        all_issues = (
            completeness_issues +
            grounding_issues +
            clarity_issues +
            length_issues +
            language_issues
        )
        
        suggestions = self._generate_suggestions(all_issues)
        
        return QualityScore(
            completeness=completeness_score,
            grounding=grounding_score,
            clarity=clarity_score,
            length=length_score,
            language=language_score,
            overall=overall,
            issues=all_issues,
            suggestions=suggestions
        )
    
    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on issues"""
        suggestions = []
        
        if any('too short' in i.lower() for i in issues):
            suggestions.append("✏️ Expand answer: Add more details, examples, or context")
        
        if any('not grounded' in i.lower() for i in issues):
            suggestions.append("📖 Check sources: Ensure claims are from provided documents")
        
        if any('structure' in i.lower() for i in issues):
            suggestions.append("📋 Format better: Use bullets, paragraphs, or numbered lists")
        
        if any('english' in i.lower() for i in issues):
            suggestions.append("🇻🇳 Use Vietnamese: Replace English terms with Vietnamese equivalents")
        
        if any('long sentence' in i.lower() for i in issues):
            suggestions.append("📝 Simplify: Break long sentences into shorter, clearer ones")
        
        return suggestions
    
    def is_acceptable(self, quality_score: QualityScore) -> bool:
        """Check if response meets minimum quality threshold"""
        return quality_score.overall >= self.min_acceptable_score


class ResponseImprover:
    """Suggests and applies improvements to responses"""
    
    @staticmethod
    def improve_length(response: str, target_words: int = 80) -> str:
        """
        Adjust response length.
        If too short, return as-is (LLM should regenerate).
        If too long, trim to target.
        """
        words = response.split()
        
        if len(words) < target_words:
            return response  # Too short - needs regeneration
        
        # Trim to target, keeping complete sentences
        trimmed = ' '.join(words[:target_words])
        
        # Ensure we end at sentence boundary
        last_sentence_end = max(
            trimmed.rfind('.'),
            trimmed.rfind('!'),
            trimmed.rfind('?')
        )
        
        if last_sentence_end > target_words * 0.8:
            trimmed = trimmed[:last_sentence_end + 1]
        
        return trimmed
    
    @staticmethod
    def improve_clarity(response: str) -> str:
        """
        Improve response clarity by:
        - Replacing ambiguous pronouns
        - Adding formatting
        - Simplifying complex sentences
        """
        # Replace ambiguous pronouns
        response = re.sub(r'\bnó\b', 'nó/vấn đề này', response)
        
        # Add structure if response is long and lacks it
        if len(response) > 200 and '\n' not in response:
            # Try to split into logical sections
            sentences = re.split(r'([.!?])', response)
            
            if len(sentences) > 6:  # Multiple sentences
                # Group into paragraphs
                sections = []
                current = []
                for sent in sentences:
                    current.append(sent)
                    if len(current) >= 4:  # 2 sentences per paragraph
                        sections.append(''.join(current).strip())
                        current = []
                if current:
                    sections.append(''.join(current).strip())
                
                response = '\n\n'.join(sections)
        
        return response
    
    @staticmethod
    def add_citations(response: str, source_pages: List[int]) -> str:
        """
        Add source citations to response.
        """
        if not source_pages:
            return response
        
        page_text = f"[Trang: {', '.join(map(str, sorted(set(source_pages))))}]"
        
        if response.endswith('.'):
            response = response[:-1] + f" {page_text}."
        else:
            response = f"{response} {page_text}"
        
        return response


# Example usage for testing
if __name__ == "__main__":
    validator = ResponseValidator()
    
    question = "Bao nhiêu ngày nghỉ phép mỗi năm?"
    response = "Mỗi nhân viên được hưởng 12 ngày nghỉ phép hàng năm."
    context = "Theo điều 5, mỗi nhân viên được hưởng 12 ngày nghỉ phép hàng năm."
    
    quality = validator.assess_overall(question, response, context)
    
    print(f"Overall Score: {quality.overall:.1%}")
    print(f"Issues: {quality.issues}")
    print(f"Suggestions: {quality.suggestions}")
    print(f"Acceptable: {validator.is_acceptable(quality)}")
