"""
Smart Context Retrieval & Ranking
Multi-stage ranking to select the BEST chunks for HR Q&A
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# HR Domain Keywords for better relevance detection
HR_KEYWORDS = {
    "leave": ["nghỉ phép", "kỳ nghỉ", "ngày nghỉ", "phép annual", "hưởng phép"],
    "sick_leave": ["nghỉ ốm", "ngày ốm", "chứng chỉ y tế", "đơn xin nghỉ ốm"],
    "salary": ["lương", "hạch toán lương", "bảng lương", "trả lương", "thưởng"],
    "overtime": ["làm thêm giờ", "tăng ca", "overtime", "phụ cấp tăng ca"],
    "benefits": ["bảo hiểm", "phúc lợi", "trợ cấp", "chế độ"],
    "contract": ["hợp đồng", "ký hợp đồng", "điều khoản hợp đồng", "kết thúc hợp đồng"],
    "attendance": ["chấm công", "điểm danh", "vắng mặt", "có mặt"],
    "discipline": ["kỷ luật", "vi phạm", "hình phạt", "cảnh cáo"],
    "promotion": ["thăng chức", "nâng bậc", "nâng lương", "xếp loại"],
    "resignation": ["từ chức", "thôi việc", "xin nghỉ", "kỳ báo trước"],
    "training": ["đào tạo", "huấn luyện", "khóa học", "nâng cao kỹ năng"],
    "health": ["sức khỏe", "an toàn", "y tế", "khám sức khỏe"],
}

@dataclass
class RankedChunk:
    """Chunk with detailed ranking scores"""
    text: str
    metadata: Dict
    semantic_score: float  # From embeddings (0-1)
    keyword_score: float   # From HR keyword matching (0-1)
    relevance_score: float # From context understanding (0-1)
    combined_score: float  # Weighted average
    rank_reason: str      # Why it ranked high


class SmartContextRetriever:
    """Multi-stage context ranking for better retrieval"""
    
    def __init__(self):
        self.hr_keywords = HR_KEYWORDS
        
    def extract_keywords_from_question(self, question: str) -> List[str]:
        """Extract HR domain keywords from question"""
        question_lower = question.lower()
        keywords = []
        
        for category, terms in self.hr_keywords.items():
            for term in terms:
                if term in question_lower:
                    keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates
    
    def score_keyword_relevance(self, chunk_text: str, question_keywords: List[str]) -> float:
        """
        Score chunk based on keyword overlap with question.
        Returns 0-1 score.
        """
        if not question_keywords:
            return 0.5  # Neutral if no keywords
        
        chunk_lower = chunk_text.lower()
        matches = sum(1 for kw in question_keywords if kw in chunk_lower)
        
        # Normalize: matches / total_keywords, max 1.0
        score = min(1.0, matches / len(question_keywords) if question_keywords else 0)
        return score
    
    def score_semantic_coherence(self, chunk_text: str, context_tokens: int) -> float:
        """
        Score chunk for coherence within context.
        - Complete sentences preferred
        - Proper length (not too short/long)
        - Minimal fragmentation
        """
        # Check for complete sentences
        sentences = re.split(r'[.!?]', chunk_text)
        complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(complete_sentences) < 2:
            return 0.5  # Fragment, moderate score
        
        # Check length appropriateness
        word_count = len(chunk_text.split())
        ideal_words = context_tokens // 4  # Target ~25% of context per chunk
        
        if 20 <= word_count <= 300:
            length_score = 1.0
        elif 10 <= word_count <= 400:
            length_score = 0.8
        else:
            length_score = 0.5
        
        return length_score
    
    def score_specificity(self, chunk_text: str, question: str) -> float:
        """
        Check if chunk is specific to the question, not generic.
        High score for specific policy info, low for generic text.
        """
        # Count specific terms (numbers, dates, policies)
        specific_indicators = re.findall(
            r'\d+|ngày|tháng|năm|điều \d+|khoản|phần|chương',
            chunk_text.lower()
        )
        
        specificity = min(1.0, len(specific_indicators) / 3)
        
        # Penalize generic responses
        generic_phrases = ["có thể", "tùy thuộc", "nói chung", "thường"]
        has_generic = any(phrase in chunk_text.lower() for phrase in generic_phrases)
        
        if has_generic:
            specificity = max(0.3, specificity - 0.3)
        
        return specificity
    
    def rank_chunks(
        self,
        chunks: List[Dict],
        question: str,
        semantic_scores: List[float],
        weights: Dict = None
    ) -> List[RankedChunk]:
        """
        Multi-stage ranking of chunks.
        
        Args:
            chunks: List of retrieved chunks with metadata
            question: Original user question
            semantic_scores: Pre-computed semantic similarity scores (0-1)
            weights: Custom scoring weights
            
        Returns:
            List of RankedChunk objects, sorted by combined_score (descending)
        """
        if weights is None:
            weights = {
                'semantic': 0.4,      # Semantic similarity
                'keyword': 0.35,      # HR domain keyword match
                'specificity': 0.15,  # Specific vs generic
                'coherence': 0.1,     # Context coherence
            }
        
        # Validate weights sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"
        
        # Extract question keywords
        question_keywords = self.extract_keywords_from_question(question)
        context_tokens = sum(len(c['text'].split()) for c in chunks)
        
        ranked_chunks = []
        
        for idx, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Get pre-computed semantic score
            sem_score = semantic_scores[idx] if idx < len(semantic_scores) else 0.5
            
            # Compute other scores
            kw_score = self.score_keyword_relevance(text, question_keywords)
            spec_score = self.score_specificity(text, question)
            coh_score = self.score_semantic_coherence(text, context_tokens)
            
            # Weighted combination
            combined = (
                weights['semantic'] * sem_score +
                weights['keyword'] * kw_score +
                weights['specificity'] * spec_score +
                weights['coherence'] * coh_score
            )
            
            # Determine ranking reason
            if combined > 0.8:
                reason = "🌟 Excellent: Relevant + specific + coherent"
            elif combined > 0.6:
                reason = "✅ Good: Relevant content"
            elif combined > 0.4:
                reason = "⚠️ Fair: Some relevance detected"
            else:
                reason = "❌ Poor: Low relevance"
            
            ranked = RankedChunk(
                text=text,
                metadata=metadata,
                semantic_score=sem_score,
                keyword_score=kw_score,
                relevance_score=spec_score,
                combined_score=combined,
                rank_reason=reason
            )
            ranked_chunks.append(ranked)
        
        # Sort by combined score descending
        ranked_chunks.sort(key=lambda x: x.combined_score, reverse=True)
        
        return ranked_chunks
    
    def select_best_chunks(
        self,
        ranked_chunks: List[RankedChunk],
        top_k: int = 3,
        min_score: float = 0.3
    ) -> List[RankedChunk]:
        """
        Select the best chunks with quality threshold.
        
        Args:
            ranked_chunks: Already ranked chunks
            top_k: Number of top chunks to return
            min_score: Minimum combined score to include
            
        Returns:
            List of selected chunks (up to top_k, all with score > min_score)
        """
        # Filter by minimum score
        filtered = [c for c in ranked_chunks if c.combined_score >= min_score]
        
        # Return top_k
        return filtered[:top_k]
    
    def format_context_with_scores(self, chunks: List[RankedChunk]) -> str:
        """
        Format selected chunks into readable context with scoring info for debugging.
        
        Args:
            chunks: Selected RankedChunk objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            score_display = f"({chunk.combined_score:.1%} relevance)"
            page_num = chunk.metadata.get('page_num', '?')
            
            context_parts.append(
                f"[Trang {page_num}] {chunk.text[:300]}"
            )
        
        return "\n\n".join(context_parts)
    
    def explain_ranking(self, ranked_chunks: List[RankedChunk], top_k: int = 3) -> str:
        """
        Generate human-readable explanation of why chunks were selected.
        Useful for debugging.
        """
        explanation = "📊 Context Selection Report\n" + "="*40 + "\n\n"
        
        for i, chunk in enumerate(ranked_chunks[:top_k], 1):
            explanation += (
                f"{i}. {chunk.rank_reason}\n"
                f"   Semantic: {chunk.semantic_score:.0%} | "
                f"Keyword: {chunk.keyword_score:.0%} | "
                f"Specificity: {chunk.relevance_score:.0%}\n"
                f"   Combined Score: {chunk.combined_score:.0%}\n"
                f"   Text: {chunk.text[:80]}...\n\n"
            )
        
        return explanation


if __name__ == "__main__":
    # Example usage
    retriever = SmartContextRetriever()
    
    # Sample chunks
    chunks = [
        {
            'text': 'Mỗi nhân viên được hưởng 12 ngày nghỉ phép hàng năm theo quy định của pháp luật',
            'metadata': {'page_num': 5}
        },
        {
            'text': 'Các chính sách công ty được áp dụng cho tất cả nhân viên',
            'metadata': {'page_num': 1}
        }
    ]
    
    question = "Bao nhiêu ngày nghỉ phép mỗi năm?"
    semantic_scores = [0.95, 0.3]
    
    ranked = retriever.rank_chunks(chunks, question, semantic_scores)
    selected = retriever.select_best_chunks(ranked, top_k=2)
    
    print(retriever.explain_ranking(ranked))
    print("\nSelected Context:")
    print(retriever.format_context_with_scores(selected))
