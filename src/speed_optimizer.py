"""
Speed Optimization Module
Multiple speed modes for dual-model pipeline
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import time

logger = logging.getLogger(__name__)


class SpeedMode(Enum):
    """Available speed modes"""
    FULL = "full"           # 2-3s: Full dual-model with Phi-3 research
    FAST = "fast"           # 1-2s: Skip Phi-3, keep quality
    SUPER_FAST = "super"    # 0.8-1.2s: Reduce chunks + tokens
    HYBRID = "hybrid"       # 0.1-3s: Smart routing with caching


class ResponseCache:
    """Simple in-memory cache for common questions"""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
    
    def _hash_question(self, question: str) -> str:
        """Generate hash of question (case-insensitive)"""
        normalized = question.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def get(self, question: str) -> Optional[str]:
        """Get cached answer"""
        key = self._hash_question(question)
        if key in self.cache:
            self.hit_count += 1
            logger.debug(f"Cache HIT: {question}")
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, question: str, answer: str) -> None:
        """Cache an answer"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        key = self._hash_question(question)
        self.cache[key] = answer
        logger.debug(f"Cache SET: {question}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'total': total
        }
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


class QuestionClassifier:
    """Classify questions for optimal speed mode"""
    
    # Common question patterns (fast to answer)
    COMMON_PATTERNS = {
        'leave': ['nghỉ phép', 'kỳ nghỉ', 'ngày nghỉ', 'hưởng phép'],
        'salary': ['lương', 'hạch toán lương', 'bảng lương', 'trả lương'],
        'overtime': ['làm thêm', 'tăng ca', 'phụ cấp tăng ca'],
        'simple': ['bao nhiêu', 'là gì', 'có phải'],  # Simple questions
    }
    
    # Complex question patterns (need full pipeline)
    COMPLEX_PATTERNS = {
        'policy': ['chính sách', 'quy định', 'điều khoản', 'chương trình'],
        'procedure': ['quy trình', 'thủ tục', 'bước', 'làm như thế nào'],
        'comparison': ['so sánh', 'khác', 'giống', 'tương tự'],
        'complex': ['tại sao', 'liên quan', 'ảnh hưởng', 'kết quả'],
    }
    
    @staticmethod
    def classify(question: str) -> str:
        """
        Classify question type
        
        Returns:
            'simple': Use fast mode
            'complex': Use full mode
            'moderate': Use super-fast mode
        """
        q_lower = question.lower()
        
        # Check complex patterns first (stricter)
        complex_score = sum(
            1 for patterns in QuestionClassifier.COMPLEX_PATTERNS.values()
            for pattern in patterns
            if pattern in q_lower
        )
        
        # Check common patterns
        common_score = sum(
            1 for patterns in QuestionClassifier.COMMON_PATTERNS.values()
            for pattern in patterns
            if pattern in q_lower
        )
        
        # Classify based on scores
        if complex_score >= 2:
            return 'complex'
        elif common_score >= 2 and complex_score == 0:
            return 'simple'
        else:
            return 'moderate'


class SpeedOptimizedPipeline:
    """
    Wrapper around DualModelPipeline with speed optimization
    Supports multiple speed modes
    """
    
    def __init__(self, pipeline, speed_mode: str = "hybrid"):
        """
        Initialize speed-optimized pipeline
        
        Args:
            pipeline: DualModelPipeline instance
            speed_mode: 'full', 'fast', 'super', 'hybrid'
        """
        self.pipeline = pipeline
        self.speed_mode = SpeedMode(speed_mode) if isinstance(speed_mode, str) else speed_mode
        self.cache = ResponseCache(max_size=100)
        self.classifier = QuestionClassifier()
        self.mode_times = {}  # Track time by mode
        
        logger.info(f"🚀 Speed Mode: {self.speed_mode.value}")
    
    def _get_effective_mode(self, question: str) -> SpeedMode:
        """Determine effective mode based on question and selected mode"""
        
        if self.speed_mode == SpeedMode.HYBRID:
            # Smart routing based on question type
            q_type = self.classifier.classify(question)
            
            if q_type == 'simple':
                return SpeedMode.SUPER_FAST
            elif q_type == 'complex':
                return SpeedMode.FULL
            else:
                return SpeedMode.FAST
        
        return self.speed_mode
    
    def answer(self, question: str):
        """
        Answer question with speed optimization
        
        Flow:
        1. Check cache (if hybrid mode)
        2. Determine speed mode
        3. Run pipeline with optimizations
        4. Cache result (if hybrid mode)
        """
        import time
        start_time = time.time()
        
        # Step 1: Check cache (hybrid mode only)
        if self.speed_mode == SpeedMode.HYBRID:
            cached = self.cache.get(question)
            if cached:
                elapsed = time.time() - start_time
                return {
                    **cached,
                    'processing_time': elapsed,
                    'mode': 'cached',
                    'speed_source': 'cache'
                }
        
        # Step 2: Determine effective mode
        effective_mode = self._get_effective_mode(question)
        
        # Step 3: Run pipeline with optimizations
        result = self._answer_with_mode(question, effective_mode)
        
        elapsed = time.time() - start_time
        result['processing_time'] = elapsed
        result['mode'] = effective_mode.value
        
        # Step 4: Cache result (hybrid mode only)
        if self.speed_mode == SpeedMode.HYBRID:
            self.cache.set(question, result)
        
        # Track timing
        if effective_mode.value not in self.mode_times:
            self.mode_times[effective_mode.value] = []
        self.mode_times[effective_mode.value].append(elapsed)
        
        return result
    
    def _answer_with_mode(self, question: str, mode: SpeedMode):
        """Run pipeline with specific optimizations"""
        
        if mode == SpeedMode.FULL:
            # Full pipeline: Phi-3 research + Qwen response + validation
            logger.debug("Using FULL mode (2-3s)")
            self.pipeline.use_phi3 = True
            self.pipeline.retriever.top_k = 5
            self.pipeline.qwen_model.max_tokens = 128
            
        elif mode == SpeedMode.FAST:
            # Skip Phi-3, keep quality (1-2s)
            logger.debug("Using FAST mode (1-2s)")
            self.pipeline.use_phi3 = False  # Skip research
            self.pipeline.retriever.top_k = 3
            self.pipeline.qwen_model.max_tokens = 128
            
        elif mode == SpeedMode.SUPER_FAST:
            # Minimize everything (0.8-1.2s)
            logger.debug("Using SUPER_FAST mode (0.8-1.2s)")
            self.pipeline.use_phi3 = False  # Skip research
            self.pipeline.retriever.top_k = 2  # Fewer chunks
            self.pipeline.qwen_model.max_tokens = 80  # Shorter response
        
        # Run pipeline
        result = self.pipeline.answer(question)
        
        # Convert to dict if needed
        if hasattr(result, '__dict__'):
            return result.__dict__
        return result
    
    def set_mode(self, mode: str) -> None:
        """Change speed mode"""
        self.speed_mode = SpeedMode(mode) if isinstance(mode, str) else mode
        logger.info(f"Speed mode changed to: {self.speed_mode.value}")
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        avg_times = {}
        for mode, times in self.mode_times.items():
            if times:
                avg_times[mode] = sum(times) / len(times)
        
        cache_stats = self.cache.get_stats()
        
        return {
            'current_mode': self.speed_mode.value,
            'average_times': avg_times,
            'cache': cache_stats,
            'question_classification': {
                'simple': 'Uses SUPER_FAST mode',
                'moderate': 'Uses FAST mode',
                'complex': 'Uses FULL mode'
            }
        }
    
    def clear_cache(self) -> None:
        """Clear response cache"""
        self.cache.clear()
        logger.info("Response cache cleared")


if __name__ == "__main__":
    # Test classification
    classifier = QuestionClassifier()
    
    test_questions = [
        "Bao nhiêu ngày nghỉ phép mỗi năm?",  # simple
        "Tại sao công ty có chính sách này?",  # complex
        "Lương cơ bản được tính như thế nào?",  # moderate
    ]
    
    for q in test_questions:
        q_type = classifier.classify(q)
        print(f"Q: {q}")
        print(f"   Type: {q_type}\n")
