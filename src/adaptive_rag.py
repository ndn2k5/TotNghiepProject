"""
Adaptive RAG with Question Router
Classifies questions and routes to optimal retrieval strategy
"""

import logging
from typing import Dict, List, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions for routing"""
    FACTUAL = "factual"           # "What is...", "How many..."
    PROCEDURAL = "procedural"     # "How to...", "What's the process..."
    COMPARATIVE = "comparative"   # "Difference between...", "Compare..."
    ANALYTICAL = "analytical"     # "Why...", "What causes..."
    POLICY = "policy"             # "Policy for...", "Rules about..."
    UNCLEAR = "unclear"           # Can't determine
    

class QuestionRouter:
    """
    Routes questions to optimal RAG strategy based on question type
    This is the core of Adaptive RAG
    """
    
    # Keywords for each question type
    TYPE_KEYWORDS = {
        QuestionType.FACTUAL: [
            'bao nhiêu', 'là gì', 'là ai', 'khi nào', 'ở đâu', 'cái nào',
            'what', 'how many', 'when', 'where', 'who'
        ],
        QuestionType.PROCEDURAL: [
            'làm thế nào', 'quy trình', 'bước', 'cách', 'hướng dẫn',
            'how to', 'process', 'step', 'procedure'
        ],
        QuestionType.COMPARATIVE: [
            'so sánh', 'khác', 'giống', 'giữa', 'khác biệt',
            'tương tự', 'difference', 'compare', 'vs', 'versus'
        ],
        QuestionType.ANALYTICAL: [
            'tại sao', 'vì sao', 'nguyên nhân', 'lý do', 'kết quả',
            'why', 'cause', 'reason', 'effect', 'consequence'
        ],
        QuestionType.POLICY: [
            'chính sách', 'quy định', 'điều khoản', 'luật', 'quy tắc',
            'policy', 'regulation', 'rule', 'requirement', 'provision'
        ]
    }
    
    @staticmethod
    def classify_question(question: str) -> QuestionType:
        """
        Classify question type based on keywords
        """
        q_lower = question.lower()
        
        # Count keyword matches for each type
        type_scores = {}
        
        for q_type, keywords in QuestionRouter.TYPE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in q_lower)
            type_scores[q_type] = score
        
        # Return type with highest score
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        return QuestionType.UNCLEAR
    
    @staticmethod
    def get_retrieval_strategy(q_type: QuestionType) -> Dict:
        """
        Get optimal retrieval strategy for question type
        
        Returns configuration dict with:
        - top_k: number of documents
        - rerank: whether to use re-ranking
        - chunking: chunk strategy
        - generation: generation strategy
        """
        
        strategies = {
            QuestionType.FACTUAL: {
                'top_k': 3,
                'use_rerank': True,
                'chunking': 'semantic',
                'generation': 'stuff',
                'temperature': 0.1,
                'description': 'Factual question - retrieve precise info'
            },
            QuestionType.PROCEDURAL: {
                'top_k': 5,
                'use_rerank': True,
                'chunking': 'small-to-big',
                'generation': 'refine',
                'temperature': 0.2,
                'description': 'Procedural question - need step-by-step'
            },
            QuestionType.COMPARATIVE: {
                'top_k': 6,
                'use_rerank': True,
                'chunking': 'small-to-big',
                'generation': 'map-reduce',
                'temperature': 0.3,
                'description': 'Comparative question - need multiple docs'
            },
            QuestionType.ANALYTICAL: {
                'top_k': 5,
                'use_rerank': False,  # Need full context for analysis
                'chunking': 'semantic',
                'generation': 'map-reduce',
                'temperature': 0.4,
                'description': 'Analytical question - need reasoning'
            },
            QuestionType.POLICY: {
                'top_k': 4,
                'use_rerank': True,
                'chunking': 'semantic',
                'generation': 'stuff',
                'temperature': 0.1,
                'description': 'Policy question - need exact rules'
            },
            QuestionType.UNCLEAR: {
                'top_k': 4,
                'use_rerank': True,
                'chunking': 'semantic',
                'generation': 'refine',
                'temperature': 0.2,
                'description': 'Unclear question - use default strategy'
            }
        }
        
        return strategies.get(q_type, strategies[QuestionType.UNCLEAR])


class AdaptiveRAG:
    """
    Adaptive RAG: Routes questions to optimal retrieval + generation strategy
    
    This is one of the highest-scoring RAG techniques (score: 0.86/1)
    """
    
    def __init__(self, retriever, generator, reranker=None):
        """
        Initialize Adaptive RAG
        
        Args:
            retriever: Hybrid retriever with search capability
            generator: Generation strategy selector
            reranker: Optional re-ranker
        """
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.router = QuestionRouter()
        self.query_log = []  # For analytics
    
    def answer(self, question: str) -> Dict:
        """
        Answer using adaptive RAG strategy
        
        Flow:
        1. Classify question type
        2. Get optimal retrieval strategy
        3. Retrieve documents with strategy
        4. Optional: Re-rank if needed
        5. Generate with appropriate strategy
        """
        
        # Step 1: Classify question
        q_type = self.router.classify_question(question)
        logger.info(f"Question type: {q_type.value}")
        
        # Step 2: Get strategy
        strategy = self.router.get_retrieval_strategy(q_type)
        logger.info(f"Strategy: {strategy['description']}")
        
        # Step 3: Retrieve documents
        retrieved = self.retriever.search(
            question,
            top_k=strategy['top_k']
        )
        
        doc_texts = [doc['text'] for doc in retrieved]
        
        # Step 4: Re-rank if needed
        if self.reranker and strategy['use_rerank']:
            retrieved = self.reranker.rerank(question, retrieved, top_k=strategy['top_k'])
            doc_texts = [doc['text'] for doc in retrieved]
        
        # Step 5: Generate with strategy
        generation_result = self.generator.generate(
            question,
            doc_texts,
            strategy=strategy['generation']
        )
        
        # Combine results
        result = {
            'question': question,
            'question_type': q_type.value,
            'strategy_used': strategy['generation'],
            'answer': generation_result['answer'],
            'num_documents_retrieved': len(retrieved),
            'confidence': self._calculate_confidence(retrieved, q_type),
            'source_documents': [doc.get('metadata', {}) for doc in retrieved]
        }
        
        # Log for analytics
        self.query_log.append(result)
        
        return result
    
    def _calculate_confidence(self, documents: List[Dict], q_type: QuestionType) -> str:
        """Calculate confidence based on documents and question type"""
        
        if not documents:
            return 'low'
        
        # Get average relevance score
        scores = [doc.get('rerank_score', 0.5) for doc in documents]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # For factual questions, high confidence needs very good matches
        if q_type == QuestionType.FACTUAL:
            if avg_score >= 0.8:
                return 'high'
            elif avg_score >= 0.6:
                return 'medium'
            else:
                return 'low'
        
        # For other types, lower threshold
        if avg_score >= 0.7:
            return 'high'
        elif avg_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def get_analytics(self) -> Dict:
        """Get analytics on routing decisions"""
        
        type_counts = {}
        strategy_counts = {}
        
        for query in self.query_log:
            q_type = query['question_type']
            strategy = query['strategy_used']
            
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_queries': len(self.query_log),
            'question_type_distribution': type_counts,
            'strategy_distribution': strategy_counts,
            'recent_queries': self.query_log[-10:]  # Last 10
        }


if __name__ == "__main__":
    print("Adaptive RAG Module Ready")
    print("- Question Classification ✓")
    print("- Routing Strategy ✓")
    print("- Adaptive Generation ✓")
