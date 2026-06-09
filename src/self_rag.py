# -*- coding: utf-8 -*-
"""
Self-RAG (Self-Reflective RAG)
Model can self-evaluate when to retrieve and whether answer is grounded
"""

import logging
from typing import Dict, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class RetrievalDecision(Enum):
    """Model's decision on whether to retrieve"""
    RETRIEVE = "retrieve"      # Need to retrieve info
    CONTINUE = "continue"      # Already have enough info
    GENERATE = "generate"      # Can answer without retrieval


class GradingDecision(Enum):
    """Model's self-evaluation of answer quality"""
    RELEVANT = "relevant"       # Answer is based on retrieved docs
    PARTIALLY = "partially"     # Partially based on retrieved docs
    IRRELEVANT = "irrelevant"   # Not based on retrieved docs


class SelfRAG:
    """
    Self-RAG: Model decides when to retrieve and self-grades answers
    
    This enables:
    1. Adaptive retrieval (only retrieve when needed)
    2. Self-grading (verify answer is grounded in documents)
    3. Iterative refinement (retrieve again if grade is low)
    """
    
    def __init__(self, llm_model, retriever, max_iterations: int = 3):
        """
        Initialize Self-RAG
        
        Args:
            llm_model: LLM to use
            retriever: Retriever for documents
            max_iterations: Max refinement iterations
        """
        self.llm = llm_model
        self.retriever = retriever
        self.max_iterations = max_iterations
    
    def decide_retrieve(self, question: str, context: str = "") -> Tuple[bool, str]:
        """
        Decide whether to retrieve documents
        
        Returns:
            (should_retrieve, reasoning)
        """
        
        prompt = f"""Xem xét câu hỏi này. Bạn có đủ thông tin để trả lời mà không cần tìm kiếm tài liệu bổ sung không?

Câu hỏi: {question}
{"Ngữ cảnh hiện tại: " + context if context else ""}

Trả lời với "RETRIEVE" nếu cần tìm tài liệu, hoặc "CONTINUE" nếu không cần:"""
        
        try:
            response = self.llm.generate(prompt).strip().upper()
            should_retrieve = "RETRIEVE" in response
            reasoning = response
            return should_retrieve, reasoning
        except Exception as e:
            logger.error(f"Retrieve decision failed: {e}")
            # Default to retrieve if error
            return True, "Error in decision making, retrieving by default"
    
    def generate_answer(
        self,
        question: str,
        documents: List[str] = None,
        context: str = ""
    ) -> str:
        """Generate answer with optional context"""
        
        doc_context = ""
        if documents:
            doc_context = "\n".join([f"- {doc}" for doc in documents])
            doc_context = f"Tài liệu tham khảo:\n{doc_context}\n\n"
        
        prompt = f"""{doc_context}Câu hỏi: {question}

Hãy trả lời dựa trên tài liệu (nếu có). Trả lời ngắn gọn, rõ ràng:"""
        
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Không thể tạo câu trả lời."
    
    def grade_answer(
        self,
        question: str,
        answer: str,
        documents: List[str]
    ) -> Tuple[str, float, str]:
        """
        Self-grade: Evaluate if answer is grounded in documents
        
        Returns:
            (grade, confidence, explanation)
        """
        
        doc_context = "\n".join([f"- {doc}" for doc in documents])
        
        prompt = f"""Đánh giá xem câu trả lời này có dựa trên tài liệu được cung cấp không:

Câu hỏi: {question}

Tài liệu:
{doc_context}

Câu trả lời: {answer}

Chọn một trong ba đánh giá:
1. RELEVANT - Câu trả lời hoàn toàn dựa trên tài liệu
2. PARTIALLY - Câu trả lời một phần dựa trên tài liệu
3. IRRELEVANT - Câu trả lời không dựa trên tài liệu

Đánh giá:"""
        
        try:
            response = self.llm.generate(prompt).strip().upper()
            
            # Extract grade
            if "RELEVANT" in response:
                grade = "relevant"
                confidence = 0.9
            elif "PARTIALLY" in response:
                grade = "partially"
                confidence = 0.6
            else:
                grade = "irrelevant"
                confidence = 0.3
            
            return grade, confidence, response
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            return "uncertain", 0.5, str(e)
    
    def answer_with_reflection(self, question: str) -> Dict:
        """
        Answer using self-reflection:
        1. Decide if retrieval needed
        2. Retrieve if needed
        3. Generate answer
        4. Self-grade answer
        5. Refine if grade is low
        """
        
        logger.info(f"Self-RAG: Answering '{question}'")
        
        iteration = 0
        answer = None
        documents = []
        grade_history = []
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}")
            
            # Step 1: Decide if retrieval needed
            context = answer if answer else ""
            should_retrieve, reasoning = self.decide_retrieve(question, context)
            
            logger.info(f"Retrieve decision: {should_retrieve}")
            
            # Step 2: Retrieve if needed
            if should_retrieve and not documents:
                logger.info("Retrieving documents...")
                retrieved = self.retriever.search(question, top_k=3)
                documents = [doc['text'] for doc in retrieved]
            
            # Step 3: Generate answer
            answer = self.generate_answer(question, documents, context)
            logger.info(f"Generated: {answer[:100]}...")
            
            # Step 4: Self-grade
            if documents:
                grade, confidence, explanation = self.grade_answer(
                    question, answer, documents
                )
                grade_history.append({
                    'iteration': iteration,
                    'grade': grade,
                    'confidence': confidence
                })
                
                logger.info(f"Grade: {grade} (confidence: {confidence})")
                
                # Step 5: Check if we should refine
                if grade == "relevant" or confidence >= 0.7:
                    logger.info("Answer accepted")
                    break
                elif iteration < self.max_iterations:
                    # Try to retrieve again with different approach
                    logger.info("Grade low, retrieving more documents...")
                    retrieved = self.retriever.search(
                        question,
                        top_k=5  # Get more documents
                    )
                    documents = [doc['text'] for doc in retrieved]
            else:
                break
        
        return {
            'question': question,
            'answer': answer,
            'documents_used': len(documents),
            'iterations': iteration,
            'grade_history': grade_history,
            'is_grounded': len(grade_history) > 0 and grade_history[-1]['grade'] == 'relevant',
            'final_confidence': grade_history[-1]['confidence'] if grade_history else 0.5
        }
    
    def batch_answer(self, questions: List[str]) -> List[Dict]:
        """Answer multiple questions with self-reflection"""
        results = []
        for question in questions:
            result = self.answer_with_reflection(question)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about self-RAG performance"""
        return {
            'description': 'Self-RAG enables adaptive retrieval and self-grading',
            'benefits': [
                'Only retrieve when necessary (faster)',
                'Self-evaluate answer quality (more reliable)',
                'Iteratively refine if needed (better accuracy)'
            ]
        }


class SimplifiedSelfRAG:
    """
    Simplified Self-RAG for cases where full reflection is too expensive
    Only does retrieval decision + grading, no iterative refinement
    """
    
    def __init__(self, llm_model, retriever):
        self.llm = llm_model
        self.retriever = retriever
    
    def answer(self, question: str) -> Dict:
        """Quick self-RAG without iteration"""
        
        # Always retrieve for now (can optimize later)
        retrieved = self.retriever.search(question, top_k=3)
        documents = [doc['text'] for doc in retrieved]
        
        # Generate
        doc_context = "\n".join([f"- {doc}" for doc in documents])
        
        prompt = f"""Tài liệu:
{doc_context}

Câu hỏi: {question}

Trả lời ngắn gọn dựa trên tài liệu:"""
        
        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            answer = "Lỗi khi tạo câu trả lời."
        
        # Quick grade
        grade_prompt = f"""Câu trả lời này có dựa trên tài liệu được cung cấp không?
Tài liệu: {doc_context}
Câu trả lời: {answer}

Trả lời: CÓ hoặc KHÔNG"""
        
        try:
            grade_response = self.llm.generate(grade_prompt).upper()
            is_grounded = "CÓ" in grade_response or "YES" in grade_response
        except:
            is_grounded = True  # Default to true if error
        
        return {
            'question': question,
            'answer': answer,
            'documents_used': len(documents),
            'is_grounded': is_grounded
        }


if __name__ == "__main__":
    print("Self-RAG Module Ready")
    print("- Retrieval Decision ✓")
    print("- Self-Grading ✓")
    print("- Iterative Refinement ✓")
    print("- Simplified Version ✓")
