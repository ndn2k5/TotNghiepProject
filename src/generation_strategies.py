"""
Map-Reduce Generation Strategy
Process multiple documents independently then reduce to final answer
"""

import logging
from typing import List, Dict
import asyncio

logger = logging.getLogger(__name__)


class MapReduceGenerator:
    """
    Map-Reduce approach for generation:
    1. Map: Process each document independently
    2. Reduce: Combine results into final answer
    """
    
    def __init__(self, llm_model):
        """
        Initialize map-reduce generator
        
        Args:
            llm_model: LLM model instance
        """
        self.llm = llm_model
    
    def _map_step(self, query: str, document: str) -> str:
        """
        Map: Extract relevant information from single document
        
        Asks the LLM: "Given this document and question, what's the key info?"
        """
        map_prompt = f"""Dựa trên tài liệu dưới đây, hãy trích xuất thông tin chính để trả lời câu hỏi:

Câu hỏi: {query}

Tài liệu:
{document}

Thông tin chính (ngắn gọn, 1-2 câu):"""
        
        try:
            result = self.llm.generate(map_prompt)
            return result
        except Exception as e:
            logger.error(f"Map step failed: {e}")
            return ""
    
    def _reduce_step(self, query: str, mapped_results: List[str]) -> str:
        """
        Reduce: Combine all mapped results into final answer
        
        Asks the LLM: "Combining these pieces of info, what's the full answer?"
        """
        # Filter out empty results
        valid_results = [r for r in mapped_results if r.strip()]
        
        if not valid_results:
            return "Không tìm thấy thông tin liên quan."
        
        combined_info = "\n".join(
            [f"- {r}" for r in valid_results]
        )
        
        reduce_prompt = f"""Dựa trên các thông tin dưới đây, hãy tổng hợp câu trả lời hoàn chỉnh cho câu hỏi:

Câu hỏi: {query}

Thông tin đã trích xuất:
{combined_info}

Câu trả lời hoàn chỉnh (tiếng Việt, rõ ràng, ngắn gọn):"""
        
        try:
            result = self.llm.generate(reduce_prompt)
            return result
        except Exception as e:
            logger.error(f"Reduce step failed: {e}")
            return "Lỗi khi tạo câu trả lời."
    
    def generate(self, query: str, documents: List[str]) -> Dict:
        """
        Generate answer using map-reduce strategy
        
        Args:
            query: User question
            documents: List of relevant documents
            
        Returns:
            {
                'answer': final answer,
                'map_results': intermediate results from map step,
                'strategy': 'map-reduce'
            }
        """
        if not documents:
            return {
                'answer': 'Không tìm thấy tài liệu liên quan.',
                'map_results': [],
                'strategy': 'map-reduce'
            }
        
        logger.info(f"Map-Reduce: processing {len(documents)} documents")
        
        # Map step: Process each document
        map_results = []
        for i, doc in enumerate(documents):
            logger.debug(f"Map step {i+1}/{len(documents)}")
            result = self._map_step(query, doc)
            if result.strip():
                map_results.append(result)
        
        # Reduce step: Combine results
        logger.debug("Reduce step: combining results")
        final_answer = self._reduce_step(query, map_results)
        
        return {
            'answer': final_answer,
            'map_results': map_results,
            'num_documents': len(documents),
            'strategy': 'map-reduce'
        }


class StuffGenerator:
    """
    Stuff strategy (original): Put all docs into context at once
    (for comparison)
    """
    
    def __init__(self, llm_model):
        self.llm = llm_model
    
    def generate(self, query: str, documents: List[str]) -> Dict:
        """Generate using stuff strategy"""
        
        if not documents:
            return {'answer': 'Không tìm thấy tài liệu.', 'strategy': 'stuff'}
        
        combined_docs = "\n\n".join(documents)
        
        prompt = f"""Dựa trên thông tin dưới đây, hãy trả lời câu hỏi:

Tài liệu:
{combined_docs}

Câu hỏi: {query}

Trả lời (tiếng Việt):"""
        
        try:
            answer = self.llm.generate(prompt)
            return {
                'answer': answer,
                'num_documents': len(documents),
                'strategy': 'stuff'
            }
        except Exception as e:
            logger.error(f"Stuff generation failed: {e}")
            return {'answer': 'Lỗi khi tạo câu trả lời.', 'strategy': 'stuff'}


class RefineGenerator:
    """
    Refine strategy: Process first doc, then refine with others
    """
    
    def __init__(self, llm_model):
        self.llm = llm_model
    
    def generate(self, query: str, documents: List[str]) -> Dict:
        """Generate using refine strategy"""
        
        if not documents:
            return {'answer': 'Không tìm thấy tài liệu.', 'strategy': 'refine'}
        
        # First document
        answer = self._initial_answer(query, documents[0])
        
        # Refine with remaining documents
        for doc in documents[1:]:
            answer = self._refine_answer(query, answer, doc)
        
        return {
            'answer': answer,
            'num_documents': len(documents),
            'strategy': 'refine'
        }
    
    def _initial_answer(self, query: str, document: str) -> str:
        """Generate initial answer from first document"""
        prompt = f"""Dựa trên tài liệu, hãy trả lời câu hỏi:

Tài liệu:
{document}

Câu hỏi: {query}

Trả lời:"""
        
        return self.llm.generate(prompt)
    
    def _refine_answer(self, query: str, previous_answer: str, new_doc: str) -> str:
        """Refine answer with new document"""
        prompt = f"""Câu hỏi: {query}

Câu trả lời trước đó: {previous_answer}

Tài liệu bổ sung:
{new_doc}

Dựa trên tài liệu bổ sung, hãy cải thiện/hoàn thiện câu trả lời (nếu cần):"""
        
        return self.llm.generate(prompt)


class SmartGenerationStrategy:
    """
    Smart choice of generation strategy based on document count
    """
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.map_reduce = MapReduceGenerator(llm_model)
        self.stuff = StuffGenerator(llm_model)
        self.refine = RefineGenerator(llm_model)
    
    def generate(self, query: str, documents: List[str], strategy: str = 'auto') -> Dict:
        """
        Generate answer with smart strategy selection
        
        Args:
            query: Question
            documents: List of documents
            strategy: 'auto', 'map-reduce', 'stuff', 'refine'
        """
        
        if strategy == 'auto':
            # Auto-select based on document count
            if len(documents) <= 2:
                strategy = 'stuff'  # Small context, use stuff
            elif len(documents) <= 5:
                strategy = 'refine'  # Medium context, use refine
            else:
                strategy = 'map-reduce'  # Large context, use map-reduce
            
            logger.info(f"Auto-selected strategy: {strategy} for {len(documents)} docs")
        
        # Generate using selected strategy
        if strategy == 'map-reduce':
            return self.map_reduce.generate(query, documents)
        elif strategy == 'refine':
            return self.refine.generate(query, documents)
        else:  # stuff
            return self.stuff.generate(query, documents)


if __name__ == "__main__":
    print("Map-Reduce Generation Module Ready")
    print("- Map-Reduce Strategy ✓")
    print("- Stuff Strategy ✓")
    print("- Refine Strategy ✓")
    print("- Smart Selection ✓")
