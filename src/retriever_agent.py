# -*- coding: utf-8 -*-
"""
Retriever Agent – AI-powered intelligent document filtering and summarization.

This agent uses a local LLM (e.g., Qwen-2.5-1.5B) to:
1. Read the user's question
2. Analyze retrieved chunks from ChromaDB
3. Filter to only relevant chunks
4. Generate a concise summary of the relevant information
5. Return as JSON for structured processing

This adds an AI layer between raw vector search and final answer generation,
improving relevance and reducing noise in the context passed to the responder.
"""

import json
import logging
from typing import List, Dict, Optional
from src.gguf_models import LocalGGUFModel

logger = logging.getLogger(__name__)


PROMPT_RETRIEVER_VI = """Bạn là chuyên gia phân tích tài liệu nhân sự. Nhiệm vụ của bạn:
1. Đọc câu hỏi của người dùng
2. Xem xét danh sách các đoạn văn bản được lấy từ sổ tay nhân viên
3. Chọn ra những đoạn THỰC SỰ liên quan (ghi số thứ tự)
4. Tóm tắt nội dung những đoạn đó thành một đoạn ngắn gọn, chính xác, không thêm ý kiến
5. Trả lời bằng JSON

Nếu không có đoạn nào liên quan, trả lời: {{"relevant_chunks": [], "summary": "KHÔNG CÓ THÔNG TIN LIÊN QUAN"}}

Câu hỏi: {question}

Các đoạn văn bản từ sổ tay:
{chunks_text}

Trả lời CHỈ bằng JSON (không có text dư):
{{"relevant_chunks": [số_thứ_tự_1, số_thứ_tự_2, ...], "summary": "nội dung tóm tắt"}}"""

PROMPT_RETRIEVER_EN = """You are an HR document analysis expert. Your task:
1. Read the user's question
2. Review the list of document excerpts retrieved from the employee handbook
3. Select ONLY the truly relevant excerpts (record their indices)
4. Summarize those excerpts into one concise, accurate summary without adding opinions
5. Respond in JSON

If no excerpts are relevant, respond: {{"relevant_chunks": [], "summary": "NO RELEVANT INFORMATION"}}

Question: {question}

Document excerpts from handbook:
{chunks_text}

Respond ONLY with JSON (no extra text):
{{"relevant_chunks": [index_1, index_2, ...], "summary": "summary content"}}"""


class RetrieverAgent:
    """
    AI agent that intelligently filters and summarizes retrieved chunks.
    
    This is optional and can be disabled for backward compatibility.
    When disabled, the RAG pipeline works exactly as before.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        language: str = "vi",
        enabled: bool = True
    ):
        """
        Initialize the retriever agent.
        
        Args:
            model_path: Path to the GGUF model (e.g., Qwen-2.5-1.5B)
                       If None, agent is disabled
            language: 'vi' for Vietnamese, 'en' for English
            enabled: If False, agent is completely disabled
        """
        self.language = language
        self.enabled = enabled and model_path is not None
        self.model = None
        
        if self.enabled:
            try:
                logger.info(f"🤖 Loading RetrieverAgent model: {model_path}")
                self.model = LocalGGUFModel(
                    model_path,
                    n_ctx=1024,
                    n_gpu_layers=-1  # Use GPU if available
                )
                self.prompt_template = (
                    PROMPT_RETRIEVER_VI if language == "vi" else PROMPT_RETRIEVER_EN
                )
                logger.info("✅ RetrieverAgent ready")
            except Exception as e:
                logger.warning(f"⚠️  Could not load RetrieverAgent: {e}. "
                              "Falling back to default retrieval.")
                self.enabled = False
        else:
            logger.info("ℹ️  RetrieverAgent disabled")
    
    def is_enabled(self) -> bool:
        """Check if agent is active and ready"""
        return self.enabled and self.model is not None
    
    def process(
        self,
        question: str,
        chunks: List[Dict],
        max_tokens: int = 256,
        temperature: float = 0.1
    ) -> Dict:
        """
        Process chunks through the AI retriever agent.
        
        Args:
            question: User's question
            chunks: List of retrieved chunks with 'text' and 'metadata' keys
            max_tokens: Max tokens for agent output
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            Dict with:
            - 'summary': AI-generated summary of relevant info
            - 'selected_chunks': Filtered list of relevant chunks
            - 'relevant_indices': Indices of selected chunks
            - 'is_relevant': Boolean indicating if any relevant chunks found
        """
        
        if not self.is_enabled():
            # Fallback: return all chunks as-is
            logger.debug("RetrieverAgent disabled, returning all chunks")
            return {
                "summary": None,
                "selected_chunks": chunks,
                "relevant_indices": list(range(len(chunks))),
                "is_relevant": len(chunks) > 0,
                "used_agent": False
            }
        
        if not chunks:
            return {
                "summary": "KHÔNG CÓ THÔNG TIN" if self.language == "vi" else "NO INFORMATION",
                "selected_chunks": [],
                "relevant_indices": [],
                "is_relevant": False,
                "used_agent": True
            }
        
        # Build chunks text with indices
        chunks_text = "\n\n".join(
            [f"[{i+1}] {c['text'][:500]}" for i, c in enumerate(chunks)]
        )
        
        # Create prompt
        prompt = self.prompt_template.format(
            question=question,
            chunks_text=chunks_text
        )
        
        # Generate with agent
        logger.debug(f"Running RetrieverAgent on {len(chunks)} chunks")
        try:
            raw_output = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["}\n\n", "}\n\n\n"]  # Stop after JSON
            )
            
            # Extract JSON from output
            # Sometimes the model outputs extra text, so we extract the JSON part
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                logger.warning("Could not find JSON in agent output, using all chunks")
                return {
                    "summary": None,
                    "selected_chunks": chunks,
                    "relevant_indices": list(range(len(chunks))),
                    "is_relevant": True,
                    "used_agent": True
                }
            
            json_str = raw_output[json_start:json_end]
            data = json.loads(json_str)
            
            # Validate and filter chunks
            relevant_indices = data.get("relevant_chunks", [])
            summary = data.get("summary", "")
            
            # Convert to 0-indexed and filter valid indices
            valid_indices = [
                i - 1 for i in relevant_indices 
                if isinstance(i, int) and 1 <= i <= len(chunks)
            ]
            
            # Select chunks
            selected_chunks = [chunks[i] for i in valid_indices]
            
            if not selected_chunks:
                logger.info("Agent found no relevant chunks")
                summary = ("KHÔNG CÓ THÔNG TIN LIÊN QUAN" if self.language == "vi" 
                          else "NO RELEVANT INFORMATION")
            
            logger.debug(f"Agent selected {len(selected_chunks)} of {len(chunks)} chunks")
            
            return {
                "summary": summary,
                "selected_chunks": selected_chunks,
                "relevant_indices": valid_indices,
                "is_relevant": len(selected_chunks) > 0,
                "used_agent": True
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in agent output: {e}. Using all chunks.")
            return {
                "summary": None,
                "selected_chunks": chunks,
                "relevant_indices": list(range(len(chunks))),
                "is_relevant": True,
                "used_agent": False
            }
        except Exception as e:
            logger.error(f"Error in RetrieverAgent: {e}. Using all chunks.")
            return {
                "summary": None,
                "selected_chunks": chunks,
                "relevant_indices": list(range(len(chunks))),
                "is_relevant": True,
                "used_agent": False
            }


if __name__ == "__main__":
    print("✓ RetrieverAgent module ready")
    print("  - Intelligent chunk filtering")
    print("  - AI-powered summarization")
    print("  - Graceful fallback to standard retrieval")
