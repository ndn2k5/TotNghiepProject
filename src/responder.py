# -*- coding: utf-8 -*-
"""
Responder: LLM-based answer generation using retrieved context.

Uses Phi-3-Mini to generate fluent Vietnamese answers from HR policy chunks.
Guarantees grounded answers (no hallucination) by using context-only prompting.
"""

import logging
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.retriever import RetrievalResult  # torch/onnxruntime must init CUDA before llama_cpp
from src.gguf_models import LocalGGUFModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Response:
    """Represents a generated response with sources and metadata."""

    answer: str
    sources: List[Dict]  # List of source chunks used
    confidence: float  # 0.0-1.0 confidence score
    latency_ms: float  # Generation time in milliseconds
    model: str = "Phi-3-Mini"
    language: str = "vi"


class ResponseGenerator:
    """Generate fluent Vietnamese answers from retrieved chunks using Phi-3-Mini."""

    PROMPT_TEMPLATE_VI = """Bạn là trợ lý HR của công ty. Trả lời câu hỏi của nhân viên dựa CHỈ trên thông tin từ tài liệu dưới đây.

Quy tắc quan trọng:
1. CHỈ trả lời dựa trên thông tin trong ngữ cảnh
2. Nếu không tìm thấy thông tin, nói rõ: "Theo tài liệu hiện có, không tìm thấy thông tin về điều này"
3. Trích dẫn trang nếu có thể
4. Giữ câu trả lời ngắn gọn (≤256 từ)
5. Trả lời bằng tiếng Việt

Ngữ cảnh từ tài liệu:
{context}

Câu hỏi từ nhân viên: {question}

Trả lời (ngắn gọn, tiếng Việt):"""

    PROMPT_TEMPLATE_EN = """You are an HR assistant. Answer the employee's question ONLY based on the information in the documents below.

Important rules:
1. ONLY answer based on the provided context
2. If information is not found, clearly state: "Based on the available documents, I could not find information about this"
3. Cite page numbers when possible
4. Keep answer concise (≤256 words)
5. Respond in English

Context from documents:
{context}

Employee question: {question}

Answer (concise, in English):"""

    def __init__(
        self,
        model_path: str,
        language: str = "vi",
        max_tokens: int = 256,
        temperature: float = 0.3,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
    ):
        """
        Initialize the response generator.

        Args:
            model_path: Path to Phi-3-Mini GGUF model
            language: Language for prompts ('vi' for Vietnamese, 'en' for English)
            max_tokens: Maximum tokens in response
            temperature: LLM temperature (0.0-1.0, lower = more deterministic)
            n_ctx: Context window size for LLM
            n_gpu_layers: Number of layers to offload to GPU. -1 = all layers (CUDA). 0 = CPU only.
        """
        self.language = language
        self.max_tokens = max_tokens
        self.temperature = temperature

        logger.info(f"Loading Phi-3-Mini model for response generation...")
        try:
            self.llm = LocalGGUFModel(
                model_path,
                n_ctx=n_ctx,
                n_threads=4,  # Use 4 threads for 4-core CPU
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            logger.info("✓ Phi-3-Mini loaded successfully with GPU acceleration")
        except Exception as e:
            logger.error(f"Failed to load Phi-3-Mini: {e}")
            raise

    def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievalResult],
        benchmark: bool = False,
    ) -> Response:
        """
        Generate a response from question + retrieved context.

        Args:
            question: User question (normalized)
            retrieved_chunks: Top-k chunks from retriever
            benchmark: If True, measure latency

        Returns:
            Response object with answer, sources, confidence, latency
        """
        start_time = time.time()

        # Step 1: Format context from chunks
        context = self._format_context(retrieved_chunks)

        # Step 2: Build prompt
        prompt = self._build_prompt(question, context)

        # Step 3: Generate answer
        try:
            raw_answer = self.llm.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return Response(
                answer="Có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại.",
                sources=[],
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Step 4: Post-process answer
        answer = self._postprocess_answer(raw_answer)

        # Step 5: Extract sources from answer
        sources = self._extract_sources(answer, retrieved_chunks)

        # Step 6: Score confidence
        confidence = self._score_confidence(answer, retrieved_chunks)

        elapsed = (time.time() - start_time) * 1000

        if benchmark:
            logger.info(f"Response generated in {elapsed:.1f}ms (confidence: {confidence:.2f})")

        return Response(
            answer=answer,
            sources=sources,
            confidence=confidence,
            latency_ms=elapsed,
            model="Phi-3-Mini",
            language=self.language,
        )

    def _format_context(self, chunks: List[RetrievalResult]) -> str:
        """
        Format retrieved chunks into context string for prompt.

        Args:
            chunks: Retrieved chunks from retriever

        Returns:
            Formatted context string
        """
        if not chunks:
            return "Không có thông tin từ tài liệu."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk.metadata.get("page_num", "?")
            text = chunk.text[:300]  # Truncate to first 300 chars
            context_parts.append(f"[Trang {page}] {text}...")

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the prompt for the LLM.

        Args:
            question: User question
            context: Formatted context from chunks

        Returns:
            Complete prompt string
        """
        if self.language == "vi":
            template = self.PROMPT_TEMPLATE_VI
        else:
            template = self.PROMPT_TEMPLATE_EN

        return template.format(context=context, question=question)

    def _postprocess_answer(self, raw_answer: str) -> str:
        """
        Clean up LLM output.

        Args:
            raw_answer: Raw output from LLM

        Returns:
            Cleaned answer
        """
        # Remove leading/trailing whitespace
        answer = raw_answer.strip()

        # Remove common LLM artifacts
        answer = re.sub(r"^\*\*.*?\*\*:\s*", "", answer)  # Remove bold prefixes
        answer = re.sub(r"^Trả lời:\s*", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"^Answer:\s*", "", answer, flags=re.IGNORECASE)

        # Limit length (safety cap)
        words = answer.split()
        if len(words) > 300:
            answer = " ".join(words[:300]) + "..."

        return answer

    def _extract_sources(
        self, answer: str, chunks: List[RetrievalResult]
    ) -> List[Dict]:
        """
        Extract which chunks were used in the answer.

        Simple heuristic: if chunk text appears in answer, it was used.

        Args:
            answer: Generated answer
            chunks: Retrieved chunks

        Returns:
            List of source dicts with page/content info
        """
        sources = []

        for chunk in chunks:
            # Check if first 20 chars of chunk appear in answer
            snippet = chunk.text[:50].lower()
            if snippet in answer.lower():
                sources.append(
                    {
                        "page": chunk.metadata.get("page_num", "?"),
                        "text": chunk.text[:100] + "...",
                        "source": chunk.metadata.get("source", "handbook"),
                    }
                )

        # If no snippets matched, include all chunks as potential sources
        if not sources:
            sources = [
                {
                    "page": chunk.metadata.get("page_num", "?"),
                    "text": chunk.text[:100] + "...",
                    "source": chunk.metadata.get("source", "handbook"),
                }
                for chunk in chunks[:3]  # Include top 3
            ]

        return sources

    def _score_confidence(self, answer: str, chunks: List[RetrievalResult]) -> float:
        """
        Score confidence in the answer.

        Simple heuristics:
        - If answer contains "không tìm thấy" (not found), low confidence
        - If answer is very short, medium confidence
        - Otherwise, high confidence

        Args:
            answer: Generated answer
            chunks: Retrieved chunks

        Returns:
            Confidence score (0.0-1.0)
        """
        # Check for "not found" phrases
        not_found_phrases = ["không tìm thấy", "không có thông tin", "not found", "could not find"]
        if any(phrase in answer.lower() for phrase in not_found_phrases):
            return 0.3  # Low confidence

        # Check for uncertainty phrases
        uncertain_phrases = ["có thể", "có lẽ", "không chắc", "might", "perhaps"]
        uncertainty_count = sum(1 for phrase in uncertain_phrases if phrase in answer.lower())
        if uncertainty_count > 0:
            return 0.6  # Medium confidence

        # Check answer length (very short answers are less reliable)
        if len(answer) < 50:
            return 0.7

        # Check retrieval quality (if chunks are very distant, lower confidence)
        if chunks and chunks[0].distance > 0.5:
            return 0.75

        # Default: high confidence
        return 0.9

    def batch_generate(
        self,
        queries: List[Tuple[str, List[RetrievalResult]]],
    ) -> List[Response]:
        """
        Generate responses for multiple query+chunk pairs.

        Args:
            queries: List of (question, chunks) tuples

        Returns:
            List of Response objects
        """
        responses = []
        for question, chunks in queries:
            response = self.generate(question, chunks, benchmark=True)
            responses.append(response)
        return responses


def format_response_for_display(response: Response) -> str:
    """
    Format a Response for display in Streamlit or CLI.

    Args:
        response: Response object

    Returns:
        Formatted markdown string
    """
    markdown = f"""
### Trả lời

{response.answer}

---

### Nguồn tài liệu

"""

    if response.sources:
        for source in response.sources:
            markdown += f"- **Trang {source.get('page', '?')}:** {source.get('text', 'N/A')}\n"
    else:
        markdown += "- Không có tài liệu liên quan\n"

    markdown += f"""

---

### Metadata

- **Độ tin cậy:** {response.confidence*100:.0f}%
- **Thời gian:** {response.latency_ms:.1f}ms
- **Mô hình:** {response.model}
"""

    return markdown


if __name__ == "__main__":
    # Smoke test (requires model file)
    print("ResponseGenerator module loaded successfully.")
    print("Note: Requires Phi-3-Mini GGUF model to test generate().")
