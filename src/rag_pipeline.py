# -*- coding: utf-8 -*-
"""
RAG Pipeline: Retrieval-Augmented Generation using local components.

Flow:
  1. User asks a question
  2. Embed question → search ChromaDB for relevant chunks
  3. Build prompt with retrieved context
  4. Generate answer using local GGUF model
  5. Return answer + sources

All processing is local — no external API calls.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.embeddings import LocalEmbedder, VectorStoreManager
from src.chunking import chunk_pages
from src.pdf_extraction import PDFExtractor
from src.gguf_models import LocalGGUFModel
from src.retriever_agent import RetrieverAgent
from src.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Prompt templates ────────────────────────────────────────────────

PROMPT_TEMPLATE_VI = """<|user|>
Bạn là trợ lý nội bộ của công ty. Dựa vào các đoạn trích từ tài liệu nội bộ bên dưới, hãy trả lời câu hỏi một cách chính xác và ngắn gọn bằng tiếng Việt.

Quy tắc:
- Chỉ trả lời dựa trên thông tin có trong ngữ cảnh.
- Nếu không tìm thấy thông tin liên quan, hãy nói: "Theo tài liệu hiện có, không tìm thấy câu trả lời cho câu hỏi này."
- Trích dẫn nguồn nếu có thể.

Ngữ cảnh:
{context}

Câu hỏi: {question}<|end|>
<|assistant|>
"""

PROMPT_TEMPLATE_EN = """<|user|>
You are an internal company assistant. Based on the document excerpts below, answer the question accurately and concisely.

Rules:
- Only answer based on the provided context.
- If the information is not found, say: "Based on the available documents, I could not find an answer to this question."
- Cite page numbers when possible.

Context:
{context}

Question: {question}<|end|>
<|assistant|>
"""


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve context → generate answer."""

    def __init__(
        self,
        model_path: str,
        persist_dir: str = "./chroma_db",
        collection_name: str = "handbook_chunks",
        n_ctx: int = 2048,
        language: str = "vi",
        n_gpu_layers: int = -1,
        use_reranker: bool = False,
        reranker_model: Optional[str] = None,
        retriever_agent_model_path: Optional[str] = None,
    ):
        """
        Initialize the RAG pipeline with all local components.

        Args:
            model_path: Path to the GGUF model file
            persist_dir: ChromaDB persistent storage directory
            collection_name: Name of the vector collection
            n_ctx: Context window for the LLM
            language: Prompt language ('vi' for Vietnamese, 'en' for English)
            n_gpu_layers: Number of layers to offload to GPU. -1 = all layers (CUDA). 0 = CPU only.
            use_reranker: If True, use RetrieverAgent to filter chunks.
            reranker_model: Path to the model used for reranking (aliases retriever_agent_model_path).
            retriever_agent_model_path: Explicit path to RetrieverAgent model.
        """
        logger.info("Initializing RAG pipeline ...")

        # Embedding model (sentence-transformers, local with CUDA support)
        self.embedder = LocalEmbedder()

        # Vector store (ChromaDB, local)
        self.vector_store = VectorStoreManager(persist_dir=persist_dir)
        self.vector_store.create_collection(name=collection_name)

        # Hybrid retriever: BM25 keyword + vector semantic search
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            alpha=0.5,   # equal weight BM25 / vector
        )

        # LLM (GGUF via llama-cpp, local with CUDA support)
        self.llm = LocalGGUFModel(model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

        # AI Retriever Agent (Optional)
        # Use retriever_agent_model_path if provided, otherwise reranker_model if use_reranker is True
        agent_path = retriever_agent_model_path or (reranker_model if use_reranker else None)
        self.retriever_agent = RetrieverAgent(
            model_path=agent_path,
            language=language,
            enabled=use_reranker or retriever_agent_model_path is not None
        )

        # Prompt template
        self.prompt_template = (
            PROMPT_TEMPLATE_VI if language == "vi" else PROMPT_TEMPLATE_EN
        )

        logger.info("✅ RAG pipeline ready with CUDA acceleration enabled")

    # ── Document Ingestion ──────────────────────────────────────────

    def ingest_pdf(
        self,
        pdf_path: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> int:
        """
        Extract text from a PDF, chunk it, embed, and store in ChromaDB.

        Args:
            pdf_path: Path to the PDF file
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            Number of chunks ingested
        """
        logger.info(f"Ingesting PDF: {pdf_path}")

        with PDFExtractor(pdf_path) as extractor:
            pages = extractor.extract_all_text()

        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not chunks:
            logger.warning(f"No chunks extracted from {pdf_path} — PDF may be scan-only or empty")
            return 0

        # Add source_file metadata
        for chunk in chunks:
            chunk["source_file"] = Path(pdf_path).name

        self.vector_store.add_chunks(chunks, self.embedder)
        self.hybrid_retriever.invalidate()  # force BM25 rebuild on next query
        logger.info(f"Ingested {len(chunks)} chunks from {Path(pdf_path).name}")
        return len(chunks)

    def ingest_directory(self, directory: str, **kwargs) -> int:
        """
        Ingest all PDF files from a directory.

        Returns:
            Total number of chunks ingested
        """
        pdf_dir = Path(directory)
        total = 0
        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            total += self.ingest_pdf(str(pdf_file), **kwargs)
        logger.info(f"Total ingested: {total} chunks from {directory}")
        return total

    # ── Question Answering ──────────────────────────────────────────

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant chunks using hybrid BM25 + vector search.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Returns:
            List of chunk dicts with text, metadata, distance, rrf_score
        """
        return self.hybrid_retriever.retrieve(question, top_k=top_k)

    def build_prompt(self, question: str, chunks: List[Dict]) -> str:
        """
        Build the full prompt with context and question.

        Args:
            question: User's question
            chunks: Retrieved context chunks

        Returns:
            Formatted prompt string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk["metadata"].get("page_num", "?")
            source = chunk["metadata"].get("source", "tài liệu")
            context_parts.append(
                f"[Đoạn {i} — Trang {page}, Nguồn: {source}]\n{chunk['text']}"
            )

        context = "\n\n".join(context_parts)
        return self.prompt_template.format(context=context, question=question)

    def answer(
        self,
        question: str,
        top_k: int = 3,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve context → generate answer.

        Args:
            question: User's question
            top_k: Number of context chunks to retrieve
            max_tokens: Max tokens for the answer
            temperature: Sampling temperature

        Returns:
            Dict with: question, answer, sources, chunks, timing
        """
        start = time.time()

        # Step 1: Retrieve (Vector Search)
        chunks = self.retrieve(question, top_k=top_k)

        if not chunks:
            return {
                "question": question,
                "answer": "Chưa có tài liệu nào được nạp vào hệ thống. "
                          "Vui lòng chạy ingest_pdf() trước.",
                "sources": [],
                "chunks": [],
                "timing": {"total_seconds": time.time() - start},
            }

        retrieval_time = time.time() - start

        # Step 1.5: AI Filtering (Optional Agent)
        agent_start = time.time()
        agent_result = self.retriever_agent.process(question, chunks)
        
        # If agent is enabled and found relevant chunks, update context
        if agent_result.get("used_agent") and agent_result.get("is_relevant"):
            chunks = agent_result["selected_chunks"]
            logger.info(f"🤖 AI Agent filtered to {len(chunks)} relevant chunks")
        
        agent_time = time.time() - agent_start

        # Step 2: Build prompt
        prompt = self.build_prompt(question, chunks)

        # Step 3: Generate answer
        gen_start = time.time()
        result = self.llm.generate_with_metadata(
            prompt, max_tokens=max_tokens, temperature=temperature
        )
        generation_time = time.time() - gen_start

        # Step 4: Format sources
        sources = []
        for c in chunks:
            meta = c["metadata"]
            sources.append({
                "page": meta.get("page_num", "?"),
                "source": meta.get("source", "unknown"),
                "distance": c.get("distance", 0),
            })

        total_time = time.time() - start

        return {
            "question": question,
            "answer": result["text"],
            "sources": sources,
            "chunks": chunks,
            "agent_summary": agent_result.get("summary"),
            "used_agent": agent_result.get("used_agent", False),
            "timing": {
                "retrieval_seconds": round(retrieval_time, 3),
                "agent_seconds": round(agent_time, 3),
                "generation_seconds": round(generation_time, 3),
                "total_seconds": round(total_time, 3),
            },
            "token_usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
            },
        }

    # ── Utilities ───────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return pipeline statistics."""
        return {
            "model": self.llm.model_name,
            "documents_in_store": self.vector_store.count(),
            "embedding_dimension": self.embedder.dimension,
        }


if __name__ == "__main__":
    import sys

    # Quick smoke test
    if len(sys.argv) < 2:
        print("Usage: python -m src.rag_pipeline <model_path> [pdf_path] [question]")
        sys.exit(1)

    model_path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_handbook.pdf"
    question = sys.argv[3] if len(sys.argv) > 3 else "What is the vacation policy?"

    pipeline = RAGPipeline(model_path)
    pipeline.ingest_pdf(pdf_path)

    result = pipeline.answer(question)
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"Timing: {result['timing']}")
