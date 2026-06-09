# -*- coding: utf-8 -*-
"""
Dual-Model RAG Pipeline
Phi-3-Mini (Context Researcher) + Qwen (Response Generator)

Architecture:
┌─────────────────────────────────────────────────┐
│ User Question                                   │
└────────────────┬────────────────────────────────┘
                 ↓
         ┌───────────────────┐
         │ Question Normalizer│
         └─────────┬─────────┘
                   ↓
    ┌──────────────────────────────────┐
    │   Phi-3-Mini (BACKGROUND WORKER) │
    │  ├─ Semantic search in vector DB │
    │  ├─ Analyze chunks              │
    │  ├─ Extract key information     │
    │  └─ Prepare context             │
    └─────────────┬────────────────────┘
                  ↓
    ┌──────────────────────────────────┐
    │  Qwen2.5 (INTERACTIVE AGENT)     │
    │  ├─ Receive prepared context    │
    │  ├─ Generate response text       │
    │  ├─ Communicate with user        │
    │  └─ Return final answer          │
    └──────────────┬───────────────────┘
                   ↓
         ┌─────────────────────┐
         │  User-Facing Answer │
         └─────────────────────┘
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.question_normalizer import QuestionNormalizer
from src.retriever import Retriever
from src.embeddings import LocalEmbedder, VectorStoreManager
from src.gguf_models import LocalGGUFModel
from src.smart_retriever import SmartContextRetriever
from src.response_validator import ResponseValidator

logger = logging.getLogger(__name__)


@dataclass
class DualModelResponse:
    """Response from dual-model pipeline"""
    user_question: str
    final_answer: str
    source_pages: List[int]
    quality_score: float
    confidence: str
    context_summary: str  # What Phi-3 found
    processing_time: float


class DualModelPipeline:
    """
    Dual-model pipeline for intelligent HR Q&A
    
    Phi-3-Mini Role: Context Researcher (Backend)
    - Searches vector database
    - Analyzes and ranks chunks
    - Prepares context for Qwen
    - Works silently in background
    
    Qwen2.5 Role: Response Generator (Frontend)
    - Receives prepared context
    - Generates final response
    - Communicates with user
    - Provides answer quality
    """
    
    def __init__(
        self,
        phi3_model_path: str = "./models/phi-3-mini.gguf",
        qwen_model_path: str = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_phi3_for_research: bool = True,
        verbose: bool = False
    ):
        """
        Initialize dual-model pipeline
        
        Args:
            phi3_model_path: Path to Phi-3-Mini GGUF model
            qwen_model_path: Path to Qwen2.5 GGUF model
            embedding_model: Sentence transformer for embeddings
            use_phi3_for_research: If True, use Phi-3 for context research
            verbose: Print debug info
        """
        self.phi3_path = phi3_model_path
        self.qwen_path = qwen_model_path
        self.use_phi3 = use_phi3_for_research
        self.verbose = verbose
        
        logger.info("🚀 Initializing Dual-Model Pipeline...")
        
        # Initialize shared components
        self.normalizer = QuestionNormalizer(use_llm=False)
        self.embedder = LocalEmbedder()
        self.vector_store = VectorStoreManager()
        self.vector_store.create_collection()
        self.smart_retriever = SmartContextRetriever()
        self.validator = ResponseValidator()
        
        # Initialize retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            use_reranking=False,
            top_k=5  # Retrieve 5 for Phi-3 to analyze
        )
        
        # Initialize Phi-3-Mini (Context Researcher)
        if self.use_phi3:
            logger.info("📚 Loading Phi-3-Mini (Context Researcher)...")
            self.phi3_model = LocalGGUFModel(
                model_path=phi3_model_path,
                model_type="phi3",
                n_ctx=1024,
                max_tokens=200,  # More tokens for detailed research
                temperature=0.2,  # Low temp for focused analysis
                top_k=2,
                n_gpu_layers=-1,  # Full GPU
                verbose=False
            )
        else:
            self.phi3_model = None
        
        # Initialize Qwen2.5 (Response Generator)
        logger.info("💬 Loading Qwen2.5 (Response Generator)...")
        self.qwen_model = LocalGGUFModel(
            model_path=qwen_model_path,
            model_type="qwen",
            n_ctx=1024,
            max_tokens=128,  # Optimized for concise answers
            temperature=0.1,  # Very low for consistency
            top_k=2,
            n_gpu_layers=-1,  # Full GPU
            verbose=False
        )
        
        logger.info("✅ Dual-Model Pipeline initialized successfully!")
    
    def ingest_pdf(self, pdf_path: str) -> None:
        """Ingest PDF documents"""
        from src.pdf_extraction import PDFExtractor
        from src.chunking import chunk_pages
        
        extractor = PDFExtractor()
        pages = extractor.extract_all_text(pdf_path)
        
        for page in pages:
            chunks = chunk_pages([page])
            for chunk in chunks:
                self.vector_store.add_document(
                    text=chunk['text'],
                    metadata={'page_num': page['page_num'], 'source': pdf_path}
                )
        
        logger.info(f"✅ Ingested {len(pages)} pages from {pdf_path}")
    
    def _phi3_research(self, question: str, chunks: List[Dict]) -> Tuple[str, str]:
        """
        Phi-3-Mini: Background context research
        
        Analyzes retrieved chunks and prepares context summary
        
        Args:
            question: User question
            chunks: Retrieved chunks
            
        Returns:
            (prepared_context, research_summary)
        """
        if not self.use_phi3 or not self.phi3_model:
            # Fallback: just format chunks
            context = "\n\n".join([c['text'] for c in chunks])
            return context, "Direct context assembly (Phi-3 disabled)"
        
        # Smart ranking
        semantic_scores = [c.get('score', 0.5) for c in chunks]
        ranked = self.smart_retriever.rank_chunks(chunks, question, semantic_scores)
        best_chunks = self.smart_retriever.select_best_chunks(ranked, top_k=3)
        
        if not best_chunks:
            return "", "No relevant context found"
        
        # Build context
        context_parts = []
        for chunk in best_chunks:
            context_parts.append(f"[Trang {chunk.metadata.get('page_num', '?')}] {chunk.text}")
        
        prepared_context = "\n\n".join(context_parts)
        
        # Generate research summary
        research_prompt = f"""Tóm tắt ngắn gọn các thông tin chính từ tài liệu về câu hỏi: "{question}"
        
Tài liệu:
{prepared_context}

Tóm tắt (2-3 câu):"""
        
        try:
            research_summary = self.phi3_model.generate(research_prompt)
            if self.verbose:
                logger.info(f"📚 Phi-3 Research: {research_summary}")
        except Exception as e:
            logger.warning(f"⚠️ Phi-3 research failed: {e}")
            research_summary = "Prepared context from database"
        
        return prepared_context, research_summary
    
    def _qwen_respond(self, question: str, prepared_context: str) -> str:
        """
        Qwen2.5: Interactive response generation
        
        Generates user-facing answer based on prepared context
        
        Args:
            question: User question
            prepared_context: Context prepared by Phi-3
            
        Returns:
            Generated response
        """
        response_prompt = f"""Dựa trên thông tin dưới đây, hãy trả lời câu hỏi một cách rõ ràng và chính xác:

Thông tin liên quan:
{prepared_context}

Câu hỏi: {question}

Trả lời (tiếng Việt, ngắn gọn, rõ ràng):"""
        
        response = self.qwen_model.generate(response_prompt)
        return response
    
    def answer(self, question: str) -> DualModelResponse:
        """
        Generate answer using dual-model pipeline
        
        Flow:
        1. Normalize question
        2. Retrieve chunks from vector DB
        3. [BACKGROUND] Phi-3-Mini analyzes & prepares context
        4. [INTERACTIVE] Qwen2.5 generates response
        5. Validate quality
        
        Args:
            question: User question in Vietnamese
            
        Returns:
            DualModelResponse with answer + metadata
        """
        import time
        start_time = time.time()
        
        # Step 1: Normalize question
        normalized_q = self.normalizer.normalize(question)
        if self.verbose:
            logger.info(f"📝 Normalized: {normalized_q}")
        
        # Step 2: Retrieve from vector DB
        retrieval_result = self.retriever.retrieve(normalized_q)
        chunks = retrieval_result.chunks
        
        if not chunks:
            elapsed = time.time() - start_time
            return DualModelResponse(
                user_question=question,
                final_answer="Xin lỗi, không tìm thấy thông tin liên quan trong cơ sở dữ liệu.",
                source_pages=[],
                quality_score=0.0,
                confidence="low",
                context_summary="No matching documents found",
                processing_time=elapsed
            )
        
        # Step 3: Phi-3-Mini (BACKGROUND) - Context Research
        if self.verbose:
            logger.info("🔍 [Phi-3] Researching context...")
        prepared_context, research_summary = self._phi3_research(normalized_q, chunks)
        
        # Step 4: Qwen2.5 (INTERACTIVE) - Generate Response
        if self.verbose:
            logger.info("💬 [Qwen] Generating response...")
        raw_answer = self._qwen_respond(question, prepared_context)
        
        # Step 5: Validate quality
        quality = self.validator.assess_overall(question, raw_answer, prepared_context)
        
        # Extract source pages
        source_pages = list(set(
            int(c.get('metadata', {}).get('page_num', 0))
            for c in chunks if c.get('metadata', {}).get('page_num')
        ))
        
        # Determine confidence
        if quality.grounding > 0.8:
            confidence = "high"
        elif quality.grounding > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            logger.info(f"✅ Pipeline complete in {elapsed:.2f}s")
        
        return DualModelResponse(
            user_question=question,
            final_answer=raw_answer,
            source_pages=source_pages,
            quality_score=quality.overall,
            confidence=confidence,
            context_summary=research_summary,
            processing_time=elapsed
        )
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            'phi3_model': str(self.phi3_path),
            'qwen_model': str(self.qwen_path),
            'phi3_enabled': self.use_phi3,
            'embeddings_model': 'all-MiniLM-L6-v2',
            'vector_db_size': len(self.vector_store.get_all_documents())
        }


# Quick test
if __name__ == "__main__":
    pipeline = DualModelPipeline(verbose=True)
    
    # Test question
    question = "Nhân viên được hưởng bao nhiêu ngày nghỉ phép mỗi năm?"
    
    print("\n" + "="*70)
    print(f"Q: {question}")
    print("="*70)
    
    result = pipeline.answer(question)
    
    print(f"\n📝 Answer:\n{result.final_answer}")
    print(f"\n📊 Context Research Summary:\n{result.context_summary}")
    print(f"\n📈 Quality: {result.quality_score:.1%}")
    print(f"🎯 Confidence: {result.confidence}")
    print(f"📄 Sources: Page {', '.join(map(str, result.source_pages))}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")
