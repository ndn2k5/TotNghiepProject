"""
Final Chatbot: Hybrid Search + Caching + Gemini
Production-ready, minimal complexity, maximum efficiency.

Usage:
    python chatbot_final_hybrid.py
"""

import os
import json
import logging
import hashlib
from typing import Dict, List
from pathlib import Path
from datetime import datetime

import google.generativeai as genai
 
import google.genai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.hybrid_search_simple import SimpleHybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridChatbot:
    """
    Production chatbot combining:
    1. Hybrid search (BM25 + semantic)
    2. Query caching
    3. Gemini LLM
    
    Simple, fast, reliable.
    """
    
    def __init__(
        self,
        pdf_path: str = "./documents/handbook.pdf",
        api_key: str = None,
        cache_file: str = ".cache/chat_cache.json",
        vector_db_path: str = "./chroma_db",
        use_hybrid: bool = True,
        top_k: int = 3
    ):
        """
        Initialize chatbot
        
        Args:
            pdf_path: Path to HR handbook PDF
            api_key: Google Gemini API key
            cache_file: Cache file for Q&A pairs
            vector_db_path: Path to vector database
            use_hybrid: Enable hybrid search
            top_k: Number of documents to retrieve
        """
        self.pdf_path = pdf_path
        self.cache_file = cache_file
        self.vector_db_path = vector_db_path
        self.use_hybrid = use_hybrid
        self.top_k = top_k
        
        # Setup API
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        
        # Load cache
        self.cache = self._load_cache()
        
        # Initialize components
        logger.info("Initializing chatbot components...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"trust_remote_code": True}
        )
        
        # Load or create vector DB
        self.vectorstore = self._setup_vectorstore()
        
        # Create retriever
        if self.use_hybrid:
            logger.info("Using HYBRID retrieval (BM25 + Semantic)")
            self.chunks = self._load_chunks()
            self.retriever = SimpleHybridRetriever(self.vectorstore, self.chunks)
        else:
            logger.info("Using SEMANTIC retrieval only")
            self.retriever = self.vectorstore
        
        logger.info("✅ Chatbot ready!")
    
    def _load_cache(self) -> Dict:
        """Load query cache from file"""
        cache_path = Path(self.cache_file)
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                logger.info(f"✓ Loaded cache with {len(json.load(f))} entries")
                f.seek(0)
                return json.load(f)
        return {}
    
    def _save_cache(self) -> None:
        """Save query cache to file"""
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def _hash_query(self, query: str) -> str:
        """Hash query for caching"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def _setup_vectorstore(self):
        """Setup or load vector store"""
        db_path = Path(self.vector_db_path)
        
        if db_path.exists() and list(db_path.glob("*.db")):
            logger.info("✓ Loading existing vector store...")
            vectorstore = Chroma(
                persist_directory=str(db_path),
                embedding_function=self.embeddings,
                collection_name="handbook"
            )
            return vectorstore
        
        logger.info("Creating new vector store from PDF...")
        
        # Load PDF
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        logger.info(f"  Loaded {len(pages)} pages")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(pages)
        logger.info(f"  Split into {len(chunks)} chunks")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory=str(db_path),
            collection_name="handbook"
        )
        vectorstore.persist()
        logger.info("  ✓ Vector store created and persisted")
        
        return vectorstore
    
    def _load_chunks(self) -> List:
        """Load chunks for hybrid search"""
        db_path = Path(self.vector_db_path)
        
        if not db_path.exists():
            logger.warning("Vector DB not found, cannot load chunks")
            return []
        
        # Load chunks from ChromaDB
        collection = self.vectorstore._collection
        results = collection.get()
        
        # Reconstruct Document objects
        from langchain_core.documents import Document
        chunks = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(results['documents'], results['metadatas'])
        ]
        
        logger.info(f"✓ Loaded {len(chunks)} chunks for hybrid search")
        return chunks
    
    def _retrieve_context(self, query: str) -> List[str]:
        """Retrieve relevant documents"""
        try:
            if self.use_hybrid:
                docs = self.retriever.search(query, top_k=self.top_k)
                return [doc['text'] for doc in docs]
            else:
                docs = self.retriever.similarity_search(query, k=self.top_k)
                return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def chat(self, question: str) -> Dict:
        """
        Answer a question using cached or live retrieval
        
        Args:
            question: User question in Vietnamese
        
        Returns:
            {
                'answer': str,
                'cached': bool,
                'source_pages': List[int],
                'processing_time': float,
                'retrieval_method': str
            }
        """
        import time
        start_time = time.time()
        
        # Check cache
        query_hash = self._hash_query(question)
        if query_hash in self.cache:
            cached_result = self.cache[query_hash]
            logger.info(f"📦 Cache hit!")
            return {
                **cached_result,
                'cached': True,
                'processing_time': time.time() - start_time
            }
        
        logger.info(f"🔍 Retrieving context for: {question[:50]}...")
        
        # Retrieve context
        context_docs = self._retrieve_context(question)
        context = "\n\n".join(context_docs)
        
        if not context:
            answer = "Tôi không tìm thấy thông tin liên quan trong tài liệu."
            retrieval_method = "none"
        else:
            # Generate with Gemini
            logger.info(f"💬 Generating answer with Gemini...")
            
            prompt = f"""Bạn là một trợ lý HR thân thiện chuyên giải đáp các câu hỏi về chính sách công ty dựa trên tài liệu được cung cấp.

Tài liệu:
{context}

Câu hỏi: {question}

Hãy trả lời:
1. Dựa CHỈ trên thông tin trong tài liệu
2. Rõ ràng, ngắn gọn (2-3 câu)
3. Nếu không biết, nói "Tôi không tìm thấy thông tin này"
4. Luôn hữu ích và chuyên nghiệp

Trả lời:"""
            
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                answer = response.text
                retrieval_method = self.use_hybrid and "hybrid" or "semantic"
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                answer = f"Lỗi khi tạo câu trả lời: {e}"
                retrieval_method = "error"
        
        # Extract source pages (from metadata)
        source_pages = []
        try:
            # If using hybrid search
            if self.use_hybrid:
                docs = self.retriever.search(question, top_k=self.top_k)
                source_pages = list(set(
                    doc.get('metadata', {}).get('page', 0)
                    for doc in docs if isinstance(doc, dict)
                ))
            else:
                docs = self.retriever.similarity_search(question, k=self.top_k)
                source_pages = list(set(
                    doc.metadata.get('page', 0) for doc in docs
                ))
        except:
            source_pages = []
        
        # Prepare result
        result = {
            'answer': answer,
            'source_pages': sorted(source_pages),
            'retrieval_method': retrieval_method,
            'processing_time': time.time() - start_time
        }
        
        # Cache result
        self.cache[query_hash] = {
            'answer': answer,
            'source_pages': source_pages,
            'retrieval_method': retrieval_method,
            'cached_at': datetime.now().isoformat()
        }
        self._save_cache()
        
        return {
            **result,
            'cached': False
        }
    
    def batch_chat(self, questions: List[str]) -> List[Dict]:
        """Answer multiple questions"""
        results = []
        for q in questions:
            result = self.chat(q)
            results.append({
                'question': q,
                **result
            })
        return results
    
    def get_stats(self) -> Dict:
        """Get chatbot statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_hits': sum(1 for k, v in self.cache.items() if v.get('cached')),
            'vector_db_path': self.vector_db_path,
            'retrieval_type': 'hybrid' if self.use_hybrid else 'semantic',
            'top_k': self.top_k
        }


def main():
    """Demo"""
    
    # Initialize chatbot
    bot = HybridChatbot(
        pdf_path="./documents/handbook.pdf",
        use_hybrid=True,
        top_k=3
    )
    
    # Test questions
    questions = [
        "Bao nhiêu ngày nghỉ phép mỗi năm?",
        "Cách xin phép như thế nào?",
        "Công ty có chính sách làm việc từ xa không?",
        "Bao nhiêu ngày nghỉ phép mỗi năm?",  # Duplicate - will use cache
    ]
    
    print("\n" + "=" * 60)
    print("HR HANDBOOK CHATBOT - HYBRID SEARCH + CACHING")
    print("=" * 60)
    
    # Chat
    for q in questions:
        print(f"\n📝 Q: {q}")
        result = bot.chat(q)
        
        cached_icon = "📦" if result['cached'] else "🔍"
        print(f"{cached_icon} Retrieval: {result['retrieval_method']} "
              f"({'cached' if result['cached'] else 'live'})")
        print(f"⏱️  Time: {result['processing_time']:.2f}s")
        print(f"📄 Sources: Pages {result['source_pages']}")
        print(f"💬 A: {result['answer'][:150]}...")
    
    # Stats
    print("\n" + "=" * 60)
    stats = bot.get_stats()
    print(f"Cache entries: {stats['cache_size']}")
    print(f"Retrieval type: {stats['retrieval_type']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
