# -*- coding: utf-8 -*-
"""
Test Retrieval Quality: Kiểm tra xem RAG system có truy xuất được tài liệu đúng không.

Chạy script này để:
1. Kiểm tra xem vector store có dữ liệu không
2. Test retrieval với câu hỏi mẫu
3. Xem tài liệu truy xuất được có hợp lý không
4. Khoanh vùng vấn đề (data vs retrieval vs generation)
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from src.rag_pipeline import RAGPipeline


def test_retrieval_single_question():
    """Test retrieval với một câu hỏi cụ thể."""
    
    # Câu hỏi test - nên là câu hỏi từ tài liệu của bạn
    test_questions = [
        "Nhân viên được nghỉ phép mỗi năm bao nhiêu ngày?",
        "Thủ tục xin nghỉ phép như thế nào?",
        "Lương tối thiểu hàng tháng là bao nhiêu?",
        "Công ty cấp bảo hiểm gì cho nhân viên?",
    ]
    
    print("=" * 70)
    print("🔍 TEST RETRIEVAL QUALITY")
    print("=" * 70)
    
    # Khởi tạo pipeline
    print("\n[*] Khởi tạo RAG Pipeline...")
    try:
        # Tìm model GGUF
        model_path = None
        for potential_path in [
            "./models/phi-3-mini.gguf",
            "./models/phi-3-mini-4k-instruct-q4_k_m.gguf",
        ]:
            if Path(potential_path).exists():
                model_path = potential_path
                break
        
        if not model_path:
            print("⚠️  CẢNH BÁO: Không tìm thấy model GGUF. Chỉ kiểm tra retrieval (không cần model).")
            model_path = "./models/phi-3-mini.gguf"  # Dummy path
        
        pipeline = RAGPipeline(
            model_path=model_path,
            persist_dir="./chroma_db",
            collection_name="handbook_chunks",
            language="vi"
        )
        print("✓ Pipeline khởi tạo thành công\n")
    except Exception as e:
        print(f"✗ Lỗi khởi tạo pipeline: {e}")
        print("  (Điều này có thể OK nếu model file chưa có - chúng ta chỉ kiểm tra retrieval)")
        print()
    
    # Test từng câu hỏi
    for idx, question in enumerate(test_questions, 1):
        print(f"\n{'-' * 70}")
        print(f"TEST {idx}: {question}")
        print(f"{'-' * 70}")
        
        try:
            # Retrieve chunks
            chunks = pipeline.retrieve(question, top_k=3)
            
            if not chunks:
                print("❌ KHÔNG TÌM THẤY chunks nào!")
                print("→ Vấn đề: Vector store trống HOẶC câu hỏi không khớp với dữ liệu")
                continue
            
            print(f"✓ Tìm được {len(chunks)} chunks\n")
            
            # Hiển thị chi tiết từng chunk
            for i, chunk in enumerate(chunks, 1):
                text = chunk.get('text', '').strip()
                metadata = chunk.get('metadata', {})
                
                # Điểm số (distance hoặc rrf_score)
                score = chunk.get('distance')
                if score is None:
                    score = chunk.get('rrf_score', 'N/A')
                
                print(f"  CHUNK {i}:")
                print(f"    Score: {score}")
                print(f"    Source: {metadata.get('source_file', metadata.get('source', 'Unknown'))}")
                print(f"    Page: {metadata.get('page_num', 'N/A')}")
                print(f"    Text: {text[:200]}..." if len(text) > 200 else f"    Text: {text}")
                print()
            
            # Phân tích
            print("  📊 PHÂN TÍCH:")
            has_answer = False
            for i, chunk in enumerate(chunks, 1):
                text = chunk.get('text', '').lower()
                # Kiểm tra keywords từ câu hỏi
                keywords = question.lower().split()
                matching_keywords = [kw for kw in keywords if len(kw) > 2 and kw in text]
                if matching_keywords:
                    has_answer = True
                    print(f"    ✓ Chunk {i}: Chứa từ khóa {matching_keywords}")
            
            if not has_answer:
                print(f"    ⚠️  Chunks truy xuất không chứa keywords từ câu hỏi")
                print(f"    → Cần tối ưu retrieval (embedding, BM25 weight, chunking)")
            else:
                print(f"    ✓ Chunks có vẻ hợp lý!")
        
        except Exception as e:
            print(f"✗ Lỗi khi truy xuất: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("📋 KẾT LUẬN:")
    print("=" * 70)
    print("""
Nếu Chunks truy xuất chứa thông tin trả lời:
  → Lỗi nằm ở LLM/Prompt generation
  → Tập trung vào tuning prompt template
  
Nếu Chunks KHÔNG chứa thông tin đúng:
  → Lỗi nằm ở Retrieval
  → Tối ưu theo các bước:
     1. Kiểm tra dữ liệu gốc (PDF hay JSON)
     2. Kiểm tra embedding model (all-MiniLM-L6-v2)
     3. Tuning chunk size/overlap
     4. Tuning BM25 weight (hybrid search)
     5. Thay đổi embedding model
    """)


if __name__ == "__main__":
    test_retrieval_single_question()
