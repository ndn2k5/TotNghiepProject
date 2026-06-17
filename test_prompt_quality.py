# -*- coding: utf-8 -*-
"""
Test Prompt Quality: Kiểm tra xem prompt template có đúng không (mà không cần generation đầy đủ).

Nhanh hơn vì:
- Chỉ kiểm tra prompt building
- Không cần chạy LLM inference
- Có thể kiểm tra ngay
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from src.rag_pipeline import RAGPipeline


def test_prompt_quality():
    """Test mà không cần generate từ LLM - chỉ kiểm tra prompt structure."""
    
    print("=" * 80)
    print("📝 TEST PROMPT QUALITY (Retrieve → Build Prompt, no LLM generation)")
    print("=" * 80)
    
    test_questions = [
        "Nhân viên được nghỉ phép mỗi năm bao nhiêu ngày?",
        "Thủ tục xin nghỉ phép như thế nào?",
        "Công ty cấp bảo hiểm gì cho nhân viên?",
    ]
    
    # Khởi tạo pipeline
    print("\n[*] Khởi tạo RAG Pipeline (chỉ embedding, không LLM)...")
    try:
        model_path = "./models/phi-3-mini.gguf"  # Dummy, không load
        
        pipeline = RAGPipeline(
            model_path=model_path,
            persist_dir="./chroma_db",
            collection_name="handbook_chunks",
            language="vi"
        )
        print("✓ Pipeline khởi tạo thành công\n")
    except Exception as e:
        # Bỏ qua lỗi model loading
        if "No such file" in str(e) or "model" in str(e).lower():
            print(f"⚠️  Model file không tồn tại (OK, chỉ test retrieval): {e}")
            print("   → Tiếp tục test retrieval + prompt building\n")
        else:
            raise
    
    # Test từng câu hỏi
    for idx, question in enumerate(test_questions, 1):
        print(f"\n{'-' * 80}")
        print(f"TEST {idx}: {question}")
        print(f"{'-' * 80}")
        
        try:
            # Retrieve chunks
            chunks = pipeline.retrieve(question, top_k=3)
            
            if not chunks:
                print("❌ FAIL: Không tìm được chunks")
                continue
            
            print(f"✓ Retrieved {len(chunks)} chunks\n")
            
            # Build prompt
            prompt = pipeline.build_prompt(question, chunks)
            
            # Kiểm tra prompt structure
            print("📋 PROMPT STRUCTURE:")
            print(f"   Length: {len(prompt)} characters")
            print(f"   Has question: {'Câu hỏi:' in prompt or 'Question:' in prompt}")
            print(f"   Has context: {len(chunks) > 0 and any('Đoạn' in p or 'Excerpt' in p for p in prompt.split('\\n'))}")
            print(f"   Has instructions: {'Quy tắc:' in prompt or 'Rules:' in prompt or 'tài liệu' in prompt.lower()}")
            
            # Hiển thị prompt preview
            print(f"\n🔍 PROMPT PREVIEW (first 300 chars):")
            print(prompt[:300])
            print("\n...\n")
            print(prompt[-200:])
            
            # Phân tích
            if len(prompt) > 200 and chunks:
                print("\n✅ PASS: Prompt structure hợp lý")
            else:
                print("\n❌ FAIL: Prompt quá ngắn hoặc thiếu context")
        
        except Exception as e:
            print(f"❌ FAIL: Exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("📊 CONCLUSION")
    print("=" * 80)
    print("""
Nếu prompt structure OK:
  → Có thể generate từng từ từ LLM có thể sẽ chậm
  → Cần tối ưu LLM parameters:
     - Temperature (hiện tại 0.1 - rất cẩn thận)
     - max_tokens (hiện tại 256)
     - top_p / top_k

Nếu prompt có vấn đề:
  → Sửa prompt template trước khi test generation
    """)


if __name__ == "__main__":
    test_prompt_quality()
