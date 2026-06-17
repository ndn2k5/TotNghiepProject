# -*- coding: utf-8 -*-
"""
Script chẩn đoán Retrieval: Kiểm tra xem hệ thống có tìm đúng tài liệu không.
Sử dụng để khoanh vùng lỗi giữa Retrieval và Generation.
"""

import sys
import os
from pathlib import Path

# Thêm thư mục gốc vào path để import src
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.rag_pipeline import RAGPipeline

def debug_retrieval(question: str, model_path: str):
    print(f"\n[?] Đang kiểm tra truy xuất cho câu hỏi: '{question}'")
    
    # Khởi tạo pipeline (giả định đã có database chroma_db)
    pipeline = RAGPipeline(
        model_path=model_path,
        persist_dir="./chroma_db",
        collection_name="handbook_chunks",
        language="vi"
    )
    
    # Chỉ thực hiện bước Retrieve
    print("-" * 50)
    print("🔍 KẾT QUẢ TRUY XUẤT (TOP 3 CHUNKS):")
    chunks = pipeline.retrieve(question, top_k=3)
    
    if not chunks:
        print("❌ Không tìm thấy đoạn văn bản nào liên quan!")
        return

    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        score = chunk.get('distance') or chunk.get('rrf_score', 'N/A')
        
        print(f"\n--- Chunk {i} (Score: {score}) ---")
        print(f"Nguồn: {metadata.get('source_file', 'Không rõ')}")
        print(f"Nội dung: {text[:300]}...") # In 300 ký tự đầu
    
    print("-" * 50)
    print("Phân tích: Nếu các Chunk trên chứa câu trả lời đúng, thì lỗi nằm ở LLM/Prompt.")
    print("Nếu các Chunk trên không liên quan, cần tối ưu lại Embedding hoặc Hybrid Search.")

if __name__ == "__main__":
    # Đường dẫn model GGUF của bạn
    MODEL_PATH = r"d:\Data_Ngoc\Test\TotNghiepProject\models\phi3-mini-viet-merged.gguf"
    
    # Thử nghiệm với một câu hỏi đang bị sai trong eval_results
    test_question = "Nhân viên mới có thời gian thử việc bao lâu?"
    
    if os.path.exists(MODEL_PATH):
        debug_retrieval(test_question, MODEL_PATH)
    else:
        print(f"Lỗi: Không tìm thấy model tại {MODEL_PATH}")
