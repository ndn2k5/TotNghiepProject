# -*- coding: utf-8 -*-
"""
Test End-to-End QA: Kiểm tra xem chatbot trả lời đúng không.

Đây là test hoàn chỉnh: Retrieve context → LLM generates answer
So sánh câu trả lời với expected answer để xem quality
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from src.rag_pipeline import RAGPipeline


# Test cases: (Question, Expected keywords in answer)
TEST_CASES = [
    {
        "question": "Nhân viên được nghỉ phép mỗi năm bao nhiêu ngày?",
        "expected_keywords": ["ngày", "phép", "năm"],
        "expected_min_length": 50,
    },
    {
        "question": "Thủ tục xin nghỉ phép như thế nào?",
        "expected_keywords": ["quy trình", "nộp", "phê duyệt"],
        "expected_min_length": 80,
    },
    {
        "question": "Công ty cấp bảo hiểm gì cho nhân viên?",
        "expected_keywords": ["bảo hiểm", "công ty", "đóng"],
        "expected_min_length": 60,
    },
]


def test_qa_quality():
    """Test quality of QA output."""
    
    print("=" * 80)
    print("TEST END-TO-END QA QUALITY")
    print("=" * 80)
    
    # Khởi tạo pipeline
    print("\n[*] Khởi tạo RAG Pipeline...")
    try:
        model_path = None
        for potential_path in [
            "./models/phi-3-mini.gguf",
            "./models/phi-3-mini-4k-instruct-q4_k_m.gguf",
        ]:
            if Path(potential_path).exists():
                model_path = potential_path
                break
        
        if not model_path:
            print("[ERROR] Khong tim thay model GGUF!")
            print("  Hay download model tu: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
            return

        pipeline = RAGPipeline(
            model_path=model_path,
            persist_dir="./chroma_db",
            collection_name="handbook_chunks",
            language="vi"
        )
        print("[OK] Pipeline khoi tao thanh cong\n")
    except Exception as e:
        print(f"[ERROR] Loi khoi tao: {e}")
        return
    
    # Test từng case
    passed = 0
    failed = 0
    
    for idx, test_case in enumerate(TEST_CASES, 1):
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        expected_min_length = test_case["expected_min_length"]
        
        print(f"\n{'-' * 80}")
        print(f"TEST {idx}: {question}")
        print(f"{'-' * 80}")
        
        try:
            # Full RAG pipeline
            result = pipeline.answer(question, top_k=3, max_tokens=256, temperature=0.1)
            
            answer = result.get("answer", "").strip()
            chunks = result.get("chunks", [])
            
            if not answer:
                print("[FAIL] LLM khong tra loi (answer trong)")
                failed += 1
                continue

            print(f"\nCau tra loi:")
            print(f"{answer}\n")

            # Kiem tra:
            # 1. Do dai cau tra loi
            if len(answer) < expected_min_length:
                print(f"[WARN] Cau tra loi qua ngan ({len(answer)} < {expected_min_length} chars)")

            # 2. Chua expected keywords
            answer_lower = answer.lower()
            found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]

            print(f"Chat luong:")
            print(f"   Do dai: {len(answer)} characters")
            print(f"   Keywords tim duoc: {found_keywords} / {expected_keywords}")

            if len(found_keywords) >= len(expected_keywords) * 0.5:  # >= 50% keywords
                print(f"   [PASS] Cau tra loi chua thong tin hop ly")
                passed += 1
            else:
                print(f"   [FAIL] Cau tra loi thieu keywords quan trong")
                failed += 1

            # 3. Hien thi source chunks
            print(f"\nSources ({len(chunks)} chunks):")
            for i, chunk in enumerate(chunks[:2], 1):
                source = chunk.get("metadata", {}).get("source_file", "Unknown")
                text = chunk.get("text", "")[:100]
                print(f"   [{i}] {source}: {text}...")
        
        except Exception as e:
            print(f"✗ FAIL: Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {passed / (passed + failed) * 100:.0f}%")

    if passed == len(TEST_CASES):
        print("\n[PASS] Tat ca test passed! Chatbot hoat dong tot.")
        print("   -> Co the chuyen sang production hoac fine-tune prompt de tot hon")
    elif passed >= len(TEST_CASES) * 0.5:
        print("\n[WARN] Mot so test failed. Can toi uu prompt template.")
        print("   -> Thu dieu chinh:")
        print("     - Temperature (hien: 0.1)")
        print("     - Context window")
        print("     - Prompt wording")
    else:
        print("\n[FAIL] Hau het test failed. Can khac phuc van de lon hon.")
        print("   -> Kiem tra:")
        print("     - Model GGUF co hoat dong khong")
        print("     - Prompt template (co null/undefined khong)")
        print("     - Retrieved chunks (co phu hop khong)")


if __name__ == "__main__":
    test_qa_quality()
