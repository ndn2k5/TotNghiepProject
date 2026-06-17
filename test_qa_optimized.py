# -*- coding: utf-8 -*-
"""
Optimized QA Quality Test: So sánh trước/sau khi tối ưu.

Test cases:
1. Câu hỏi thứ 2 (TEST 2) - cái bị fail trước: "Thủ tục xin nghỉ phép như thế nào?"
2. Tất cả 3 câu hỏi với optimized parameters

Parameters được tối ưu:
- Prompt template: Thêm yêu cầu "chi tiết", "liệt kê các bước"
- Temperature: 0.1 → 0.3 (tạo câu dài hơn, có sáng tạo hơn)
- top_p: Thêm 0.9 (nucleus sampling - balance giữa quality & diversity)
- top_k: Thêm 40 (limit đến 40 tokens có xác suất cao nhất)
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from src.rag_pipeline import RAGPipeline


TEST_CASES = [
    {
        "question": "Nhân viên được nghỉ phép mỗi năm bao nhiêu ngày?",
        "expected_keywords": ["ngày", "phép", "năm"],
        "name": "TEST 1: Số ngày nghỉ phép"
    },
    {
        "question": "Thủ tục xin nghỉ phép như thế nào?",
        "expected_keywords": ["quy trình", "nộp", "phê duyệt"],
        "name": "TEST 2: Quy trình xin nghỉ (bị fail trước)"
    },
    {
        "question": "Công ty cấp bảo hiểm gì cho nhân viên?",
        "expected_keywords": ["bảo hiểm", "công ty", "đóng"],
        "name": "TEST 3: Chính sách bảo hiểm"
    },
]


def test_qa_optimized():
    """Test QA với optimized parameters."""
    
    print("=" * 90)
    print("[TEST] QA QUALITY AFTER OPTIMIZATION")
    print("=" * 90)
    print("\n[INFO] Optimizations Applied:")
    print("   1. Prompt: Thêm yêu cầu chi tiết + liệt kê các bước")
    print("   2. Temperature: 0.1 -> 0.3 (more creative)")
    print("   3. top_p: 0.9 (nucleus sampling)")
    print("   4. top_k: 40 (limit probability mass)")
    print()
    
    # Khởi tạo pipeline
    print("[*] Khởi tạo RAG Pipeline...")
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
    
    # Test từng case với optimized parameters
    passed = 0
    failed = 0
    test_results = []
    
    for test_case in TEST_CASES:
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        test_name = test_case["name"]
        
        print(f"\n{'-' * 90}")
        print(f"[TEST] {test_name}")
        print(f"{'-' * 90}")
        print(f"Question: {question}\n")
        
        try:
            # Call với optimized parameters
            result = pipeline.answer(
                question,
                top_k=5,              # Tang len 5 de bat duoc nhieu section hon
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                top_k_sampling=40,
            )

            answer = result.get("answer", "").strip()
            chunks = result.get("chunks", [])

            # Debug: In cac chunks duoc retrieve de kiem tra
            print(f"[DEBUG] Retrieved {len(chunks)} chunks:")
            for ci, ch in enumerate(chunks, 1):
                src = ch.get("metadata", {}).get("source_file", "?")
                preview = ch.get("text", "")[:120].replace("\n", " ")
                print(f"   [{ci}] {src}: {preview}...")
            
            if not answer:
                print("[FAIL] LLM khong tra loi (answer trong)")
                failed += 1
                test_results.append({
                    "test": test_name,
                    "status": "FAIL",
                    "reason": "Empty answer"
                })
                continue
            
            # Hien thi cau tra loi
            print(f"[ANSWER]\n{answer}\n")
            
            # Kiem tra keywords
            answer_lower = answer.lower()
            found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            
            keyword_ratio = len(found_keywords) / len(expected_keywords)
            
            print(f"[QUALITY]:")
            print(f"   Length: {len(answer)} characters")
            print(f"   Keywords: {found_keywords} / {expected_keywords}")
            print(f"   Ratio: {keyword_ratio * 100:.0f}%")
            
            if keyword_ratio >= 0.66:  # >= 2/3 keywords
                print(f"   [OK] PASS")
                passed += 1
                status = "PASS"
            else:
                print(f"   [FAIL] FAIL")
                failed += 1
                status = "FAIL"
            
            test_results.append({
                "test": test_name,
                "status": status,
                "keyword_ratio": keyword_ratio,
                "answer_length": len(answer),
            })
        
        except Exception as e:
            print(f"[FAIL] Exception: {e}")
            failed += 1
            test_results.append({
                "test": test_name,
                "status": "FAIL",
                "reason": str(e)
            })
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print("\nDetailed Results:")
    for result in test_results:
        status_icon = "[PASS]" if result["status"] == "PASS" else "[FAIL]"
        print(f"  {status_icon} {result['test']}: {result['status']}")
        if "keyword_ratio" in result:
            print(f"      Keyword ratio: {result['keyword_ratio'] * 100:.0f}%")
        if "reason" in result:
            print(f"      Reason: {result['reason']}")

    print(f"\nPassed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {passed / (passed + failed) * 100:.0f}%")

    print("\n" + "=" * 90)
    if passed == len(TEST_CASES):
        print("[PASS] PERFECT! Tat ca test passed!")
        print("\nCongratulations! RAG chatbot dang hoat dong tot.")
        print("   -> Co the chuyen sang production hoac fine-tune them")
    elif passed >= len(TEST_CASES) * 0.67:
        print("[PASS] GOOD! Hau het test passed.")
        if failed > 0:
            print(f"\nTip: Thu tang temperature hoac max_tokens neu muon chi tiet hon")
    else:
        print("[WARN] Mot so test van fail.")
        print("\nCo the thu:")
        print("   - Tang temperature them (0.3 -> 0.5)")
        print("   - Tang max_tokens (512 -> 1024)")
        print("   - Xem lai prompt template")
        print("   - Thu model khac (Llama 2, Mistral)")
    print("=" * 90)


if __name__ == "__main__":
    test_qa_optimized()
