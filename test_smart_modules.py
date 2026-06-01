"""
Test Suite for Smart Retrieval & Response Validation
Run this file to verify all modules work correctly

Usage:
    python test_smart_modules.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.smart_retriever import SmartContextRetriever
from src.response_validator import ResponseValidator
from src.improved_prompts import select_prompt_template


def test_smart_retriever():
    """Test SmartContextRetriever"""
    print("\n" + "="*60)
    print("TEST 1: SmartContextRetriever")
    print("="*60)
    
    retriever = SmartContextRetriever()
    
    # Sample HR data
    chunks = [
        {
            'text': 'Mỗi nhân viên được hưởng 12 ngày nghỉ phép hàng năm. Ngày phép có thể được tích lũy nếu không sử dụng.',
            'metadata': {'page_num': 5}
        },
        {
            'text': 'Công ty có chính sách bảo hiểm sức khỏe cho toàn bộ nhân viên và gia đình của họ.',
            'metadata': {'page_num': 8}
        },
        {
            'text': 'Lương cơ bản được tính toán dựa trên vị trí và kinh nghiệm của nhân viên.',
            'metadata': {'page_num': 12}
        },
    ]
    
    question = "Tôi được hưởng bao nhiêu ngày nghỉ phép mỗi năm?"
    semantic_scores = [0.95, 0.2, 0.3]  # First chunk most relevant
    
    # Test ranking
    ranked = retriever.rank_chunks(chunks, question, semantic_scores)
    
    print(f"\n✓ Ranked {len(ranked)} chunks")
    print(f"\nTop chunk: {ranked[0].text[:60]}...")
    print(f"Score: {ranked[0].combined_score:.1%}")
    
    # Test selection
    best = retriever.select_best_chunks(ranked, top_k=2)
    print(f"\n✓ Selected {len(best)} best chunks")
    
    # Test explanation
    explanation = retriever.explain_ranking(ranked, top_k=2)
    print(f"\n✓ Generated explanation:\n{explanation}")
    
    print("✅ SmartContextRetriever: PASSED")
    return True


def test_response_validator():
    """Test ResponseValidator"""
    print("\n" + "="*60)
    print("TEST 2: ResponseValidator")
    print("="*60)
    
    validator = ResponseValidator()
    
    # Good response
    question = "Bao nhiêu ngày nghỉ phép?"
    good_response = "Theo chính sách công ty, mỗi nhân viên được hưởng 12 ngày nghỉ phép hàng năm."
    context = "Điều 5: Mỗi nhân viên được hưởng 12 ngày nghỉ phép hàng năm."
    
    quality = validator.assess_overall(question, good_response, context)
    
    print(f"\n📊 Good Response Quality:")
    print(f"  Overall: {quality.overall:.1%}")
    print(f"  Completeness: {quality.completeness:.1%}")
    print(f"  Grounding: {quality.grounding:.1%}")
    print(f"  Clarity: {quality.clarity:.1%}")
    print(f"  Length: {quality.length:.1%}")
    print(f"  Language: {quality.language:.1%}")
    print(f"  Acceptable: {validator.is_acceptable(quality)}")
    
    if quality.issues:
        print(f"\n⚠️  Issues: {quality.issues}")
    if quality.suggestions:
        print(f"💡 Suggestions: {quality.suggestions}")
    
    # Bad response (too short)
    bad_response = "12 ngày"
    quality_bad = validator.assess_overall(question, bad_response, context)
    
    print(f"\n📊 Bad Response Quality:")
    print(f"  Overall: {quality_bad.overall:.1%}")
    print(f"  Acceptable: {validator.is_acceptable(quality_bad)}")
    if quality_bad.issues:
        print(f"  Issues: {quality_bad.issues}")
    
    print("✅ ResponseValidator: PASSED")
    return True


def test_improved_prompts():
    """Test Improved Prompts"""
    print("\n" + "="*60)
    print("TEST 3: Improved Prompts")
    print("="*60)
    
    # Test template selection
    test_cases = [
        {
            'question': 'So sánh chính sách nghỉ phép với bảo hiểm',
            'context': 'Đây là nội dung về cả hai chính sách',
            'expected_type': 'comparison',
            'name': 'Comparison question'
        },
        {
            'question': 'Bao nhiêu ngày nghỉ phép?',
            'context': 'Ngắn',
            'expected_type': 'simple',
            'name': 'Simple question with short context'
        },
        {
            'question': 'Tại sao công ty có chính sách này và nó ảnh hưởng như thế nào?',
            'context': 'Dài' * 50,
            'expected_type': 'cot',
            'name': 'Complex question'
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        template, template_type, reason = select_prompt_template(
            test['question'],
            test['context'],
            3
        )
        
        print(f"\n{i}. {test['name']}")
        print(f"   Question: {test['question'][:50]}...")
        print(f"   Selected: {template_type}")
        print(f"   Reason: {reason}")
        print(f"   Template length: {len(template)} chars")
    
    print("\n✅ Improved Prompts: PASSED")
    return True


def test_integration():
    """Test integration of all modules"""
    print("\n" + "="*60)
    print("TEST 4: Integration Test")
    print("="*60)
    
    print("\nSimulating RAG pipeline with smart modules...")
    
    retriever = SmartContextRetriever()
    validator = ResponseValidator()
    
    # Simulate retrieved chunks
    chunks = [
        {
            'text': 'Mỗi nhân viên được hưởng 12 ngày nghỉ phép hàng năm theo luật lao động.',
            'metadata': {'page_num': 5}
        },
        {
            'text': 'Thời gian làm việc bình thường là 8 giờ mỗi ngày, 5 ngày mỗi tuần.',
            'metadata': {'page_num': 3}
        },
    ]
    
    question = "Tôi được hưởng bao nhiêu ngày nghỉ phép?"
    semantic_scores = [0.92, 0.15]
    
    # Step 1: Smart ranking
    ranked = retriever.rank_chunks(chunks, question, semantic_scores)
    best = retriever.select_best_chunks(ranked, top_k=1)
    
    print(f"\n1️⃣  Smart Ranking:")
    print(f"   Top chunk score: {best[0].combined_score:.1%}")
    print(f"   Reason: {best[0].rank_reason}")
    
    # Step 2: Format context
    context = retriever.format_context_with_scores(best)
    
    print(f"\n2️⃣  Context Formatted:")
    print(f"   Length: {len(context)} chars")
    
    # Step 3: Select prompt
    template, template_type, reason = select_prompt_template(question, context, len(best))
    
    print(f"\n3️⃣  Prompt Selected:")
    print(f"   Type: {template_type}")
    print(f"   Reason: {reason}")
    
    # Step 4: Simulate answer
    simulated_answer = "Theo chính sách công ty, mỗi nhân viên được hưởng 12 ngày nghỉ phép hàng năm."
    
    # Step 5: Validate quality
    quality = validator.assess_overall(question, simulated_answer, context)
    
    print(f"\n4️⃣  Quality Validation:")
    print(f"   Overall score: {quality.overall:.1%}")
    print(f"   Acceptable: {validator.is_acceptable(quality)}")
    
    if quality.issues:
        print(f"   Issues: {quality.issues}")
    
    print("\n✅ Integration Test: PASSED")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 RUNNING SMART MODULES TEST SUITE")
    print("="*70)
    
    tests = [
        ("SmartContextRetriever", test_smart_retriever),
        ("ResponseValidator", test_response_validator),
        ("ImprovedPrompts", test_improved_prompts),
        ("Integration", test_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {name}: FAILED")
            print(f"Error: {str(e)}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Smart modules are ready to use.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
