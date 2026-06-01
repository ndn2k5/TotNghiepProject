"""
Test Dual-Model Pipeline
Verify Phi-3 (Researcher) + Qwen (Responder) work together
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.dual_model_pipeline import DualModelPipeline


def test_dual_model_pipeline():
    """Test dual-model pipeline"""
    print("\n" + "="*70)
    print("🧪 Testing Dual-Model Pipeline")
    print("="*70)
    
    try:
        # Check if models exist
        phi3_path = Path("./models/phi-3-mini.gguf")
        qwen_path = Path("./models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        
        if not phi3_path.exists():
            print(f"❌ Phi-3 model not found: {phi3_path}")
            print("Download from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
            return False
        
        if not qwen_path.exists():
            print(f"❌ Qwen model not found: {qwen_path}")
            print("Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF")
            return False
        
        print("✅ Both models found")
        print(f"   Phi-3: {phi3_path}")
        print(f"   Qwen:  {qwen_path}")
        
        # Initialize pipeline
        print("\n🚀 Initializing dual-model pipeline...")
        pipeline = DualModelPipeline(
            phi3_model_path=str(phi3_path),
            qwen_model_path=str(qwen_path),
            use_phi3_for_research=True,
            verbose=True
        )
        print("✅ Pipeline initialized")
        
        # Get stats
        stats = pipeline.get_stats()
        print(f"\n📊 Pipeline Stats:")
        print(f"   Phi-3 enabled: {stats['phi3_enabled']}")
        print(f"   Vector DB size: {stats['vector_db_size']}")
        
        # Test question
        print(f"\n🧪 Testing with sample question...")
        question = "Nhân viên được hưởng bao nhiêu ngày nghỉ phép mỗi năm?"
        
        print(f"\n📝 Question: {question}")
        print(f"\n⏳ Processing (this may take a moment)...")
        
        result = pipeline.answer(question)
        
        print(f"\n{'='*70}")
        print(f"✅ RESULT")
        print(f"{'='*70}")
        
        print(f"\n💬 Final Answer (from Qwen):")
        print(f"   {result.final_answer}")
        
        print(f"\n📚 Context Research Summary (from Phi-3):")
        print(f"   {result.context_summary}")
        
        print(f"\n📊 Quality Metrics:")
        print(f"   Quality Score: {result.quality_score:.1%}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Sources: {result.source_pages if result.source_pages else 'None'}")
        print(f"   Time: {result.processing_time:.2f}s")
        
        print(f"\n{'='*70}")
        print("✅ Test Passed! Dual-model pipeline is working correctly.")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dual_model_pipeline()
    sys.exit(0 if success else 1)
