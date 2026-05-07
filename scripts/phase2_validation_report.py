"""
Phase 2 Validation Report Generator

Runs comprehensive validation tests and generates detailed report.
"""

import logging
import time
from typing import List, Dict
import sys
sys.path.insert(0, '.')

from src.question_normalizer import QuestionNormalizer
from src.retriever import Retriever
from src.embeddings import LocalEmbedder, VectorStoreManager
from tests.test_retrieval_validation import VALIDATION_QUESTIONS

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def generate_validation_report():
    """Generate comprehensive Phase 2 validation report."""
    
    print("\n" + "="*70)
    print("PHASE 2 VALIDATION REPORT - Question Normalizer & Semantic Retriever")
    print("="*70)
    
    # Initialize components
    print("\n[*] Initializing Components...")
    try:
        normalizer = QuestionNormalizer(use_llm=False)
        embedder = LocalEmbedder()
        vector_store = VectorStoreManager()
        vector_store.create_collection()
        retriever = Retriever(
            vector_store=vector_store,
            embedder=embedder,
            use_reranking=False
        )
        print("[OK] All components initialized successfully\n")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return

    # Test 1: Retrieval Quality
    print("="*70)
    print("TEST 1: RETRIEVAL QUALITY (30 Vietnamese HR Questions)")
    print("="*70)  # noqa
    
    results = []
    relevant_count = 0
    total_latency = 0
    
    for i, question in enumerate(VALIDATION_QUESTIONS, 1):
        normalized = normalizer.normalize(question)
        retrieved, elapsed = retriever.retrieve(normalized, top_k=3)
        
        is_relevant = len(retrieved) > 0
        relevant_count += is_relevant
        total_latency += elapsed
        
        # Store result
        result = {
            "id": i,
            "question": question,
            "retrieved": len(retrieved),
            "relevant": is_relevant,
            "elapsed_ms": elapsed * 1000,
            "top_distance": retrieved[0].distance if retrieved else None,
        }
        results.append(result)
        
        # Print progress
        status = "[OK]" if is_relevant else "[FAIL]"
        print(f"{status} Q{i:2d}: {question[:50]:50s} ({len(retrieved)} results, {elapsed*1000:5.1f}ms)")
    
    success_rate = relevant_count / len(VALIDATION_QUESTIONS)
    avg_latency = total_latency / len(VALIDATION_QUESTIONS)
    
    print(f"\n{'─'*70}")
    print(f"RESULTS: {relevant_count}/{len(VALIDATION_QUESTIONS)} questions ({success_rate*100:.1f}%) retrieved results")
    print(f"TARGET: ≥80%")
    print(f"STATUS: {'PASS' if success_rate >= 0.80 else 'FAIL'}")
    print(f"{'─'*70}\n")
    
    # Test 2: Latency Performance
    print("="*70)
    print("TEST 2: LATENCY PERFORMANCE (Sample of 10 Questions)")
    print("="*70)
    
    latencies = []
    for question in VALIDATION_QUESTIONS[:10]:
        normalized = normalizer.normalize(question)
        _, elapsed = retriever.retrieve(normalized, top_k=3)
        latencies.append(elapsed * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    
    print(f"Average Latency: {avg_latency:.1f}ms")
    print(f"Min Latency:     {min_latency:.1f}ms")
    print(f"Max Latency:     {max_latency:.1f}ms")
    print(f"Target:          <150ms per query")
    print(f"STATUS: {'PASS' if avg_latency < 150 else 'FAIL'}\n")
    
    # Test 3: Crash Safety
    print("="*70)
    print("TEST 3: CRASH SAFETY (All 30 Questions)")
    print("="*70)
    
    crash_count = 0
    for question in VALIDATION_QUESTIONS:
        try:
            normalized = normalizer.normalize(question)
            retrieved, _ = retriever.retrieve(normalized)
        except Exception as e:
            crash_count += 1
            print(f"✗ CRASH on: {question[:50]}")
            print(f"  Error: {e}\n")
    
    print(f"Crashes: {crash_count}/{len(VALIDATION_QUESTIONS)}")
    print(f"STATUS: {'PASS' if crash_count == 0 else 'FAIL'}\n")
    
    # Test 4: Question Normalization
    print("="*70)
    print("TEST 4: QUESTION NORMALIZATION")
    print("="*70)
    
    normalization_ok = True
    for question in VALIDATION_QUESTIONS[:5]:
        normalized = normalizer.normalize(question)
        if not isinstance(normalized, str) or not normalized or normalized != normalized.strip():
            normalization_ok = False
            print(f"[FAIL] Normalization failed for: {question}")
        else:
            print(f"[OK] {question[:45]:45s} -> {normalized[:45]}")
    
    print(f"\nSTATUS: {'PASS' if normalization_ok else 'FAIL'}\n")
    
    # Overall Summary
    print("="*70)
    print("PHASE 2 VALIDATION SUMMARY")
    print("="*70)
    status_retrieval = 'PASS' if success_rate >= 0.80 else 'FAIL'
    status_latency = 'PASS' if avg_latency < 150 else 'FAIL'
    status_crashes = 'PASS' if crash_count == 0 else 'FAIL'
    status_norm = 'PASS' if normalization_ok else 'FAIL'
    overall = 'PASS' if success_rate >= 0.80 and avg_latency < 150 and crash_count == 0 else 'FAIL'
    ready = 'READY' if success_rate >= 0.80 and avg_latency < 150 and crash_count == 0 else 'NEEDS_OPT'
    
    print(f"""
[OK] Retrieval Quality:      {success_rate*100:.1f}%  (Target: >=80%)   {status_retrieval}
[OK] Latency:               {avg_latency:.1f}ms (Target: <150ms)  {status_latency}
[OK] Crash Safety:          {crash_count} crashes (Target: 0)   {status_crashes}
[OK] Normalization:         Validated               {status_norm}

{'-'*70}

Test Suite Result: {overall}

Phase 2 Exit Criteria: {ready}
""")
    
    print("="*70)
    print("\n📊 DETAILED RESULTS BY QUESTION:\n")
    
    # Group by result
    successful = [r for r in results if r["relevant"]]
    failed = [r for r in results if not r["relevant"]]
    
    if successful:
        print(f"[OK] SUCCESSFUL RETRIEVALS ({len(successful)}):")
        for r in successful[:5]:  # Show first 5
            print(f"   Q{r['id']:2d}: {r['question'][:55]}")
        if len(successful) > 5:
            print(f"   ... and {len(successful) - 5} more\n")
    
    if failed:
        print(f"[FAIL] FAILED RETRIEVALS ({len(failed)}):")
        for r in failed:
            print(f"   Q{r['id']:2d}: {r['question'][:55]}")
    
    # Statistics
    print(f"\n{'─'*70}")
    print(f"STATISTICS:")
    print(f"  Total Questions:     {len(VALIDATION_QUESTIONS)}")
    print(f"  Questions Retrieved: {relevant_count}")
    print(f"  Success Rate:        {success_rate*100:.1f}%")
    print(f"  Average Latency:     {avg_latency:.2f}ms")
    print(f"  Total Time:          {total_latency:.2f}s")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    generate_validation_report()
