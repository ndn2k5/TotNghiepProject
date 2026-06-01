# UAT — Fine-Tuned Model + RAG Pipeline

**Date:** 2026-06-01  
**Phase:** Fine-Tuning Integration (Milestone 2 completion)  
**Tester:** Developer  
**Model:** `models/phi-3-mini.gguf` (QLoRA fine-tuned, 2.32 GB)

---

## Pre-conditions Checked

| Item | Status | Detail |
|------|--------|--------|
| Fine-tuned GGUF | ✅ | `models/phi-3-mini.gguf` — 2.32 GB |
| llama-cpp-python | ✅ | Importable, GPU acceleration enabled |
| ChromaDB | ✅ | Installed, 1503 chunks indexed from `data/raw_chunks.jsonl` |
| sentence-transformers | ✅ | `all-MiniLM-L6-v2` — 384-dim embeddings |
| src.gguf_models | ✅ | Importable, `LocalGGUFModel` loads without error |

---

## Test Results

### Test 1 — Model loads and infers
**Input:** `How many days of leave per year?`  
**Expected:** Coherent answer (generic OK without context)  
**Result:** ✅ PASS  
> "In most countries, employees are entitled to a minimum of 20 days of paid leave per year..."

---

### Test 2 — RAG: Core values retrieval
**Input:** `What are the core values at Clef?`  
**Retrieved chunk:** "Clef is a values-driven company..."  
**Expected:** Lists actual Clef values  
**Result:** ✅ PASS  
> "1. Be better today than yesterday. 2. Treat others the way they'd like to be treated."

---

### Test 3 — RAG: Time-off request
**Input:** `How do employees request time off?`  
**Retrieved chunk:** Holiday/PTO policy chunk  
**Expected:** Specific steps from handbook  
**Result:** ⚠️ PARTIAL — generic answer ("submit through appropriate channels"). Retrieval hit PTO chunk but model didn't ground tightly to it.

---

### Test 4 — RAG: Performance reviews  
**Input:** `What is the policy on performance reviews?`  
**Retrieved chunk:** "At Cuesoft, we believe in fairness, transparency, and growth..."  
**Expected:** Specific review cadence  
**Result:** ✅ PASS  
> "Quarterly reviews at Cuesoft. Clear objectives at start, review at end, written summary documented."

---

## Summary

| Metric | Value |
|--------|-------|
| Tests run | 4 |
| PASS | 3 |
| PARTIAL | 1 |
| FAIL | 0 |
| Model load time | ~3s (GPU) |
| Inference speed | ~2-4s per response |

**Overall verdict: PASS** — pipeline functional end-to-end.

---

## Known Gaps (Non-Blocking)

1. **Time-off answer is generic** — retrieval returns PTO chunk but model doesn't quote specific steps. Likely needs more specific Q&A pairs in training data about step-by-step processes.
2. **Phase 4 data count short** — 730 QA pairs generated vs. ≥1500 target. Training still succeeded (model converged). Can generate more pairs if needed for defense demo.
3. **data/splits/ missing** — train/dev/test split files not generated. Needed if embedding fine-tuning (Phase 5) is still planned.
4. **No sample PDF** — `data/sample_handbook.pdf` missing. CLI demo and `populate_vector_store.py` need it. Can generate from markdown source.

---

## Next Actions

- [ ] (Optional) Generate sample PDF from `data/raw/hshadab/` markdown for demo
- [ ] (Optional) Run `streamlit run streamlit_app.py` for UI smoke test
- [ ] (Decide) Is Phase 5 (embedding fine-tuning) still needed? Splits + embedding_train.jsonl missing.
