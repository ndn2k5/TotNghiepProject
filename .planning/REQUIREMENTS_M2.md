# Requirements – Milestone 2: Fine-Tuning & Dataset Synthesis

## Milestone Goal

Fix all four quality failures in the v1 RAG system by synthesizing a Vietnamese HR Q&A dataset from public handbooks and fine-tuning both the embedding model and the generation LLM.

**Prerequisite:** Milestone 1 (v1 RAG pipeline) complete ✓

---

## Context

| Item | Detail |
|------|--------|
| **Problem** | v1 uses off-the-shelf models; all four quality metrics fail (hallucinations, bad Vietnamese, poor retrieval, vague answers) |
| **Dataset source** | 3 public GitHub handbooks (English Markdown) → synthesize Vietnamese QA pairs |
| **Training hardware** | H100 96GB (4+ hours allocated), A100 80GB fallback |
| **Deployment hardware** | NVIDIA T1000 4GB VRAM (friend's local machine) |
| **Inference runtime** | llama.cpp (GGUF) for LLM; sentence-transformers for embeddings |

**Handbook sources:**

- `https://github.com/hshadab/handbook`
- `https://github.com/cuesoftinc/handbook`
- `https://github.com/ultralytics/handbook`

---

## Feature Requirements

### 1. Data Synthesis Pipeline (Phase 4)

| Requirement | Detail | Acceptance Criteria |
|-------------|--------|-------------------|
| **Handbook ingestion** | Clone/fetch 3 GitHub handbooks, extract all Markdown content | All `.md` files parsed, total content >50k words |
| **Chunk & clean** | Split into ~500-char sections preserving headers | No orphaned fragments; sections retain context |
| **Vietnamese QA generation** | Use teacher LLM to generate Vietnamese question + answer pairs per chunk | ≥1500 QA pairs total (≥400 per handbook) |
| **Dataset split** | 80/10/10 train/dev/test split | Test set held out; not used in training |
| **Embedding training format** | Create (anchor, positive, hard_negative) triplets for contrastive learning | ≥1500 triplets; hard negatives from same handbook |
| **SFT training format** | Create JSONL `{instruction, input, output}` records for LLM fine-tuning | Consistent format; Vietnamese output |

**QA generation strategy:** For each handbook chunk, teacher LLM generates:
- 2–3 Vietnamese questions an employee might ask about that policy
- Ground-truth Vietnamese answer sourced only from that chunk (no hallucination)

### 2. Embedding Model Fine-Tuning (Phase 5)

| Requirement | Detail | Acceptance Criteria |
|-------------|--------|-------------------|
| **Base model** | `intfloat/multilingual-e5-small` (118M params, Vietnamese-capable) | Chosen over all-MiniLM-L6-v2 for better multilingual coverage |
| **Training method** | Supervised contrastive learning (MNR loss or TripletLoss) | Fine-tuning script runs on H100 without OOM |
| **Training data** | 1500+ (anchor, positive, hard_negative) triplets | Hard negatives: semantically close but wrong answers |
| **Export** | Save as HuggingFace model directory | Loadable via `SentenceTransformer('path/to/model')` |
| **Retrieval target** | >85% top-1 accuracy on held-out test set | Up from ~70% with off-the-shelf model |
| **Training time** | ≤30 minutes on H100 | Stop/checkpoint every 5 epochs |

### 3. LLM Fine-Tuning & GGUF Export (Phase 6)

| Requirement | Detail | Acceptance Criteria |
|-------------|--------|-------------------|
| **Base model** | `Qwen/Qwen2.5-3B-Instruct` | 3B params → fits T1000 at Q4_K_M (~2.5GB VRAM) |
| **Fine-tuning method** | QLoRA (4-bit NF4) on H100 | r=16, alpha=32, target modules: q/k/v/o_proj |
| **Training data** | 1500+ Vietnamese HR instruction-response pairs | System prompt: Vietnamese HR assistant persona |
| **Epochs** | 3–5 (with early stopping on dev loss) | Dev loss plateau = stop |
| **Export format** | GGUF Q4_K_M via llama.cpp convert scripts | File size ≤3GB |
| **VRAM at inference** | ≤3.5GB on T1000 (leave headroom) | Verified via `nvidia-smi` during inference |
| **Hallucination rate** | 0% on held-out HR policy test set (20 questions) | Manual review: every answer traceable to source chunk |
| **Vietnamese fluency** | Native-quality answers; correct diacritics, formal register | Manual review: no grammar errors in 20 sample answers |

### 4. RAG Integration & Validation (Phase 7)

| Requirement | Detail | Acceptance Criteria |
|-------------|--------|-------------------|
| **Drop-in swap** | Replace embedding model + LLM in existing RAG pipeline | `src/*.py` updated; no new external services |
| **Handbook PDF** | Convert at least 1 handbook to PDF for demo | `streamlit_app.py` loads and indexes it |
| **Re-run existing tests** | All 9 existing tests pass with new models | `pytest tests/` passes |
| **Benchmark comparison** | Compare v1 vs. v2 on 30-query test set | v2 retrieval ≥85%, v1 was ≥80% |
| **Latency check** | ≤5s E2E on T1000 with GGUF model | Measured on T1000 (4GB VRAM, GPU inference via llama.cpp CUDA) |
| **Notebooks finalized** | All 6 notebooks in `notebooks/` are runnable end-to-end | Each notebook has tested cells, no `TODO`/`pass` stubs |

---

## Technical Requirements

### Hardware & Compute

| Phase | Hardware | Estimated Time |
|-------|----------|---------------|
| Data synthesis (Phase 4) | CPU + API calls | 2–4 hours |
| Embedding fine-tune (Phase 5) | H100 96GB | ~30 min |
| QLoRA LLM fine-tune (Phase 6) | H100 96GB | ~1–2 hours |
| GGUF export | CPU | ~20 min |
| Integration & test (Phase 7) | T1000 4GB | ~1 hour |

### Model Stack (v2)

| Component | v1 | v2 |
|-----------|----|----|
| Embedding | `all-MiniLM-L6-v2` (384-dim) | `multilingual-e5-small` fine-tuned (384-dim) |
| Question normalizer | `Qwen-2.5-1.5B` GGUF | Keep as-is (already adequate) |
| Answer generator | `Phi-3-Mini` GGUF | `Qwen2.5-3B-Instruct` QLoRA fine-tuned → Q4_K_M GGUF |

### Quantization Target for T1000

| Bit depth | Model size (3B) | T1000 VRAM used | Status |
|-----------|----------------|-----------------|--------|
| Q4_K_M | ~2.0GB | ~2.5GB | ✓ Target |
| Q5_K_M | ~2.5GB | ~3.0GB | ✓ Acceptable |
| Q8_0 | ~3.5GB | ~4.0GB | ⚠ Tight |

---

## Out of Scope (Milestone 2)

- ❌ Fine-tuning the question normalizer (Qwen-2.5-1.5B) — keep as-is
- ❌ Training on proprietary company data — only public handbooks
- ❌ Multilingual support beyond Vietnamese — Vietnamese output only
- ❌ RLHF or DPO — supervised fine-tuning only
- ❌ Model serving API — local GGUF via llama.cpp only
- ❌ Automated re-training pipeline — manual fine-tune cycle only

---

## Success Criteria (Milestone 2 Ship Gate)

| Criterion | Target |
|-----------|--------|
| **Dataset generated** | ≥1500 Vietnamese QA pairs, split and formatted |
| **Embedding improved** | >85% top-1 retrieval on test set |
| **LLM quality** | 0 hallucinations on 20-question held-out set |
| **Vietnamese fluency** | Native-quality; confirmed by manual review |
| **T1000 compatible** | GGUF model loads and runs in ≤3.5GB VRAM |
| **Tests pass** | All 9 existing pytest tests pass with new models |
| **Notebooks runnable** | All 6 notebooks complete end-to-end without errors |

---

**Created:** 2026-06-01  
**Owner:** Solo developer + collaborator  
**Hardware confirmed:** H100 96GB (training), T1000 4GB (deployment)
