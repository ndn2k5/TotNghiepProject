# Project State – HR Policy RAG Chatbot v1

## Current Status

| Item | Value |
|------|-------|
| **Milestone** | Milestone 5 — Bulk Dataset Generation (Vast.ai H200) |
| **Phase** | Phase 9 — Local Prep (NOT STARTED) |
| **Progress** | M4 thesis revisions applied; M5 created 2026-06-21 |
| **Last update** | 2026-06-21 |
| **Owner** | Solo dev + friend |
| **Budget** | $1.50 Vast.ai (~20 min H200 141GB VRAM) |
| **Blocker** | None — scripts need writing before renting GPU |

---

## Workflow Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Granularity** | Standard | Weekly phases (Week 1, Week 2, Week 3); aligns with 3-week timeline |
| **Execution** | Sequential | One phase at a time; prevents RAM contention between multiple GGUF models on 8GB CPU |
| **Rationale** | Tuần tự, tuần lịch | Dễ theo dõi tiến độ; tránh crash do memory overcommit |
| **Git tracking** | Yes | Planning docs committed to repo for audit trail |
| **Research approach** | Manual | Developer has clear model requirements; no need for external research agents |
| **Workspace** | d:\Data_Ngoc\Test\TotNghiepProject | Local Windows development |

---

## Phase Progress

### Phase 4: Data Synthesis Pipeline

- **Status:** Complete ✓ — 730 QA pairs in data/qa_training_data.csv
- **Goal:** Vietnamese QA dataset from 3 GitHub handbooks (≥1500 pairs)
- **Exit gate:** All splits valid, notebook runs, Phase 5+6 unblocked
- **Estimated hours:** 6 developer hours (~10 clock hours, Task 3 unattended)
- **Plan:** `.planning/phases/04-data-synthesis/PLAN.md` (8 tasks)
- **Last updated:** 2026-06-01

---

### Phase 1: Foundation & Data Pipeline

- **Status:** Complete ✓
- **Goal:** PDF→Chunks→Embeddings pipeline
- **Exit gate:** PDF ingestion, 300 chunks, embeddings load <3s
- **Estimated hours:** 5
- **Last updated:** 2026-06-01

### Phase 2: Core Logic – Retrieval & Normalization

- **Status:** Complete ✓ (100% pass rate on validation)
- **Goal:** Question normalizer (Qwen), semantic retrieval (top-3)
- **Exit gate:** 80% retrieval relevance on 30-query test
- **Estimated hours:** 4
- **Last updated:** 2026-06-01

### Phase 3: Responder, UI & Integration

- **Status:** Complete ✓ (all 9 tests passing, deployment ready)
- **Goal:** Phi-3-Mini responder, Streamlit UI, E2E integration
- **Exit gate:** ≤5s latency, 80%+ validation pass, zero hallucination
- **Estimated hours:** 6
- **Last updated:** 2026-06-01

---

### Phase 6: QLoRA Fine-Tuning

- **Status:** Complete ✓ — but model output CATASTROPHIC (eval shows hallucination, Medicare answers in Vietnamese HR context)
- **Root cause:** Crawled PDFs were garbage — Japanese labor law translations, US Medicare docs, election law. Only 376/3343 rows were actual HR content.
- **Decision:** Abandon fine-tuned model. Use base phi-3-mini.gguf + RAG. Document in thesis as negative result.
- **Last updated:** 2026-06-10

---

### Phase 7: Vietnamese Knowledge Base

- **Status:** Complete ✓
- **Goal:** Populate ChromaDB with quality Vietnamese HR content so RAG gives grounded answers
- **What was done:**
  - 8 hardcoded Vietnamese HR policy documents written directly (`scripts/download_viet_labor_docs.py`)
  - 12 additional documents generated via Minimax-M2.7 API (`scripts/generate_docs_minimax.py`)
  - 20 documents total, covering full Vietnamese HR policy range
  - Multilingual embeddings: `paraphrase-multilingual-MiniLM-L12-v2`
  - Hybrid BM25+vector retrieval with RRF implemented (`src/hybrid_retriever.py`)
  - Phi-3-Mini chat format tokens fixed (`<|user|>...<|end|>\n<|assistant|>`)
  - Merged collaborator's `RetrieverAgent` commit
- **Exit gate met:** ChromaDB has 200+ Vietnamese HR chunks; hybrid retrieval finds correct doc for "1 tháng được nghỉ bao ngày"
- **Last updated:** 2026-06-17

---

### Phase 8: Thesis Report Writing (CURRENT)

- **Status:** In progress — Day 1 of 7
- **Goal:** Complete submission-ready bachelor thesis following USTH ICT template
- **Deadline:** 2026-06-24
- **Outline:** `.planning/docs/REPORT_OUTLINE.md`
- **Requirements:** `.planning/REQUIREMENTS_M4.md`
- **Day plan:**
  - Day 1 (Jun 17): Abstract + Introduction
  - Day 2 (Jun 18): Objectives + Methods Part 1 (architecture, KB, chunking, embedding)
  - Day 3 (Jun 19): Methods Part 2 (hybrid retrieval, LLM, QLoRA negative result)
  - Day 4 (Jun 20): Results Part 1 (retrieval comparison, demo)
  - Day 5 (Jun 21): Results Part 2 (QLoRA analysis, discussion)
  - Day 6 (Jun 22): Conclusion + References + Abbreviations + figures
  - Day 7 (Jun 23): Full review, formatting, submit
- **Exit gate:** All sections complete, ≥10 references, ≥4 figures, body ≤27 pages, submitted by 2026-06-24

---

## Key Decisions (Decision Log)

| # | Decision | Rationale | Status | Date |
|---|----------|-----------|--------|------|
| 1 | **Use Qwen-2.5-1.5B for normalization** | Small, fast, handles Vietnamese well; off-the-shelf GGUF available | ✓ Committed | 2026-05-04 |
| 2 | **Use Phi-3-Mini for generation** | ~3B params, CPU-efficient, good Vietnamese fluency reports | ✓ Committed | 2026-05-04 |
| 3 | **all-MiniLM-L6-v2 as primary embedding** | 384-dim is fast; multilingual. Will benchmark vs. multilingual-e5-small in Phase 2 | — Pending | 2026-05-04 |
| 4 | **Local SQLite/in-memory vector store** | Avoid external service; handbook is static for v1 | ✓ Committed | 2026-05-04 |
| 5 | **Streamlit for UI** | Quick to build, Python-native, no frontend DevOps | ✓ Committed | 2026-05-04 |
| 6 | **Target 80% relevance, not 100%** | 100% is unrealistic; 80% is pragmatic for v1, acceptable for user validation | ✓ Committed | 2026-05-04 |

---

## Model Checklist

### Required Models (Download & Verify)

- [ ] **Qwen-2.5-1.5B GGUF** — URL: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF (or equivalent)
- [ ] **Phi-3-Mini GGUF** — URL: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf (or equivalent)
- [ ] **all-MiniLM-L6-v2** — HuggingFace model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

**Fallback models:**
- [ ] **multilingual-e5-small** — If all-MiniLM-L6-v2 underperforms on Vietnamese
- [ ] **sentence-transformers/multilingual-MiniLM-L12-v2** — Larger multilingual option

### Model Size Check

- Qwen-2.5-1.5B: ~1GB GGUF
- Phi-3-Mini: ~2–2.5GB GGUF
- Embeddings (all-MiniLM-L6-v2): ~90MB
- **Total:** ~3–3.5GB (well under 6GB budget)

---

## Assumptions

1. **Vietnamese diacritics in input** — System assumes UTF-8 encoding; normalizer handles all combining marks
2. **Handbook is static for v1** — No real-time updates; re-index manually for new handbook
3. **LLM models are available offline** — All GGUF files downloaded before first run
4. **8GB RAM target** — Developer will test on this hardware; acceptable if slightly slower on lower-end machines
5. **No internet after model download** — Models download once; zero cloud dependencies at runtime

---

## Known Unknowns / Research Items (Phase 1–2)

| Item | Impact | Resolution |
|------|--------|-----------|
| **Exact GGUF file URLs** | High | Will verify during Phase 1 setup; Hugging Face may have moved files |
| **Vietnamese embedding performance** | High | Benchmark all-MiniLM-L6-v2 vs. multilingual-e5-small in Phase 2 |
| **Phi-3-Mini inference speed on CPU** | High | Profile in Phase 3; may need quantization if >5s |
| **PDF parsing edge cases** | Medium | Will test with varied handbook formats in Phase 1 |
| **Chunk size sensitivity** | Low | ~500 chars is a starting guess; tuning in Phase 2 if relevance is poor |

---

## Communication & Handoff

**For handoff or pause:**
- State file updated after each phase
- All .planning/ docs committed to git
- Key decisions logged in STATE.md
- Assumptions documented for next developer

---

**Project initialized:** 2026-05-04 by Copilot GSD workflow  
**Next action:** Run `/gsd-plan-phase 1` to create detailed Phase 1 plan

---

## Quick Tasks Completed

| Date       | Task                                | Status      |
|------------|-------------------------------------|-------------|
| 2026-06-10 | multilingual-embedding-swap-viet-kb | complete ✓  |
| 2026-06-17 | gsd-new-milestone-report-plan       | complete ✓  |
