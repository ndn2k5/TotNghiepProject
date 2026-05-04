# Roadmap – HR Policy RAG Chatbot v1

## Scope & Timeline

| Aspect | Value |
|--------|-------|
| **Total Duration** | 3 weeks (~15 hours/week) |
| **Granularity** | Coarse (3 major phases) |
| **Execution Model** | Parallel (independent phases can overlap) |
| **Team** | 1 developer |
| **Success Gate** | All requirements met, 80%+ validation pass, deployable |

---

## Phase 1: Foundation & Data Pipeline

**Goal:** Build the PDF→Chunks→Embedding pipeline. Employees can upload a handbook; system auto-chunks and embeds it. No UI, no inference yet—just data plumbing.

**Duration:** ~5 hours (Week 1, Mon–Tue)  
**Owner:** Solo dev  
**Execution:** Sequential (setup → test)

### Deliverables

1. **PDF Ingestion Module**
   - Parse PDF → extract text + metadata (page #, sections)
   - Handle edge cases (OCR-heavy PDFs, missing text, encoding issues)
   - Test: upload sample HR handbook, verify all text extracted

2. **Chunking Logic**
   - Split text into ~500-char chunks with overlap
   - Preserve section headers as context
   - Aim for 200–400 chunks for typical 30–60 page handbook
   - Test: chunk a handbook, verify chunk quality and count

3. **Embedding & Storage**
   - Initialize all-MiniLM-L6-v2 model (download GGUF if needed)
   - Embed all chunks (384-dim vectors)
   - Store chunks + vectors in SQLite or in-memory dict with metadata
   - Test: verify embeddings load correctly at startup

4. **Validation**
   - Index loads in <3s
   - No text loss during chunk/embed cycle
   - Chunk count within expected range

### Exit Criteria

- ✓ PDF uploads without error
- ✓ ~300 chunks generated for sample handbook
- ✓ Embeddings stored locally; retrieval latency <100ms for top-3
- ✓ Code is modular (separate pdf.py, chunker.py, embeddings.py)

---

## Phase 2: Core Logic – Retrieval & Normalization

**Goal:** Build the retrieval pipeline + question normalizer. Given a query, retrieve top-3 relevant chunks and normalize Vietnamese questions. Still no inference yet.

**Duration:** ~4 hours (Week 1–2, Wed–Fri)  
**Owner:** Solo dev  
**Execution:** Can start in parallel with Phase 1 testing

### Deliverables

1. **Question Normalizer (Agent 1 – Qwen-2.5-1.5B)**
   - Load Qwen model (GGUF, ~1.5B params)
   - Normalize Vietnamese: handle diacritics, colloquial phrasings
   - Extract keywords for embedding search
   - Benchmark: 50ms per query on CPU
   - Test: normalize 10 sample Vietnamese HR questions

2. **Semantic Retrieval**
   - Embed incoming query using all-MiniLM-L6-v2
   - Cosine similarity search against chunk embeddings
   - Return top 3 with scores
   - Benchmark: <100ms per query
   - Test: retrieve relevant chunks for 10 test queries

3. **Re-ranking (Optional)**
   - Agent 1 (Qwen) can optionally re-rank top-3 results
   - Validate: does re-ranking improve relevance?
   - Decision: include or skip based on latency impact

4. **Validation Test Set**
   - Create 30 representative HR questions (Vietnamese)
   - Manually verify that top-3 retrievals are relevant for ≥80%
   - Log failures for debugging

### Exit Criteria

- ✓ Qwen loads without error
- ✓ Question normalization handles Vietnamese diacritics
- ✓ Top-3 retrieval latency <150ms per query
- ✓ 80%+ of 30 test queries retrieve ≥1 relevant chunk
- ✓ No Python crashes on unexpected inputs

---

## Phase 3: Responder, UI & Integration

**Goal:** Add the answer generation step (Agent 2 – Phi-3-Mini) and Streamlit UI. Integrate everything into a deployable end-to-end chatbot.

**Duration:** ~6 hours (Week 2–3)  
**Owner:** Solo dev  
**Execution:** Can start in parallel with Phase 2; depends on Phase 2 retrieval being functional

### Deliverables

1. **Responder (Agent 2 – Phi-3-Mini)**
   - Load Phi-3-Mini model (GGUF, ~3B params)
   - Given query + top-3 chunks, generate Vietnamese answer
   - Enforce Markdown formatting (bold, lists, proper structure)
   - Benchmark: 2–3s per answer on CPU
   - Test: generate answers for 10 test queries; manual fluency review

2. **Streamlit Chat Interface**
   - Question input box
   - Answer output area (Markdown-rendered)
   - Expandable cards showing top-3 source chunks (with page/section labels)
   - "Upload handbook" → re-index (manual flow, no admin UI)
   - Error handling: graceful messages for failures

3. **End-to-End Integration**
   - Tie together: PDF upload → chunking → embedding → normalization → retrieval → generation
   - Add startup checks (models downloaded, index loaded)
   - Single entry point: `streamlit run app.py`

4. **Validation & Testing**
   - Run 30–50 validation queries (mix of easy, hard, edge cases)
   - Measure: latency per query, relevance, fluency
   - Manual review: zero hallucination, all answers grounded in handbook

5. **Benchmarking Report**
   - Document: startup time, latency (p50, p95), memory usage
   - Identify bottlenecks; propose optimizations for v2

### Exit Criteria

- ✓ Phi-3-Mini loads and generates answers
- ✓ Streamlit app launches without errors
- ✓ E2E query response time ≤5s (mean ≤4s)
- ✓ 80%+ of 50 validation queries pass relevance + fluency
- ✓ Zero hallucination detected in manual review
- ✓ Source chunks are displayed correctly
- ✓ README + quick-start guide written

---

## Dependency Graph

```
Phase 1 (Foundation)
    ↓
Phase 2 (Retrieval) ← Can start in parallel with Phase 1 testing
    ↓
Phase 3 (UI & Integration) ← Depends on Phase 2 retrieval working
```

**In practice (parallel execution):**
- Weeks 1–2: Phase 1 + Phase 2 in parallel (models downloading, testing separately)
- Week 2–3: Phase 2 refinement + Phase 3 full build + validation

---

## Key Milestones

| Milestone | Target Date | Owner | Gate |
|-----------|------------|-------|------|
| **Phase 1 complete** | End of Week 1 | Dev | PDF → 300 chunks; embeddings load in <3s |
| **Phase 2 complete** | Mid-Week 2 | Dev | 80%+ retrieval relevance on 30-query test |
| **Phase 3 complete** | End of Week 3 | Dev | Full E2E working; ≤5s response time |
| **Validation & UAT** | End of Week 3 | Dev | 50-query test pass; zero hallucination |
| **Ship** | Week 3 | Dev | README, quick start, one-click deployment |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Vietnamese model underperformance** | Low relevance for Vietnam-specific phrasings | Benchmark all-MiniLM-L6-v2 vs. multilingual-e5-small in Phase 2 |
| **Inference too slow on CPU** | Violates ≤5s SLA | Profile in Phase 3; consider quantization or model swap if needed |
| **Model download failures** | Blocks start | Download models in Phase 1; verify checksums |
| **PDF parsing edge cases** | Data loss or corruption | Test with varied handbook formats (scanned, native PDF, etc.) |
| **Memory pressure** | OOM on 8GB hardware | Monitor in Phase 3; stream inference if needed |
| **Hallucination in answers** | User gets wrong info | Manual fluency + grounding review before ship |

---

## Success Criteria (Across All Phases)

✅ **Functionally complete:** All 5 features (upload, normalize, retrieve, generate, UI) working  
✅ **80% relevance:** Test queries show relevant chunks retrieved  
✅ **≤5s latency:** E2E measured on target hardware  
✅ **Fluent Vietnamese:** Manual review confirms quality  
✅ **Zero hallucination:** All answers traceable to handbook  
✅ **Deployable:** `streamlit run app.py` is the full setup  
✅ **Documented:** README, quick-start, assumptions logged

---

**Last updated:** 2026-05-04 (Roadmap created)  
**Next step:** `/gsd-plan-phase 1` to start Phase 1 detailed planning
