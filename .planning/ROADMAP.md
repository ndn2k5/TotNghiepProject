# Roadmap – HR Policy RAG Chatbot

## Milestone 1 (v1 — Complete ✓)

| Aspect | Value |
|--------|-------|
| **Duration** | 3 weeks |
| **Team** | 1 developer |
| **Status** | Complete — all 9 tests passing, thesis defense ready |

## Milestone 2 (v2 — Fine-Tuning & Dataset Synthesis)

| Aspect | Value |
|--------|-------|
| **Duration** | 2–3 weeks |
| **Team** | 2 developers (solo + collaborator) |
| **Training hardware** | H100 96GB (cloud), A100 80GB (fallback) |
| **Deploy hardware** | T1000 4GB VRAM (local) |
| **Success Gate** | ≥85% retrieval, 0 hallucinations, Vietnamese-native answers, all tests pass |
| **Requirements** | `.planning/REQUIREMENTS_M2.md` |

---

## Phase 1: Foundation & Data Pipeline

**Goal:** Build the PDF→Chunks→Embedding pipeline. Employees can upload a handbook; system auto-chunks and embeds it. No UI, no inference yet—just data plumbing.

**Duration:** ~5 hours (Week 1)  
**Owner:** Solo dev  
**Execution:** Sequential (setup → test → verification)

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

**Duration:** ~4 hours (Week 2)  
**Owner:** Solo dev  
**Execution:** Sequential (starts after Phase 1 complete)

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

**Duration:** ~6 hours (Week 3)  
**Owner:** Solo dev  
**Execution:** Sequential (starts after Phase 2 complete)

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

## Dependency Graph (Milestone 1)

```text
Phase 1 (Foundation) — Week 1
    ↓
Phase 2 (Retrieval) — Week 2
    ↓
Phase 3 (UI & Integration) — Week 3  ← COMPLETE ✓
```

---

## Milestone 2 Phases

---

## Phase 4: Data Synthesis Pipeline

**Goal:** Build a Vietnamese HR Q&A dataset from 3 public English handbooks using synthetic data generation. No manual labeling — teacher LLM writes the questions and answers.

**Duration:** ~1 week  
**Owner:** Both developers  
**Hardware:** CPU + LLM API calls  
**Notebook:** `notebooks/FINETUNING_A0-DataPreparation.ipynb`

**Phase 4 Deliverables:**

1. **Handbook Ingestion Script**
   - Clone/fetch 3 GitHub handbooks: `hshadab/handbook`, `cuesoftinc/handbook`, `ultralytics/handbook`
   - Parse all `.md` files, extract and clean text
   - Split into ~500-char chunks with section headers preserved
   - Output: `data/raw_chunks.jsonl`

2. **Vietnamese QA Generator**
   - For each chunk: teacher LLM generates 2–3 Vietnamese Q&A pairs
   - Questions: what an employee might ask about that policy
   - Answers: grounded strictly in the chunk (no hallucination)
   - Output: `data/qa_pairs.jsonl` (≥1500 pairs)

3. **Training Format Conversion**
   - Embedding training: `(anchor_question, positive_chunk, hard_negative_chunk)` triplets
   - LLM training: `{system, instruction, input, output}` JSONL records
   - Train/dev/test split: 80/10/10
   - Output: `data/embedding_train.jsonl`, `data/llm_train.jsonl`, `data/test.jsonl`

4. **Handbook → PDF Conversion**
   - Convert at least 1 handbook to PDF for demo use in existing RAG pipeline
   - Output: `data/sample_handbook.pdf`

**Phase 4 Exit Gate:**

- ≥1500 Vietnamese QA pairs generated and spot-checked (10 manual review)
- Train/dev/test splits created and valid JSON
- Both training format files ready for Phase 5 and 6
- `notebooks/FINETUNING_A0-DataPreparation.ipynb` runs end-to-end

---

## Phase 5: Embedding Model Fine-Tuning

**Goal:** Fine-tune `multilingual-e5-small` on HR domain to dramatically improve retrieval of relevant handbook chunks.

**Duration:** ~3–5 days  
**Owner:** Both developers  
**Hardware:** H100 96GB (training), CPU (export)  
**Notebook:** `notebooks/FINETUNING_01-Embedding_Finetuning.ipynb`

**Phase 5 Deliverables:**

1. **Fine-Tuning Script**
   - Load `intfloat/multilingual-e5-small`
   - Train with MultipleNegativesRankingLoss on (anchor, positive, hard_negative) triplets
   - Learning rate: 2e-5, batch size: 32, epochs: 5
   - Checkpoint every epoch; early stop on dev loss plateau
   - Output: `models/fine-tuned-embedding/`

2. **Evaluation**
   - Benchmark: top-1 and top-3 retrieval accuracy on held-out test set (20 questions)
   - Compare: off-the-shelf `multilingual-e5-small` vs. fine-tuned version
   - Report: `models/fine-tuned-embedding/eval_results.json`

3. **Integration Drop-in**
   - Fine-tuned model loads via `SentenceTransformer('models/fine-tuned-embedding')`
   - No other code changes required

**Phase 5 Exit Gate:**

- Fine-tuned model achieves >85% top-1 retrieval accuracy on test set
- Training notebook runs end-to-end on H100 in ≤30 minutes
- `models/fine-tuned-embedding/` saved and loadable
- Improvement over v1 embedding documented in eval report

---

## Phase 6: LLM Fine-Tuning & GGUF Export

**Goal:** QLoRA fine-tune `Qwen2.5-3B-Instruct` on Vietnamese HR Q&A, then export Q4_K_M GGUF for deployment on T1000 4GB VRAM.

**Duration:** ~1 week  
**Owner:** Both developers  
**Hardware:** H100 96GB (QLoRA), CPU (GGUF export)  
**Notebooks:** `notebooks/FINETUNING_02-QLoRA_Recommended.ipynb`, `notebooks/FINETUNING_A1-Export_Integration.ipynb`

**Phase 6 Deliverables:**

1. **QLoRA Fine-Tuning Script**
   - Base: `Qwen/Qwen2.5-3B-Instruct`
   - Config: 4-bit NF4, r=16, alpha=32, dropout=0.05
   - Target modules: `q_proj, k_proj, v_proj, o_proj`
   - Data: LLM training JSONL from Phase 4 (≥1500 examples)
   - System prompt: Vietnamese HR policy assistant persona
   - Epochs: 3 (with dev loss monitoring)
   - Output: LoRA adapter weights in `models/qlora-adapter/`

2. **Merge & Export**
   - Merge LoRA adapter into base model
   - Export to GGUF Q4_K_M via `llama.cpp/convert_hf_to_gguf.py`
   - Output: `models/qwen2.5-3b-hr-vietnamese-q4.gguf` (≤3GB)

3. **T1000 Validation**
   - Load GGUF on T1000 via llama.cpp; measure VRAM usage
   - Run 20 held-out HR questions; manually review for hallucinations
   - Measure inference latency (target: ≤5s E2E with RAG)

**Phase 6 Exit Gate:**

- GGUF file ≤3GB and loads on T1000 in ≤3.5GB VRAM
- 0 hallucinations on 20 held-out test questions
- Vietnamese answer quality: native-level diacritics and formal register (manual review)
- Training notebook runs on H100 in ≤2 hours
- Export notebook runs end-to-end

---

## Phase 7: Integration, Validation & Deployment

**Goal:** Swap v2 models into existing RAG pipeline. All existing tests pass. Publish updated notebooks and guide for collaborator.

**Duration:** ~3–5 days  
**Owner:** Both developers  
**Hardware:** T1000 4GB (validation)  
**Notebook:** `notebooks/FINETUNING_QUICKSTART.ipynb`

**Phase 7 Deliverables:**

1. **RAG Pipeline Update**
   - `src/embeddings.py`: swap to fine-tuned embedding model
   - `src/generation.py` (or equivalent): swap to new GGUF path
   - No external service changes; offline-capable

2. **Benchmark Comparison Report**
   - 30-query test: v1 vs. v2 retrieval accuracy
   - 20-question hallucination test: v1 vs. v2
   - Latency: v1 vs. v2 on T1000
   - Output: `.planning/docs/BENCHMARK_REPORT.md`

3. **Notebook Finalization**
   - All 6 notebooks in `notebooks/` run end-to-end without errors
   - `FINETUNING_GUIDE_00-Overview.ipynb` updated with v2 decisions
   - `FINETUNING_QUICKSTART.ipynb` usable by collaborator with zero setup

4. **Regression: Existing Tests**
   - `pytest tests/` passes all 9 tests with v2 models

**Phase 7 Exit Gate:**

- All 9 existing pytest tests pass
- v2 retrieval ≥85% on 30-query benchmark (vs. ≥80% v1)
- 0 hallucinations in held-out test (vs. "zero" claimed in v1)
- E2E latency ≤5s on T1000
- Collaborator can run quickstart notebook with no explanation needed

---

## Milestone 2 Dependency Graph

```text
Phase 4 (Data Synthesis)
    ↓ provides datasets for both
    ├── Phase 5 (Embedding Fine-Tune) ─┐
    └── Phase 6 (LLM QLoRA Fine-Tune) ─┤
                                        ↓
                               Phase 7 (Integration & Validation)
```

Phase 5 and Phase 6 can run **in parallel** on H100 once Phase 4 datasets are ready.

---

## Milestone 2 Key Milestones

| Milestone | Target | Owner | Gate |
|-----------|--------|-------|------|
| **Phase 4 complete** | Week 1 | Both | ≥1500 QA pairs, training files ready |
| **Phase 5 complete** | Week 2 | Both | >85% retrieval; H100 training done |
| **Phase 6 complete** | Week 2 | Both | GGUF ≤3GB; 0 hallucinations; T1000 verified |
| **Phase 7 complete** | Week 3 | Both | All 9 tests pass; benchmark report done |

---

## Milestone 2 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Synthetic QA quality** | Garbage data → worse models | Manual review of 50 random QA pairs before training |
| **H100 compute limit** | Can't finish fine-tuning in 4 hours | Phase 5+6 in parallel; checkpoint frequently |
| **T1000 VRAM overflow** | GGUF doesn't fit | Target Q4_K_M (2.0GB); fallback to CPU inference via llama.cpp |
| **English→Vietnamese quality** | Synthetic Vietnamese is unnatural | Use Qwen/GPT-4o as teacher; review samples |
| **Test regressions** | New models break existing tests | Run tests after each model swap |
| **Notebook rot** | Notebooks incomplete, confuse collaborator | Each notebook tested cell-by-cell before milestone closes |

---

## Success Criteria (Milestone 2)

✅ **Dataset:** ≥1500 Vietnamese HR QA pairs generated and formatted  
✅ **Embedding:** >85% top-1 retrieval on test set  
✅ **LLM:** 0 hallucinations; native Vietnamese fluency  
✅ **GGUF:** Fits T1000 (≤3.5GB VRAM)  
✅ **Tests:** All 9 existing tests pass  
✅ **Notebooks:** All 6 runnable end-to-end  
✅ **Benchmark:** v2 measurably better than v1 on all metrics  

---

**Milestone 1 last updated:** 2026-05-04  
**Milestone 2 created:** 2026-06-01  
**Milestone 3 created:** 2026-06-10 — Vietnamese KB + Hybrid Retrieval (complete ✓)  
**Milestone 4 created:** 2026-06-17 — Thesis Report Writing  
**Milestone 5 created:** 2026-06-21 — Bulk Dataset Generation (Vast.ai H200)

---

## Milestone 3 (complete ✓) — Vietnamese Knowledge Base & Hybrid Retrieval

| Aspect | Value |
|--------|-------|
| **Duration** | ~1 week |
| **Status** | Complete ✓ |
| **Deliverables** | 20 Vietnamese HR docs, multilingual embeddings, hybrid BM25+vector retrieval |

### Phase 7: Vietnamese Knowledge Base & Retrieval Improvements (Complete ✓)

- Swapped embedding model to `paraphrase-multilingual-MiniLM-L12-v2`
- Generated 8 hardcoded + 12 Minimax-M2.7 Vietnamese HR policy documents (20 total)
- Implemented hybrid BM25 + vector retrieval with Reciprocal Rank Fusion (`src/hybrid_retriever.py`)
- Fixed Phi-3-Mini prompt format (chat tokens `<|user|>...<|end|>\n<|assistant|>`)
- Fixed ChromaDB empty-chunk crash in `cli_demo.py`
- Added `RetrieverAgent` integration from collaborator (AI filtering step)
- Merged branches cleanly

**Exit gate met:** ChromaDB has 200+ Vietnamese HR chunks; hybrid retrieval finds correct docs for Vietnamese queries

---

## Milestone 4 — Thesis Report Writing

| Aspect | Value |
|--------|-------|
| **Duration** | 7 days (2026-06-17 → 2026-06-24) |
| **Team** | Solo (student) |
| **Template** | `docs/2026_required_format_thesis.md` (USTH ICT Bachelor Thesis) |
| **Language** | English |
| **Max length** | ~27 pages body |
| **Requirements** | `.planning/REQUIREMENTS_M4.md` |
| **Outline** | `.planning/docs/REPORT_OUTLINE.md` |

---

## Phase 8: Thesis Report Writing

**Goal:** Produce a complete, submission-ready bachelor thesis following USTH ICT template. All sections written, figures included, references formatted.

**Deadline:** 2026-06-24  
**Owner:** Student (solo)  
**Hardware:** Word processor / LaTeX (no compute required)

### Day-by-Day Plan

| Day | Date | Target | Output |
|-----|------|--------|--------|
| **Day 1** | Jun 17 | Abstract + Introduction draft | 3–4 pages |
| **Day 2** | Jun 18 | Objectives + Methods Part 1 (architecture, KB, chunking, embedding) | 4–5 pages |
| **Day 3** | Jun 19 | Methods Part 2 (hybrid retrieval, LLM, QLoRA experiment) | 4–5 pages |
| **Day 4** | Jun 20 | Results Part 1 (retrieval comparison, demo transcript, latency) | 3–4 pages |
| **Day 5** | Jun 21 | Results Part 2 (QLoRA negative result, discussion, limitations) | 3–4 pages |
| **Day 6** | Jun 22 | Conclusion + References + Abbreviations + figure captions | 3–4 pages |
| **Day 7** | Jun 23 | Full review, formatting pass, supervisor check, submit | polish |

### Phase 8 Deliverables

1. **Abstract** (≤250 words, 6 keywords)
2. **Section I — Introduction** (2–3 pages)
   - Vietnamese HR problem context
   - Literature review: RAG, BM25, sentence-transformers, Vietnamese NLP
   - This work's contribution
3. **Section II — Objectives** (~0.5 page)
4. **Section III — Materials and Methods** (8–10 pages)
   - System architecture figure
   - Knowledge base: 20 Vietnamese HR docs (how generated, topics covered)
   - Chunking: RecursiveCharacterTextSplitter, 600 chars, 100 overlap
   - Embedding: `paraphrase-multilingual-MiniLM-L12-v2`, 384-dim
   - Hybrid retrieval: BM25Okapi + ChromaDB cosine, merged via RRF (α=0.5, k=60)
   - LLM: Phi-3-Mini GGUF Q4, llama-cpp-python, Phi-3 chat format
   - QLoRA fine-tuning experiment (3343 pairs, T1200, 22h)
5. **Section IV — Results and Discussion** (6–8 pages)
   - Retrieval comparison: pure vector vs hybrid (table + query examples)
   - End-to-end demo: 5 example Q&A pairs with latency
   - QLoRA negative result: eval table, root cause analysis
   - Discussion: limitations, comparison to related work
6. **Section V — Conclusion & Perspective** (1–2 pages)
7. **References** (10–15 citations, numbered, DOI where available)
8. **Appendices**
   - Appendix A: Key code snippets (hybrid_retriever.py, rag_pipeline.py)
   - Appendix B: Full evaluation results table
   - Appendix C: System screenshots (CLI + Streamlit)
9. **Front matter:** Abbreviations, Tables list, Figures list, Acknowledgements

### Required Figures

| Figure | Content | Where |
|--------|---------|-------|
| Fig 1 | System architecture diagram (end-to-end pipeline) | Methods |
| Fig 2 | Hybrid retrieval flowchart (BM25 + Vector → RRF → top-k) | Methods |
| Fig 3 | Knowledge base document topic distribution | Methods |
| Fig 4 | Retrieval comparison: pure vector vs hybrid (bar chart or table) | Results |
| Fig 5 | Example chatbot session screenshot | Results |
| Fig 6 | QLoRA training loss curve (if available) | Results |

### Phase 8 Exit Gate

- [ ] All USTH template sections complete (Abstract through Appendices)
- [ ] ≥10 numbered references with DOI/URL
- [ ] ≥4 figures with captions
- [ ] ≥2 tables with captions
- [ ] Abstract ≤250 words
- [ ] Body ≤27 pages
- [ ] Sent to supervisor for review by 2026-06-24

---

## Milestone 5 — Bulk Dataset Generation (Vast.ai H200)

| Aspect | Value |
| ------ | ----- |
| **Budget** | $1.50 (~20 min H200 141GB VRAM) |
| **Team** | Solo + friend |
| **Model** | Qwen2.5-72B-Instruct (AWQ quantized) |
| **Goal** | Generate 1000+ clean Vietnamese HR Q&A pairs from 20 KB docs |
| **Success gate** | ≥1000 Q&A pairs, ≥90% domain-relevant on spot check |

---

## Phase 9: Local Prep — Generation Scripts (NO GPU)

**Goal:** Write and test all scripts locally so H200 time is 100% generation.

**Duration:** ~1 hour (local, free)
**Owner:** Solo dev

### Phase 9 Deliverables

1. **Generation script** (`scripts/vast_generate_qa.py`)
   - Load Qwen2.5-72B-Instruct via vLLM (batched inference)
   - Read all 20 docs from `data/viet_labor_docs/`
   - For each doc: generate 50-100 Vietnamese Q&A pairs
   - Output: `data/generated_qa_h200.jsonl`
   - Format: `{"question": "...", "answer": "...", "source_doc": "...", "doc_topic": "..."}`

2. **Vast.ai setup script** (`scripts/vast_setup.sh`)
   - Install vLLM, download model, run generation
   - One-command execution to minimize idle GPU time

3. **Quality validation script** (`scripts/validate_qa.py`)
   - Spot-check generated pairs for Vietnamese HR relevance
   - Flag off-domain content (the problem that killed QLoRA)
   - Compute basic stats: count, avg length, topic distribution

### Phase 9 Exit Gate

- [ ] Generation script runs locally in dry-run mode (no GPU, mock outputs)
- [ ] Setup script ready for copy-paste into Vast.ai terminal
- [ ] Validation script ready

---

## Phase 10: H200 Execution — Bulk Generation (20 min)

**Goal:** Rent H200, run generation, download results.

**Duration:** ~20 min (GPU time) + 10 min setup
**Owner:** Solo dev
**Cost:** ≤$1.50

### Phase 10 Steps

1. Rent H200 on Vast.ai
2. Upload scripts + 20 HR docs
3. Run `vast_setup.sh` → downloads model + runs generation
4. Download `generated_qa_h200.jsonl`
5. Terminate instance immediately

### Phase 10 Exit Gate

- [ ] ≥1000 Q&A pairs generated
- [ ] JSONL file downloaded locally
- [ ] Vast.ai instance terminated (no surprise charges)
- [ ] Validation script confirms ≥90% domain relevance

---

## Milestone 5 Dependency Graph

```text
Phase 9 (Local Prep — free)
    ↓
Phase 10 (H200 Execution — $1.50)
```

---

**Next step:** `/gsd-plan-phase 9` to create detailed Phase 9 plan
