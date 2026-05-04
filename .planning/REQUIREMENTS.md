# Requirements – HR Policy RAG Chatbot v1

## v1 Scope

**Timeframe:** 3 weeks  
**Team:** 1 developer (~10–15 hours/week)  
**Success Metric:** Ship a working demo + 30-question validation test passing 80%+ relevance

---

## Feature Requirements

### 1. PDF Upload & Chunking (Phase 1)

| Requirement | Details | Acceptance Criteria |
|-------------|---------|-------------------|
| **Handbook ingestion** | Accept single PDF (30–60 pages), extract text | PDF parses without error; all text extracted |
| **Smart chunking** | Split into ~500-char Markdown chunks with context headers | 200–400 chunks for typical handbook; no orphaned fragments |
| **Metadata preservation** | Retain page numbers, section names | Source references appear in Streamlit output |
| **Storage** | Save chunks + embeddings locally (SQLite or in-memory) | Index loads in <3s at startup |

### 2. Question Normalization (Phase 2)

| Requirement | Details | Acceptance Criteria |
|-------------|---------|-------------------|
| **Vietnamese diacritics** | Handle all Vietnamese tone marks (à, á, ả, ã, ạ, etc.) | Normalize "tính năng" = "tinh nang" when appropriate for search |
| **Colloquial phrasings** | Recognize variations: "sick leave" vs. "bệnh phép" vs. "nghỉ ốm" | Extract core intent regardless of phrasing |
| **Keyword extraction** | Agent 1 (Qwen) identifies key terms for embedding search | Queries like "what's the OT policy" → keywords: ["overtime", "policy", "approval"] |
| **Fallback to English** | If Vietnamese input fails, gracefully fall back to English | Debugging log shows language detection attempt |

### 3. Semantic Retrieval (Phase 2)

| Requirement | Details | Acceptance Criteria |
|-------------|---------|-------------------|
| **Embedding model** | Use all-MiniLM-L6-v2 (384-dim, cross-lingual); fallback to multilingual-e5-small if benchmarking shows underperformance | Normalize query → embedding → cosine similarity search |
| **Top-K retrieval** | Return top 3 relevant chunks (balancing coverage vs. context window) | Retrieved chunks have >0.5 cosine similarity to query |
| **Re-ranking (optional)** | Agent 1 can re-score results before passing to Agent 2 | Avoid garbage in → garbage out |

### 4. Answer Generation (Phase 3)

| Requirement | Details | Acceptance Criteria |
|-------------|---------|-------------------|
| **Phi-3-Mini responder** | Agent 2 reads retrieved chunks + question, generates answer | Answer is fluent Vietnamese, grounded in chunks |
| **Markdown formatting** | Output bold section titles, bullet lists, proper lists | Answer is readable in Streamlit markdown renderer |
| **No hallucination** | Answer references only handbook policies; admits "not in handbook" if needed | Manual review: 0 made-up policies in 30 test Q&A pairs |
| **≤5s inference** | Full pipeline (embed + search + generate) completes in ≤5s on target hardware | Measured on 8GB RAM, 4-core CPU |

### 5. Streamlit Chat Interface (Phase 3)

| Requirement | Details | Acceptance Criteria |
|-------------|---------|-------------------|
| **Chat layout** | Question input box, answer output area, source chunk carousel | UI is intuitive, no scrolling needed for typical Q&A |
| **Source references** | Show top 3 chunks as expandable cards with page/section labels | Users can verify answers against source |
| **Error handling** | Graceful messages if PDF fails, embedding times out, etc. | No Python stack traces shown to user |
| **Startup message** | Prompt user to upload PDF or select cached index | First-time flow is clear |

### 6. Validation & Testing (Ongoing)

| Requirement | Details | Acceptance Criteria |
|-------------|---------|-------------------|
| **Relevance benchmark** | Create 30–50 representative HR questions (sick leave, OT, benefits, etc.) | 80%+ of queries retrieve ≥1 relevant chunk; manual pass |
| **Response time log** | Measure E2E latency across 50 queries | Mean ≤4s, p95 ≤5s on target hardware |
| **Fluency review** | Have a Vietnamese speaker (or you) validate answer quality | No grammar errors; tone matches handbook formality |
| **Regression check** | Re-run validation after model/embedding updates | No degradation in relevance or speed |

---

## Technical Requirements

### Models & Dependencies

| Component | Model/Tool | Version | Notes |
|-----------|-----------|---------|-------|
| Preprocessor (Agent 1) | Qwen-2.5-1.5B (GGUF) | Latest | llama-cpp-python compatible |
| Embedding | all-MiniLM-L6-v2 | v1 | Fallback: multilingual-e5-small |
| Responder (Agent 2) | Phi-3-Mini (GGUF) | Latest | ~3B params, CPU-efficient |
| PDF parsing | PyPDF2 or pdfplumber | Latest | Extract text + metadata |
| UI | Streamlit | Latest | Python native |
| Vector DB | SQLite + faiss-cpu OR in-memory dict | N/A | No external service |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Startup time** | <5s | Load models + index into memory |
| **Query response** | ≤5s E2E | Includes embedding + search + inference |
| **Memory (idle)** | ≤6GB | Both model weights + embeddings + chunks in RAM |
| **Memory (spike)** | ≤7GB | During inference + garbage collection |
| **Embedding speed** | 500ms per query | Normalize + embed a typical question |
| **Search latency** | 100ms per top-K | Cosine similarity against ~300 embeddings |
| **Inference latency** | 2–3s per answer | Phi-3 on CPU (depends on answer length, prompt size) |

### Environment & Compatibility

| Requirement | Specification |
|-------------|---------------|
| **Python version** | 3.9+ (llama-cpp-python, Streamlit support) |
| **OS** | Windows, macOS, Linux (tested target: Windows 10+) |
| **Hardware** | 8GB RAM, 4 CPU cores (no GPU required) |
| **Internet** | Offline after model download (no cloud API calls) |
| **Storage** | ~2GB for models (Qwen-2.5-1.5B, Phi-3-Mini GGUF files) + ~100MB for index |

---

## Out of Scope (v1)

- ❌ **Multi-handbook support** – Planned for v2
- ❌ **Admin UI to upload handbooks** – v1: manual file drop + re-index
- ❌ **Fine-tuning on company data** – Use off-the-shelf models
- ❌ **Multi-language UI** – Vietnamese output only
- ❌ **Typo tolerance** – Handled by normalizer (Agent 1)
- ❌ **Web API** – Streamlit local app only; no FastAPI/REST for v1
- ❌ **Analytics/logging** – Basic logs only; no telemetry
- ❌ **Accessibility (A11y)** – Streamlit default only

---

## Success Criteria (Ship Gate)

✓ **Feature Complete:** All 5 main features (upload, normalize, retrieve, generate, UI) functional  
✓ **80% Relevance:** 30–50 test queries show ≥80% retrieve relevant chunk  
✓ **≤5s Response:** E2E latency measured on target hardware, mean ≤4s  
✓ **Fluent Vietnamese:** Manual review of 10 answer samples shows no grammar errors, grounded in handbook  
✓ **Zero Hallucination:** Manual review confirms all answers traceable to source chunks  
✓ **Deployable:** Single-file entry point (e.g., `streamlit run app.py`) with no manual config needed beyond PDF path

---

**Last updated:** 2026-05-04  
**Owner:** Solo developer  
**Status:** Ready for Roadmap breakdown
