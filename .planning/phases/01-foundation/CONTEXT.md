# Phase 1: Foundation & Data Pipeline — Context

**Gathered:** 2026-05-04  
**Status:** Ready for planning  
**Source:** Discuss-phase (developer input)

---

## Phase Boundary

Build the end-to-end PDF→Chunks→Embeddings pipeline. Employees can upload an employee handbook; the system automatically extracts text, splits into optimal chunks, and embeds them for retrieval. No inference, no UI yet—just robust data plumbing.

**Deliverables:**
- Modular PDF extraction (PyMuPDF)
- 3-size chunk experiments (300/600/900 chars)
- Embedding integration (all-MiniLM-L6-v2)
- Persistent vector store (ChromaDB/SQLite)
- Evaluation script to pick optimal chunk size

**Exit Gate:** PDF ingests without error → ~300 chunks generated → embeddings load in <3s → chunk size recommendation ready for Phase 2

---

## Implementation Decisions

### PDF Extraction
- **Choice**: PyMuPDF (fitz) for text extraction from native PDFs
- **Rationale**: Target documents have text layers (not scanned); PyMuPDF is fast, minimal dependencies
- **Scope**: Extract text + basic structure (page numbers); preserve headers/footers naturally
- **Out of scope**: OCR (no scanned PDFs in v1), table-specific parsing (use text extraction)
- **Fallback**: If PyMuPDF fails on unusual encodings, user gets clear error message

### Chunking Strategy
- **Experiment with 3 chunk sizes**: 300, 600, 900 characters
- **Overlap**: 10% overlap between chunks (to preserve context at boundaries)
- **Metadata**: Preserve page number, section header, chunk index
- **Phase 1 output**: Script `chunk_experiment.py` auto-tests all 3 sizes; outputs:
  - Total chunks per size
  - Coverage (% of handbook text retained)
  - Chunk distribution histogram
  - Quick search test: 5 sample queries → retrieve top-3 for each size
  - Recommendation: which size balances retrieval quality vs. context window
- **Phase 2 decision**: Use winner from Phase 1 experiment

### Embedding Model Selection
- **Primary**: all-MiniLM-L6-v2 (sentence-transformers, 384-dim, cross-lingual, ~90MB)
- **Fallback**: multilingual-e5-small (if Phase 2 benchmarking shows <0.5 cosine similarity for Vietnamese queries)
- **Download**: Automatic via `sentence_transformers.SentenceTransformer()` on first run (HuggingFace hub)
- **No checksum verification** (model is small, HuggingFace CDN is reliable)
- **Cache path**: Use HuggingFace default (`~/.cache/huggingface/hub/`) for portability

### Vector Storage
- **Choice**: ChromaDB with persistent SQLite backend
- **Storage path**: `./chroma_db/` (project root, relative path for portability)
- **Benefits**: 
  - Index persists between runs (no rebuild needed each startup)
  - Single-process access (no network/daemon required)
  - Metadata queryable (page #, section, chunk index)
- **Fallback**: FAISS in-memory (if ChromaDB fails; simpler but requires rebuild each run)
- **Not in scope**: Pinecone, Weaviate, external vector DB (complexity, dependencies)

### Project Structure
```
TotNghiepProject/
├── .planning/
│   └── phases/
│       └── 01-foundation/
│           ├── CONTEXT.md (this file)
│           ├── PLAN.md (tasks, estimates, gates)
│           └── outputs/
│               ├── pdf_extractor.py
│               ├── chunker.py
│               ├── chunk_experiment.py
│               ├── embedder.py
│               ├── vector_store.py
│               └── test_e2e.py
├── src/
│   ├── pdf_extraction.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── __init__.py
├── tests/
│   ├── test_pdf_extraction.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   └── test_vector_store.py
├── data/
│   ├── sample_handbook.pdf (test fixture)
│   └── chroma_db/ (vector index, persisted)
└── requirements.txt
```

---

## the agent's Discretion

**Dependency management:** Exact version pins (pip freeze) are developer's call, but recommend:
- PyMuPDF ~1.23.x
- sentence-transformers ~2.x
- chromadb ~0.4.x
- pytest ~7.x (testing)

**Error handling:** Graceful fallbacks for:
- PDF parsing errors → log + clear message (don't crash silently)
- Embedding model download failure → retry with exponential backoff
- ChromaDB connection failure → fallback to in-memory FAISS

**Testing strategy:** Unit tests for each module + integration test (upload sample PDF → embed → retrieve) in Phase 1 to catch issues early

**Performance profiling:** Basic timing logs (PDF extract time, chunk generation time, embedding time, total) to identify bottlenecks

---

## Canonical References

**Phase 1 must respect:**
- `.planning/REQUIREMENTS.md` — Feature requirements for PDF upload & indexing
- `.planning/ROADMAP.md` — Phase 1 exit gates and dependencies
- `.planning/PROJECT.md` — Hardware constraints (8GB RAM), target latency (≤5s E2E)

**External resources (optional reference):**
- PyMuPDF docs: https://pymupdf.readthedocs.io/
- Sentence Transformers docs: https://www.sbert.net/
- ChromaDB docs: https://docs.trychroma.com/

---

## Specific Ideas

### PDF Test Fixture
- Create `data/sample_handbook.pdf` with realistic HR content:
  - Employee handbook ~40 pages
  - Sections: Policies, Benefits, Leaves, Code of Conduct, FAQs
  - Some tables (leave accrual, salary bands)
  - Vietnamese text mixed with English terms (typical)

### Chunk Experiment Output
- Detailed CSV report: `chunk_experiment_results.csv`
  - Columns: chunk_size | num_chunks | total_chars | char_coverage | avg_chunk_quality | retrieval_f1_on_5_queries
  - Recommendation: "Use **600 chars** — balances coverage (92%) with retrieval quality (85%+ F1)"

### Success Metrics (Local, Phase 1 validation)
- ✓ PDF parses successfully (no text loss)
- ✓ Chunk counts within expected range (280–350 for 300 chars, 150–180 for 900 chars)
- ✓ Embeddings generate without error
- ✓ ChromaDB persists index
- ✓ Index loads from disk in <3s on target hardware
- ✓ Chunk size recommendation documented with rationale

---

## Deferred Ideas

- Multi-document index (v2) — Phase 1 handles single handbook only
- Table-aware parsing (v2) — Phase 1 treats tables as plain text
- Dynamic chunk sizing (v2) — Phase 1 uses fixed sizes for simplicity
- Query-time reranking (v2) — Phase 1 focuses on retrieval only, no LLM ranking
- Caching embeddings for speed (Phase 2) — Phase 1 computes on the fly

---

*Phase: 01-foundation*  
*Context gathered: 2026-05-04 (discuss-phase)*  
*Status: Ready for planning*
