# Requirements — Milestone 4: Thesis Report

**Project:** Development of an Internal AI Chatbot for Company Knowledge Retrieval  
**Institution:** University of Science and Technology of Hanoi (USTH)  
**Department:** Information and Communication Technology  
**Type:** Bachelor Thesis  
**Language:** English  
**Deadline:** 2026-06-24 (7 days from 2026-06-17)  
**Template:** `docs/2026_required_format_thesis.md`

---

## Deliverable: One complete, submission-ready thesis document

Max ~27 pages of body content (per department guideline), following USTH ICT template structure exactly.

---

## Required Sections (Template-Mandated)

| Section | Requirement | Target Length |
|---------|-------------|---------------|
| **Acknowledgements** | Thank supervisor(s), collaborators, institution | 0.5 page |
| **List of Abbreviations** | All acronyms used (RAG, LLM, BM25, RRF, etc.) | 1 page |
| **List of Tables** | Auto-generated or hand-listed | 0.5 page |
| **List of Figures** | Auto-generated or hand-listed | 0.5 page |
| **Abstract** | Max 250 words + 6 keywords, English | 1 page |
| **I/ Introduction** | Global context, literature review, problem statement | 2–3 pages |
| **II/ Objectives** | Scientific objective, strategy summary (2–3 sentences) | 0.5 page |
| **III/ Materials and Methods** | Architecture, data, models, algorithms — reproducible | 8–10 pages |
| **IV/ Results and Discussion** | Results + critical analysis + comparison with literature | 6–8 pages |
| **V/ Conclusion & Perspective** | Achievements, limitations, future work | 1–2 pages |
| **References** | Numbered, in order of appearance, DOI where possible | 1–2 pages |
| **Appendices** | Code snippets, full eval tables, screenshots | as needed |

---

## Content Requirements per Section

### Abstract
- Problem: Vietnamese employees can't quickly query HR policies offline
- Approach: RAG pipeline — local GGUF LLM + ChromaDB + hybrid BM25+vector retrieval
- Key result: system returns grounded Vietnamese answers; hybrid retrieval outperforms pure vector
- Keywords: RAG, Large Language Model, Vietnamese NLP, BM25, vector search, knowledge retrieval

### I/ Introduction
- **Context:** enterprise knowledge management, LLM capabilities in 2024–2025
- **Problem:** HR handbooks are long, employees spend 30–60 min searching; cloud LLMs leak data and cost per query
- **Literature review:** RAG [Lewis et al. 2020], retrieval augmentation, Vietnamese NLP challenges, Phi-3-Mini, sentence-transformers
- **This work:** local, offline Vietnamese HR chatbot using RAG + hybrid retrieval
- Must cite 8–12 references minimum

### II/ Objectives
- Design and implement a fully local RAG chatbot for Vietnamese HR knowledge retrieval
- Evaluate hybrid BM25 + vector retrieval against pure vector baseline
- Investigate QLoRA fine-tuning as an alternative to RAG (document as negative result)

### III/ Materials and Methods
Must cover all of the following sub-sections:

1. **System Architecture** — end-to-end pipeline diagram (question → retrieval → generation → answer)
2. **Knowledge Base Construction**
   - 8 manually-written Vietnamese HR policy documents
   - 12 additional documents generated via Minimax-M2.7 API
   - 20 documents total, covering: leave policy, contracts, salary, regulations, discipline, benefits, recruitment, social insurance, overtime, maternity, WFH, business travel, performance review, IT security, public holidays, anti-harassment, resignation, anti-corruption, workplace safety, allowances
3. **Document Processing**
   - Chunking: RecursiveCharacterTextSplitter, 600 chars, 100 overlap
   - Embedding: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ languages)
   - Vector store: ChromaDB persistent local store
4. **Hybrid Retrieval (BM25 + Vector + RRF)**
   - BM25Okapi keyword search over all chunks
   - Vector cosine similarity via ChromaDB
   - Reciprocal Rank Fusion: score(d) = α·rrf_vector(d) + (1−α)·rrf_bm25(d), k=60, α=0.5
5. **Answer Generation**
   - Model: Phi-3-Mini-4k-instruct GGUF (Q4 quantized, ~2.3GB)
   - Inference: llama-cpp-python, CPU with optional CUDA offload
   - Prompt format: Phi-3 chat tokens (`<|user|>...<|end|>\n<|assistant|>`)
   - Context: top-5 chunks formatted with source label
6. **QLoRA Fine-Tuning Experiment (Negative Result)**
   - Attempt: fine-tune Phi-3-Mini on 3343 synthetic Q&A pairs
   - Finding: 63% of training data was non-HR content (Japanese labor law, US Medicare)
   - Result: catastrophic hallucinations; abandoned in favor of RAG
   - Lesson: data quality gates are critical before fine-tuning

### IV/ Results and Discussion
Must cover:

1. **Retrieval Comparison**
   - Pure vector baseline vs. hybrid BM25+vector
   - Demonstrate: "1 tháng được nghỉ bao ngày" → pure vector returns salary doc (wrong), hybrid returns leave policy doc (correct)
   - Qualitative analysis of 5–10 example queries

2. **System Demo**
   - Screenshot/transcript of working chatbot answering Vietnamese HR questions
   - Show: retrieval latency (~0.04–0.13s), generation latency (~15–85s CPU), source attribution

3. **QLoRA Negative Result Analysis**
   - Training: 3343 pairs × 3 epochs on NVIDIA T1200 (4.3GB VRAM), 22 hours
   - Eval results showing hallucinations (Medicare answers for HR questions)
   - Root cause: data quality failure — crawled PDFs were garbage
   - Comparison: RAG + base model >> fine-tuned model on this dataset

4. **Discussion**
   - RAG is robust to model capability limitations when retrieval is good
   - Hybrid search significantly improves recall for short Vietnamese queries
   - Limitations: phi-3-mini cannot perform implicit arithmetic (12 days/year → 1/month)
   - Synthetic knowledge base trades authenticity for coverage

### V/ Conclusion
- Achieved: working offline Vietnamese HR chatbot with hybrid retrieval
- Key finding: hybrid BM25+vector retrieval outperforms pure vector for short Vietnamese queries
- Key lesson: data quality gates matter more than model size for fine-tuning
- Future work: fine-tuning with quality-gated data, larger context window, user evaluation study, web UI deployment

---

## Non-Functional Requirements

- **Format:** USTH ICT template exactly — title page, supervisor certification, numbered references [n]
- **Citations:** 10–15 references minimum; include seminal RAG paper, Phi-3 paper, sentence-transformers, BM25 original paper
- **Figures required:** system architecture diagram, pipeline flowchart, retrieval comparison table/figure, eval results table
- **Code:** key code snippets in Appendix (not in body); body uses pseudo-code / description only
- **Language:** formal English; avoid first person ("we" acceptable, "I" acceptable per template)

---

## Key References to Cite

| # | Paper | Why |
|---|-------|-----|
| [1] | Lewis et al. 2020 — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" | Foundational RAG paper |
| [2] | Robertson & Zaragoza 2009 — "The Probabilistic Relevance Framework: BM25 and Beyond" | BM25 algorithm |
| [3] | Cormack et al. 2009 — "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" | RRF formula |
| [4] | Microsoft 2024 — "Phi-3 Technical Report" | phi-3-mini model |
| [5] | Reimers & Gurevych 2019 — "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" | Sentence-transformers |
| [6] | Dettmers et al. 2023 — "QLoRA: Efficient Finetuning of Quantized LLMs" | QLoRA method |
| [7] | Hu et al. 2022 — "LoRA: Low-Rank Adaptation of Large Language Models" | LoRA |
| [8] | Chroma (2023) — ChromaDB documentation | vector store |
| [9] | BLLĐ 2019 — Bộ Luật Lao Động Việt Nam | Vietnamese labor law context |
| [10] | Nguyen et al. — Vietnamese NLP challenges | motivation for multilingual embeddings |

---

**Milestone created:** 2026-06-17  
**Deadline:** 2026-06-24  
**Owner:** Student (solo writing)  
**Exit gate:** Submitted to supervisor by 2026-06-24; all required sections complete; references formatted
