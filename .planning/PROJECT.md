# HR Policy RAG Chatbot

## What This Is

A fully local, two-agent RAG chatbot that lets employees ask questions about company HR policies in Vietnamese and get instant, accurate answers directly from their employee handbook PDF. No cloud, no API costs—just privacy and speed on any mid-range CPU machine.

## Core Value

**Employees get instant, consistent, offline answers to HR questions in Vietnamese.**

If this works well, it becomes the single source of truth for policy questions and eliminates repetitive HR inquiries. If it fails, employees waste time hunting through PDFs.

## Requirements

### Validated

(None yet — shipping v1 to validate)

### Active (v1 MVP)

- [ ] **PDF Upload & Indexing** – Accept single employee handbook PDF, auto-convert to ~500-char Markdown chunks (200–400 chunks for 30–60 page handbook)
- [ ] **Question Normalization (Agent 1)** – Qwen-2.5-1.5B preprocessor handles Vietnamese diacritics and colloquial phrasings, extracts keywords
- [ ] **Semantic Search** – all-MiniLM-L6-v2 embeddings (384-dim, cross-lingual) retrieve top 3 relevant chunks
- [ ] **Answer Generation (Agent 2)** – Phi-3-Mini responder generates fluent, Markdown-formatted Vietnamese answer based on retrieved context
- [ ] **Streamlit Chat Interface** – Clean UI showing question, answer, and source chunk references (page/section)
- [ ] **80%+ Relevance** – Manual validation: 80%+ of 30–50 test queries retrieve ≥1 relevant chunk
- [ ] **≤5s Response Time** – End-to-end inference on 8GB RAM, 4-core CPU (no GPU)
- [ ] **Zero Hallucination** – Answer grounded in handbook, no made-up policies

### Out of Scope (v1)

- Multi-handbook support — will add in v2 (complexity of multi-index management)
- Admin UI to upload new handbooks — will add in v2 (v1: manual re-index on file drop)
- Fine-tuning or domain-specific models — use off-the-shelf GGUF releases (cost/speed tradeoff)
- Multi-language UI — Vietnamese output only (scope is HR handbook, not translation engine)
- Fuzzy matching or typo tolerance — handled by question normalizer (Agent 1)

## Context

**What prompted this:**
- Time waste: employees spend 30 min–1 hour searching a 50-page PDF for a single policy
- Inconsistency: different HR reps give slightly different advice
- Privacy/latency: cloud LLMs leak data, require internet, cost per query

**Technical ecosystem:**
- GGUF inference via llama-cpp-python (CPU-only, no CUDA)
- Sentence transformers for multilingual embeddings
- Streamlit for UI (Python-friendly, no frontend build)
- Pinecone/Supabase not needed — use local in-memory or SQLite vector store for v1

**Team & constraints:**
- Solo developer, 3-week timeline, ~10–15 hours/week
- Target hardware: laptop or small VM (8GB RAM, 4 cores, e.g., AWS t3.medium, MacBook Intel i5)

## Constraints

- **Hardware**: Must run on 8GB RAM, 4 CPU cores, no GPU — total model size (both GGUF + embeddings) ≤ 6GB loaded
- **Latency**: ≤5 seconds end-to-end (includes embedding, search, inference)
- **Language**: Vietnamese input/output fluency is non-negotiable; English fallback acceptable for debugging
- **Accuracy**: No hallucination — all answers traceable to handbook chunks
- **Offline**: Zero external API calls (no OpenAI, Hugging Face Hub downloads at inference time)
- **Handbook size**: Assume 30–60 pages (~15k–30k words) → ~200–400 chunks

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| **Use off-the-shelf GGUF models (Qwen-2.5-1.5B, Phi-3-Mini)** | Avoid fine-tuning complexity; models are small, fast, well-tested on Vietnamese | ✓ Good — ships fast |
| **Two-stage pipeline (normalizer + responder)** | Simplifies prompt engineering; Agent 1 focuses on question clarity, Agent 2 focuses on answer quality | ✓ Good — modular, testable |
| **all-MiniLM-L6-v2 as primary embedding** | 384-dim is fast; multilingual; trade: may underperform Vietnamese vs. multilingual-e5-small | — Pending — will benchmark in Phase 1 |
| **Streamlit for UI** | Quick to build, Python-native, no frontend DevOps | ✓ Good — focuses on backend logic |
| **Local vector store (SQLite or in-memory)** | Avoid Pinecone/Supabase complexity; handbook is static for v1 | ✓ Good — one less service |

---

**Last updated:** 2026-05-04 (project init)  
**Team:** Solo developer  
**Status:** Scoping → Requirements → Roadmap → Phase 1 Planning

