# HR Policy RAG Chatbot

## What This Is

A fully local Vietnamese HR policy chatbot combining hybrid retrieval and a
Phi-3-Mini GGUF responder. The implementation, synthetic HR corpus, H200 QLoRA
training run, evaluation artifacts, and bachelor thesis source are all present
in this repository.

## Core Value

**Employees receive grounded Vietnamese HR answers from local company policy
documents without sending private data to cloud services.**

## Current State

**Shipped:** v1.1 Report Revision and Finalization (2026-06-22)

The thesis is evidence-accurate, compiled, and ready for supervisor review.
It contains 26 total pages, down from 42, and reports the completed H200 dataset,
QLoRA run, and evaluation with explicit factual-accuracy caveats.

## Requirements

### Validated

- ✓ Local PDF ingestion, chunking, embedding, and ChromaDB indexing — Phase 1
- ✓ Vietnamese query normalization and semantic retrieval — Phase 2
- ✓ Phi-3-Mini response generation and Streamlit interface — Phase 3
- ✓ Twenty-topic Vietnamese HR knowledge base and hybrid BM25/vector retrieval — Phase 7
- ✓ Qwen2.5-72B synthetic generation produced 4,900 unique Q&A pairs — H200 run
- ✓ Phi-3-Mini QLoRA adapter and Q4_K_M GGUF exported and downloaded — H200 run
- ✓ Base/fine-tuned comparison artifacts recorded for 20 fixed questions — H200 evaluation
- ✓ Evidence-backed 26-page USTH thesis compiled and audited — v1.1 / Phase 11

### Active

(None — awaiting supervisor feedback.)

### Out of Scope

- Additional cloud/GPU training — the Vast.ai instance is terminated and all required artifacts are local.
- New chatbot features — this milestone is report revision, not product expansion.
- Fabricated benchmark values — missing measurements must be identified as limitations.
- A new human-subject study — unavailable before the submission deadline.

## Context

- Supervisor feedback: the 42-page report is too long; target about 27 total pages.
- Deadline context: final revision is urgent and follows the completed 2026-06-22 H200 run.
- Current dataset: 4,900 pairs, 20 documents, 82.4% keyword-based domain relevance,
  zero red flags, and zero exact duplicate questions.
- Current model evaluation: automated score 0.62 to 0.70, HR-relevance flag
  13/20 to 14/20, and average latency 3.9 s to 0.9 s. Several fine-tuned
  answers remain factually incorrect, so these metrics do not establish policy accuracy.
- Current source/PDF: `report/thesis.tex` compiles cleanly to a 26-page `report/thesis.pdf`.

## Constraints

- **Length:** Approximately 27 total PDF pages, not merely 27 body pages — supervisor direction.
- **Evidence:** Every new number must be reproducible from repository artifacts — academic integrity.
- **Format:** Preserve the required USTH thesis structure and formal English.
- **Time:** Complete without new GPU or external data collection — Vast.ai is terminated.
- **Scope:** Prefer deletion and synthesis over shrinking fonts or margins — readability matters.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep hybrid RAG as the deployed architecture | It grounds answers in retrieved policy text | ✓ Good |
| Report the new QLoRA run as a mixed result | Speed and heuristic relevance improved, factual correctness remains weak | ✓ Good |
| Remove full code/evaluation appendices | They consume pages and are reproducible from the repository | ✓ Good — 26 pages |
| Use only downloaded artifacts for statistics | Prevents unsupported thesis claims | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:** review invalidated, validated, and newly emerged
requirements; record consequential decisions; and update the product description
if implementation reality changed.

**After each milestone:** review all sections, re-check the core value, audit
out-of-scope decisions, and update context with verified outcomes.

---
*Last updated: 2026-06-22 after completing milestone v1.1*
