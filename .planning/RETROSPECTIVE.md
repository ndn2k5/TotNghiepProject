# Project Retrospective

## v1.1 Report Revision and Finalization

**Completed:** 2026-06-22

### What Worked

- Treating local artifacts as an evidence ledger prevented intended targets from being reported as achieved values.
- Structural rewriting reduced the report from 42 to 26 pages without shrinking the USTH layout.
- Inspecting generated answers exposed the weakness of keyword-based hallucination flags despite improved aggregate metrics.

### What Changed

- The fine-tuning narrative moved from a negative-only first attempt to a chronological mixed result.
- RAG remains the factual-grounding architecture; fine-tuning is positioned as complementary.

### Future Evaluation

- Score base and fine-tuned GGUF models inside the same RAG pipeline.
- Use source-derived reference answers and HR-domain human review.
