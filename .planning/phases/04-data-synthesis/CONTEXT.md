# Phase 4 Context

## Why This Phase Exists

v1 RAG uses off-the-shelf models. All four quality metrics fail:
- Hallucinations in answers
- Bad Vietnamese output
- Poor retrieval (wrong chunks returned)
- Vague, generic answers

Fine-tuning (Phases 5 + 6) requires labeled data. This phase builds that data synthetically — zero manual labeling — from 3 public GitHub HR handbooks.

## Source Handbooks

| Repo | Content type |
|------|-------------|
| `hshadab/handbook` | General employee handbook |
| `cuesoftinc/handbook` | Software company handbook |
| `ultralytics/handbook` | AI company handbook |

All English Markdown. We generate Vietnamese Q&A pairs, not translate the source.

## Teacher LLM

**Claude claude-haiku-4-5-20251001** — chosen for:
- Excellent Vietnamese (native quality)
- Cheap enough for 1500+ pairs (~$0.50)
- Fast API throughput
- Can be instructed to stay grounded in source text (no hallucination)

Set `ANTHROPIC_API_KEY` before running Task 3.

## Output Formats

### Embedding Training (Phase 5 input)
```json
{"anchor": "Nhân viên được nghỉ bao nhiêu ngày?", "positive": "<handbook chunk>", "hard_negative": "<different chunk>"}
```

### LLM SFT Training (Phase 6 input)
```json
{"system": "Bạn là trợ lý nhân sự...", "instruction": "<Vietnamese question>", "input": "<handbook context>", "output": "<Vietnamese answer>"}
```

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| English source → Vietnamese Q&A | Source handbooks are English; we want Vietnamese RAG output |
| Claude Haiku as teacher | Cheapest model with reliable Vietnamese + instruction-following |
| Hard negatives from different handbook | Cross-source negatives are harder than same-source; better embedding training signal |
| Checkpoint-based generation | Task 3 runs for 1–3 hours; checkpoints allow resume on failure |
| 80/10/10 split | Standard; test set reserved for evaluation after Phase 5+6 |

## Hardware Notes

- Tasks 1–8: CPU (local Windows machine)
- LLM API calls: Anthropic cloud (Task 3)
- After Task 5: upload `data/splits/` to H100 for Phase 5 + Phase 6

## Downstream Consumers

- **Phase 5** uses `data/splits/embedding_train_split.jsonl`
- **Phase 6** uses `data/splits/llm_train_split.jsonl`
- **Phase 7** uses `data/sample_handbook.pdf` for RAG demo
- **Phase 7** uses `data/splits/embedding_test_split.jsonl` for benchmark

## Created

2026-06-01
