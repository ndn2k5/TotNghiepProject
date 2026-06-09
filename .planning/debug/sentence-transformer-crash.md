---
slug: sentence-transformer-crash
status: resolved
trigger: "SentenceTransformer crash with exit code 1"
goal: find_and_fix
tdd_mode: false
---

# Debug Session: sentence-transformer-crash

## Symptoms
- `SentenceTransformer('all-MiniLM-L6-v2', device='cpu')` crashes with exit code 1 and NO output/trace.
- `python -c "from sentence_transformers import SentenceTransformer; ..."` crashes immediately.
- `transformers` version is 4.43.4.
- `sentence-transformers` version is 5.5.1 (verified via pip show).

## Current Focus
**Hypothesis:** Version incompatibility or corrupted cache/installation.
**Next Action:** Fixed by installing stable torch version.

## Evidence
- timestamp: 2026-06-09T21:32:00Z
  observation: Initial report. Versions: transformers 4.43.4, sentence-transformers 5.5.1.
- timestamp: 2026-06-09T22:05:00Z
  observation: Reinstallation of transformers/sentence-transformers did not fix the issue. Crash still has exit code 1.
- timestamp: 2026-06-09T22:10:00Z
  observation: Reproduced crash in test venv. Found that torch 2.12.0+cpu is non-standard.
- timestamp: 2026-06-09T22:15:00Z
  observation: Installed torch 2.5.1+cpu. Import and encoding now work perfectly.

## Resolution
**root_cause:** Non-standard/corrupted `torch-2.12.0+cpu` caused silent crash on import.
**fix:** Downgraded/Reinstalled stable `torch-2.5.1+cpu` from official PyTorch index.
