# Phase 4: Data Synthesis Pipeline — Detailed Execution Plan

**Phase:** 04-data-synthesis  
**Duration:** ~1 week  
**Team:** Both developers  
**Execution Model:** Sequential within phase; Phase 5 + 6 unblock after Task 5  
**Status:** Ready to execute  
**Created:** 2026-06-01  
**Requirements:** `.planning/REQUIREMENTS_M2.md` §1

---

## Phase Goal

**Build a Vietnamese HR Q&A dataset entirely from 3 public English GitHub handbooks using a teacher LLM — zero manual labeling. Output: ≥1500 Vietnamese QA pairs in both embedding-training and LLM SFT formats, plus a sample handbook PDF for demo.**

### Measurable Success Criteria

- ≥1500 Vietnamese Q&A pairs in `data/qa_pairs.jsonl`
- Train/dev/test split: 80/10/10 — files in `data/splits/`
- Embedding triplets in `data/embedding_train.jsonl` (≥1200 rows)
- SFT JSONL in `data/llm_train.jsonl` (≥1200 rows)
- ≥10 spot-checked QA pairs manually reviewed and approved
- `data/sample_handbook.pdf` generated from at least 1 handbook
- `notebooks/FINETUNING_A0-DataPreparation.ipynb` runs cell-by-cell without error

---

## Scope

### In Scope

- Clone and parse 3 GitHub handbooks (English Markdown only)
- LLM-based Vietnamese QA generation (2–3 pairs per chunk)
- Hard negative mining from same handbook (BM25 or cosine similarity)
- Training format conversion (embedding triplets + SFT JSONL)
- Markdown → PDF conversion for demo
- Dataset quality checks (count, JSON validity, spot review)
- Notebook finalization

### Out of Scope

- Translation of handbook text (only Q&A output is Vietnamese)
- Manual annotation or labeling
- Web scraping beyond the 3 specified GitHub repos
- Vietnamese handbooks — using English source, Vietnamese Q&A
- Model training (Phase 5 + 6)

---

## Dependency Map

```text
Task 1 (Setup & Handbook Ingestion)
    ↓
Task 2 (Text Cleaning & Chunking)
    ↓
Task 3 (Vietnamese QA Generation)   ← longest task; ~2 hours LLM calls
    ↓
Task 4 (Quality Review & Filtering)
    ↓
Task 5 (Training Format Conversion) ← UNBLOCKS Phase 5 + Phase 6
    ↓
Task 6 (Handbook → PDF Conversion)
    ↓
Task 7 (Dataset Validation)
    ↓
Task 8 (Notebook Finalization & Commit)
```

**Critical Path:** Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8  
**Parallel opportunity:** After Task 5, Phase 5 (embedding) and Phase 6 (LLM QLoRA) can start on H100 while Task 6–8 complete.

---

## Source Handbooks

| Repo | URL | Expected content |
|------|-----|-----------------|
| hshadab/handbook | `https://github.com/hshadab/handbook` | Employee handbook (benefits, conduct, leave) |
| cuesoftinc/handbook | `https://github.com/cuesoftinc/handbook` | Software company handbook |
| ultralytics/handbook | `https://github.com/ultralytics/handbook` | AI company handbook |

**Total estimated content:** 50k–100k words across all 3 handbooks.  
**Target chunks:** ~600–900 chunks across all 3 → ~2–3 QA pairs each → 1500–2700 raw pairs.

---

## Work Breakdown

### Task 1: Setup & Handbook Ingestion (1 hour)

**Owner:** Both developers  
**Objective:** Clone 3 GitHub handbooks, extract all Markdown text, verify content coverage

**Acceptance Criteria:**

- [ ] `data/raw/` directory created with 3 subdirectories (one per handbook)
- [ ] All `.md` files fetched from each repo
- [ ] Total extracted text > 50k words across all 3 handbooks
- [ ] `data/raw_chunks.jsonl` created with cleaned text chunks
- [ ] Script `scripts/ingest_handbooks.py` runs without errors

**Deliverables:**

```text
scripts/ingest_handbooks.py
data/raw/hshadab/           (cloned markdown files)
data/raw/cuesoftinc/        (cloned markdown files)
data/raw/ultralytics/       (cloned markdown files)
data/raw_chunks.jsonl       (all chunks from all 3 handbooks)
```

**Code Skeleton — `scripts/ingest_handbooks.py`:**

```python
"""
Clone 3 GitHub handbooks, extract Markdown, chunk, save to data/raw_chunks.jsonl
"""
import subprocess
import os
import json
import re
from pathlib import Path

HANDBOOKS = [
    {"name": "hshadab", "url": "https://github.com/hshadab/handbook"},
    {"name": "cuesoftinc", "url": "https://github.com/cuesoftinc/handbook"},
    {"name": "ultralytics", "url": "https://github.com/ultralytics/handbook"},
]

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/raw_chunks.jsonl")
CHUNK_SIZE = 500  # chars
CHUNK_OVERLAP = 50  # chars


def clone_or_pull(repo_url: str, target_dir: Path):
    if target_dir.exists():
        print(f"  Already exists: {target_dir}, skipping clone")
        return
    subprocess.run(["git", "clone", "--depth=1", repo_url, str(target_dir)], check=True)


def extract_markdown_files(repo_dir: Path) -> list[tuple[str, str]]:
    """Return list of (filename, content) for all .md files."""
    results = []
    for md_file in repo_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            results.append((str(md_file.relative_to(repo_dir)), content))
        except Exception as e:
            print(f"  Warning: could not read {md_file}: {e}")
    return results


def clean_markdown(text: str) -> str:
    """Strip markdown syntax, keep readable text."""
    text = re.sub(r"```[\s\S]*?```", "", text)  # code blocks
    text = re.sub(r"`[^`]+`", "", text)         # inline code
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # links → text
    text = re.sub(r"#{1,6}\s*", "", text)        # headings → plain
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)      # italic
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse blank lines
    return text.strip()


def chunk_text(text: str, source: str, filename: str) -> list[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) > 100:  # skip tiny chunks
            chunks.append({
                "text": chunk,
                "source": source,
                "filename": filename,
                "char_start": start,
                "char_end": end
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for handbook in HANDBOOKS:
        name = handbook["name"]
        url = handbook["url"]
        repo_dir = RAW_DIR / name

        print(f"\n[{name}] Cloning from {url}...")
        clone_or_pull(url, repo_dir)

        md_files = extract_markdown_files(repo_dir)
        print(f"[{name}] Found {len(md_files)} markdown files")

        for filename, content in md_files:
            clean = clean_markdown(content)
            if len(clean) < 200:
                continue  # skip too-short files
            file_chunks = chunk_text(clean, source=name, filename=filename)
            all_chunks.extend(file_chunks)

        print(f"[{name}] Extracted chunks so far: {len(all_chunks)}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Total words (approx): {sum(len(c['text'].split()) for c in all_chunks)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

**Action Steps:**

1. Ensure `git` is installed and accessible
2. Create `data/` directory: `mkdir data`
3. Run ingestion:
   ```powershell
   python scripts/ingest_handbooks.py
   ```
4. Verify output:
   ```powershell
   python -c "
   import json
   with open('data/raw_chunks.jsonl') as f:
       chunks = [json.loads(l) for l in f]
   print(f'Chunks: {len(chunks)}')
   words = sum(len(c[\"text\"].split()) for c in chunks)
   print(f'Total words: {words}')
   sources = set(c[\"source\"] for c in chunks)
   print(f'Sources: {sources}')
   "
   ```

**Verification Command:**

```powershell
python -c "import json; data=open('data/raw_chunks.jsonl').readlines(); print(f'{len(data)} chunks total')"
```

**Time Estimate:** ~30–45 min (includes git clone time)

---

### Task 2: Text Cleaning & Quality Check (0.5 hours)

**Owner:** Both developers  
**Objective:** Review sample chunks, adjust cleaning, confirm chunk quality before spending H100 compute

**Note from research:** Use `langchain_text_splitters.MarkdownTextSplitter` instead of custom regex — already installed (`langchain-text-splitters==1.1.2`). It preserves markdown headers as chunk prefixes for better context.

**Acceptance Criteria:**

- [ ] Manually review 20 random chunks — all readable English HR policy text
- [ ] No chunks are code blocks, navigation menus, or boilerplate (script already strips these)
- [ ] Chunks with <100 chars filtered out
- [ ] Total chunk count documented: target 600–900 chunks
- [ ] If chunk count < 600: reduce `CHUNK_SIZE` to 400 in `ingest_handbooks.py` and re-run Task 1

**Deliverables:**

```text
data/raw_chunks.jsonl  (verified, clean)
```

**Action Steps:**

1. Spot-check 20 random chunks:
   ```python
   import json, random
   chunks = [json.loads(l) for l in open("data/raw_chunks.jsonl")]
   print(f"Total chunks: {len(chunks)}")
   sample = random.sample(chunks, 20)
   for i, c in enumerate(sample):
       print(f"\n--- Chunk {i+1} ({c['source']}/{c['filename']}) ---")
       print(c["text"][:300])
   ```
2. If chunks look bad (code, menus, garbage): adjust `clean_markdown()` in `ingest_handbooks.py` and re-run Task 1
3. Document total chunk count — if < 600, lower `CHUNK_SIZE` to 400

**Time Estimate:** ~20–30 min

---

### Task 3: Vietnamese QA Generation (2–4 hours, mostly unattended on H100)

**Owner:** Both developers  
**Objective:** Use Qwen2.5-72B on H100 (via vLLM + ngrok tunnel) to generate 2–3 Vietnamese Q&A pairs per handbook chunk

**Research finding:** Use self-hosted H100 model instead of paid API — free, no rate limits, higher quality at 72B scale.

**Acceptance Criteria:**

- [ ] vLLM running on H100 with Qwen2.5-72B (or 32B fallback)
- [ ] ngrok tunnel active; URL passed to generation script
- [ ] ≥1500 QA pairs total (≥400 per handbook)
- [ ] Each pair: `{question, answer, source, filename, chunk_text}`
- [ ] Questions: natural Vietnamese; what an employee might actually ask
- [ ] Answers: grounded strictly in chunk text — no hallucination
- [ ] `data/qa_pairs.jsonl` with checkpoint recovery (resume on failure)

**Deliverables:**

```text
scripts/generate_qa.py
data/qa_pairs.jsonl          (≥1500 pairs)
data/qa_generation_log.txt   (timing + error log)
data/qa_checkpoint.json      (resume state)
```

**Prerequisites (install before running):**

```powershell
pip install openai          # OpenAI-compatible SDK for vLLM calls
```

**H100 setup (run on Google Colab with H100 or A100 instance):**

Open a new Google Colab notebook, select the H100 or A100 runtime, and run the following in a cell to install and start vLLM with ngrok:

```python
!pip install vllm pyngrok
import subprocess
import time
from pyngrok import ngrok

# Authenticate ngrok (replace with your actual token)
ngrok.set_auth_token("YOUR_NGROK_TOKEN")

# Start vLLM in the background (using Qwen2.5-32B or 72B FP8 depending on Colab memory)
vllm_cmd = "python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --dtype bfloat16 --max-model-len 4096 --host 0.0.0.0 --port 8000"
process = subprocess.Popen(vllm_cmd.split())

# Wait for server to initialize
print("Starting vLLM server... this will take a few minutes.")
time.sleep(30) 

# Open ngrok tunnel
public_url = ngrok.connect(8000).public_url
print(f"Ngrok URL: {public_url}")
# Pass this URL to generate_qa.py via --vllm-url argument
```

**Code Skeleton — `scripts/generate_qa.py`:**

```python
"""
Generate Vietnamese Q&A pairs from handbook chunks using
Qwen2.5-72B served via vLLM + ngrok tunnel.
Checkpoint-safe: re-run after failure to resume.

Usage:
    python scripts/generate_qa.py --vllm-url https://xxxx.ngrok-free.app
"""
import argparse
import json
import re
import time
from pathlib import Path

import httpx
from openai import OpenAI

INPUT_FILE = Path("data/raw_chunks.jsonl")
OUTPUT_FILE = Path("data/qa_pairs.jsonl")
LOG_FILE = Path("data/qa_generation_log.txt")
CHECKPOINT_FILE = Path("data/qa_checkpoint.json")

SYSTEM_PROMPT = """Bạn là chuyên gia nhân sự. Hãy đọc đoạn văn bản từ sổ tay nhân viên \
(bằng tiếng Anh) và tạo ra {n_pairs} cặp hỏi-đáp bằng tiếng Việt.

Yêu cầu:
1. Câu hỏi phải là câu hỏi thực tế mà nhân viên có thể đặt ra.
2. Câu trả lời phải DỰA HOÀN TOÀN vào đoạn văn bản được cung cấp — không thêm thông tin bên ngoài.
3. Câu trả lời phải rõ ràng, ngắn gọn, bằng tiếng Việt chuẩn.
4. Sử dụng đầy đủ dấu thanh tiếng Việt (ă, â, đ, ê, ô, ơ, ư và các dấu hỏi, ngã, nặng, sắc, huyền).
5. Đầu ra phải là JSON hợp lệ theo định dạng sau, không thêm văn bản khác.

Định dạng JSON:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""

USER_TEMPLATE = """Đoạn văn bản:
{chunk_text}

Hãy tạo {n_pairs} cặp hỏi-đáp từ đoạn văn bản trên."""


def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text()).get("processed", []))
    return set()


def save_checkpoint(processed: set):
    CHECKPOINT_FILE.write_text(json.dumps({"processed": list(processed)}))


def parse_json_response(raw: str) -> list[dict]:
    """Strip markdown fences and parse JSON array."""
    # Strip code fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
    # Try to extract first [...] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def generate_qa_for_chunk(client: OpenAI, chunk: dict, model_name: str) -> list[dict]:
    n_pairs = 3 if len(chunk["text"]) >= 400 else 2
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(n_pairs=n_pairs)},
            {"role": "user", "content": USER_TEMPLATE.format(
                chunk_text=chunk["text"], n_pairs=n_pairs)},
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    pairs = parse_json_response(raw)
    for pair in pairs:
        pair["source"] = chunk["source"]
        pair["filename"] = chunk["filename"]
        pair["chunk_text"] = chunk["text"]
    return [p for p in pairs if len(p.get("question", "")) >= 10
            and len(p.get("answer", "")) >= 20]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", required=True,
                        help="ngrok HTTPS URL, e.g. https://xxxx.ngrok-free.app")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct",
                        help="Model name as served by vLLM")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: only process first 10 chunks")
    args = parser.parse_args()

    # Extend timeout — 72B model can take 30-90s per request
    client = OpenAI(
        base_url=f"{args.vllm_url}/v1",
        api_key="fake",
        http_client=httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10)),
    )

    chunks = [json.loads(l) for l in open(INPUT_FILE, encoding="utf-8")]
    if args.test:
        chunks = chunks[:10]
        print("TEST MODE: processing first 10 chunks only")

    processed = load_checkpoint()
    existing_qa = sum(1 for _ in open(OUTPUT_FILE)) if OUTPUT_FILE.exists() else 0
    print(f"Chunks: {len(chunks)} | Already processed: {len(processed)} | Existing QA: {existing_qa}")

    total_qa = existing_qa
    errors = 0
    start = time.time()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, \
         open(LOG_FILE, "a", encoding="utf-8") as log_f:

        for idx, chunk in enumerate(chunks):
            if idx in processed:
                continue
            if len(chunk["text"].split()) < 30:
                processed.add(idx)
                continue

            try:
                pairs = generate_qa_for_chunk(client, chunk, args.model)
                for pair in pairs:
                    out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_qa += len(pairs)
                processed.add(idx)

                if idx % 10 == 0:
                    elapsed = time.time() - start
                    rate = max((idx + 1) / elapsed, 1e-9)
                    eta_min = (len(chunks) - idx - 1) / rate / 60
                    print(f"[{idx+1}/{len(chunks)}] QA: {total_qa} | ETA: {eta_min:.1f}min")
                    save_checkpoint(processed)

            except json.JSONDecodeError as e:
                errors += 1
                log_f.write(f"JSON error chunk {idx}: {e}\n")
                processed.add(idx)  # skip malformed

            except Exception as e:
                errors += 1
                log_f.write(f"Error chunk {idx}: {e}\n")
                print(f"  Error chunk {idx}: {e} — retrying in 10s")
                time.sleep(10)

    save_checkpoint(processed)
    print(f"\n✓ Done. QA pairs: {total_qa} | Errors: {errors} | Time: {(time.time()-start)/60:.1f}min")


if __name__ == "__main__":
    main()
```

**Action Steps:**

1. Install dependency on local machine:
   ```powershell
   pip install openai
   ```
2. On H100 machine: start vLLM (pick Option A/B/C above), then start ngrok
3. Test on 10 chunks first:
   ```powershell
   python scripts/generate_qa.py --vllm-url https://xxxx.ngrok-free.app --test
   ```
4. Review 5 generated pairs manually for Vietnamese quality + grounding
5. If quality OK, run full generation (runs unattended 1–3 hours):
   ```powershell
   python scripts/generate_qa.py --vllm-url https://xxxx.ngrok-free.app
   ```
6. Script is checkpoint-safe — re-run with same URL if it crashes

**Verification Command:**

```powershell
python -c "
import json, random
pairs = [json.loads(l) for l in open('data/qa_pairs.jsonl', encoding='utf-8')]
print(f'Total QA pairs: {len(pairs)}')
for src in set(p['source'] for p in pairs):
    print(f'  {src}: {sum(1 for p in pairs if p[\"source\"]==src)}')
s = random.choice(pairs)
print(f'\nSample Q: {s[\"question\"]}')
print(f'Sample A: {s[\"answer\"]}')
"
```

**Time Estimate:** ~20 min setup + 1–3 hours unattended on H100

---

### Task 4: Quality Review & Filtering (0.5 hours)

**Owner:** Both developers (manual review step)  
**Objective:** Spot-check 20+ QA pairs, filter out bad ones, verify Vietnamese quality

**Acceptance Criteria:**

- [ ] Manually reviewed ≥20 random QA pairs (document results)
- [ ] No hallucinated answers (answers use only chunk text)
- [ ] No English mixed into Vietnamese questions/answers
- [ ] No malformed JSON in output file
- [ ] Filter: remove pairs where `len(question) < 10` or `len(answer) < 20`
- [ ] Post-filter count still ≥1500

**Deliverables:**

```text
data/qa_pairs_filtered.jsonl    (clean pairs only)
data/qa_review_notes.txt        (10-line manual review summary)
```

**Code Skeleton — `scripts/filter_qa.py`:**

```python
import json
from pathlib import Path

INPUT = Path("data/qa_pairs.jsonl")
OUTPUT = Path("data/qa_pairs_filtered.jsonl")

MIN_Q_LEN = 10
MIN_A_LEN = 20

def is_valid(pair: dict) -> bool:
    q = pair.get("question", "")
    a = pair.get("answer", "")
    if len(q) < MIN_Q_LEN or len(a) < MIN_A_LEN:
        return False
    # Basic check: answer should reference content from chunk
    if not pair.get("chunk_text"):
        return False
    return True

pairs = [json.loads(l) for l in open(INPUT, encoding="utf-8")]
valid = [p for p in pairs if is_valid(p)]
removed = len(pairs) - len(valid)

print(f"Input: {len(pairs)} | Valid: {len(valid)} | Removed: {removed}")

with open(OUTPUT, "w", encoding="utf-8") as f:
    for p in valid:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"✓ Saved {len(valid)} filtered pairs to {OUTPUT}")
```

**Action Steps:**

1. Run filter:
   ```powershell
   python scripts/filter_qa.py
   ```
2. Manual review script:
   ```python
   import json, random
   pairs = [json.loads(l) for l in open("data/qa_pairs_filtered.jsonl", encoding="utf-8")]
   sample = random.sample(pairs, 20)
   for i, p in enumerate(sample):
       print(f"\n=== {i+1} === [{p['source']}]")
       print(f"Q: {p['question']}")
       print(f"A: {p['answer']}")
       print(f"Chunk: {p['chunk_text'][:150]}...")
   ```
3. Write `data/qa_review_notes.txt` with observations (5–10 lines)
4. If quality is poor: adjust prompt in Task 3 and regenerate

**Time Estimate:** ~30 min

---

### Task 5: Training Format Conversion (1 hour)

**Owner:** Both developers  
**Objective:** Convert QA pairs to embedding triplet format (using `mine_hard_negatives`) and SFT JSONL (TRL `messages` format); create train/dev/test splits. This task UNBLOCKS Phase 5 and Phase 6.

**Research findings applied:**

- Hard negatives: use `sentence_transformers.util.mine_hard_negatives` (already installed v5.5.1) — semantic mining beats random selection
- SFT format: TRL `messages` (conversational) not Alpaca — natively consumed by `SFTTrainer`, no formatting function needed
- `multilingual-e5-small` requires `"query: "` prefix on anchors and `"passage: "` prefix on positives at inference time

**Acceptance Criteria:**

- [ ] `data/embedding_train.jsonl`: `{anchor, positive, hard_negative}` triplets from `mine_hard_negatives` — ≥1200 rows
- [ ] `data/llm_train.jsonl`: TRL `messages` format — ≥1200 rows
- [ ] `data/splits/` with train/dev/test (80/10/10) for each format
- [ ] All files valid JSON (checked programmatically)

**Deliverables:**

```text
scripts/convert_to_training_format.py
data/embedding_train.jsonl
data/llm_train.jsonl
data/splits/embedding_train_split.jsonl
data/splits/embedding_dev_split.jsonl
data/splits/embedding_test_split.jsonl
data/splits/llm_train_split.jsonl
data/splits/llm_dev_split.jsonl
data/splits/llm_test_split.jsonl
```

**Code Skeleton — `scripts/convert_to_training_format.py`:**

```python
"""
Convert QA pairs to:
  1. Embedding triplets (anchor, positive, hard_negative) via mine_hard_negatives
  2. LLM SFT records in TRL conversational messages format
Creates train/dev/test splits (80/10/10).
"""
import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

INPUT = Path("data/qa_pairs_filtered.jsonl")
Path("data/splits").mkdir(parents=True, exist_ok=True)

SYSTEM_VI = (
    "Bạn là trợ lý nhân sự chuyên nghiệp. Hãy trả lời câu hỏi của nhân viên "
    "dựa trên thông tin từ sổ tay công ty được cung cấp. "
    "Chỉ sử dụng thông tin trong ngữ cảnh — không thêm thông tin bên ngoài."
)


def load_pairs() -> list:
    return [json.loads(l) for l in open(INPUT, encoding="utf-8")]


def make_embedding_triplets(pairs: list) -> list:
    """
    Use mine_hard_negatives from sentence-transformers (v5.5.1, already installed).
    anchor = Vietnamese question (with 'query: ' prefix for e5 model)
    positive = English handbook chunk (with 'passage: ' prefix)
    hard_negative = semantically similar but wrong chunk
    """
    print("[Embedding] Building dataset for mine_hard_negatives...")
    raw_dataset = Dataset.from_list([
        {
            "anchor": "query: " + p["question"],
            "positive": "passage: " + p["chunk_text"],
        }
        for p in pairs
    ])

    model = SentenceTransformer("intfloat/multilingual-e5-small")

    print("[Embedding] Mining hard negatives (semantic similarity)...")
    dataset_with_neg = mine_hard_negatives(
        dataset=raw_dataset,
        model=model,
        relative_margin=0.1,     # neg must be ≤90% as similar as positive
        num_negatives=1,
        sampling_strategy="top",
        output_format="triplet", # columns: anchor, positive, negative
        use_faiss=False,         # brute-force fine at <1000 chunks
        batch_size=64,
        verbose=True,
    )
    # Rename 'negative' → 'hard_negative' to match Phase 5 expectations
    dataset_with_neg = dataset_with_neg.rename_column("negative", "hard_negative")

    return dataset_with_neg.to_list()


def make_sft_records(pairs: list) -> list:
    """
    TRL SFTTrainer conversational messages format.
    English chunk goes in user message as context; Vietnamese answer from assistant.
    SFTTrainer consumes this natively — no formatting_func needed.
    Remove 'source' column before training (SFTTrainer passes all columns to model).
    """
    records = []
    for p in pairs:
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_VI},
                {
                    "role": "user",
                    "content": f"Ngữ cảnh:\n{p['chunk_text']}\n\nCâu hỏi: {p['question']}",
                },
                {"role": "assistant", "content": p["answer"]},
            ]
        })
    return records


def split_and_save(data: list, base_name: str):
    """Shuffle and split 80/10/10; save to data/splits/."""
    data = list(data)  # copy
    random.shuffle(data)
    n = len(data)
    cuts = (int(n * 0.8), int(n * 0.9))
    for name, chunk in zip(
        ["train", "dev", "test"],
        [data[:cuts[0]], data[cuts[0]:cuts[1]], data[cuts[1]:]],
    ):
        path = Path(f"data/splits/{base_name}_{name}_split.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(chunk)} → {path}")


def main():
    pairs = load_pairs()
    print(f"Loaded {len(pairs)} filtered QA pairs")

    # 1. Embedding triplets
    print("\n[Embedding] Creating triplets with hard negative mining...")
    triplets = make_embedding_triplets(pairs)
    with open("data/embedding_train.jsonl", "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  Saved {len(triplets)} triplets → data/embedding_train.jsonl")
    split_and_save(triplets, "embedding")

    # 2. SFT records (TRL messages format)
    print("\n[SFT] Creating messages-format records...")
    sft = make_sft_records(pairs)
    with open("data/llm_train.jsonl", "w", encoding="utf-8") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(sft)} SFT records → data/llm_train.jsonl")
    split_and_save(sft, "llm")

    print("\n✓ Done. Upload data/splits/ to H100 — Phase 5 + Phase 6 can now start.")


if __name__ == "__main__":
    main()
```

**Action Steps:**

1. Run conversion:
   ```powershell
   python scripts/convert_to_training_format.py
   ```
2. Verify file counts:
   ```powershell
   python -c "
   import json
   for f in ['data/splits/embedding_train_split.jsonl',
             'data/splits/embedding_dev_split.jsonl',
             'data/splits/embedding_test_split.jsonl',
             'data/splits/llm_train_split.jsonl',
             'data/splits/llm_dev_split.jsonl',
             'data/splits/llm_test_split.jsonl']:
       count = sum(1 for _ in open(f))
       print(f'{f}: {count} records')
   "
   ```
3. ✅ **Handoff to Phase 5 + Phase 6**: Upload `data/splits/` to H100 environment

**Verification Command:**

```powershell
python -c "
import json
triplets = [json.loads(l) for l in open('data/splits/embedding_train_split.jsonl', encoding='utf-8')]
sft = [json.loads(l) for l in open('data/splits/llm_train_split.jsonl', encoding='utf-8')]
assert len(triplets) >= 900, f'Too few triplets: {len(triplets)}'
assert all('anchor' in t and 'positive' in t and 'hard_negative' in t for t in triplets[:10])
assert all('instruction' in r and 'output' in r and 'system' in r for r in sft[:10])
print(f'✓ Embedding train: {len(triplets)} triplets')
print(f'✓ SFT train: {len(sft)} records')
print('✓ Format validation passed')
"
```

**Time Estimate:** ~45–60 min

---

### Task 6: Handbook → PDF Conversion (0.5 hours)

**Owner:** Both developers  
**Objective:** Convert at least 1 handbook to PDF for use as a demo document in the existing Streamlit RAG app

**Research note:** `weasyprint` has GTK dependency issues on Windows. Use `reportlab` as primary (no native deps). Pandoc is a good option if already installed.

**Acceptance Criteria:**

- [ ] `data/sample_handbook.pdf` created from 1 handbook (preferably `hshadab/handbook`)
- [ ] PDF has text layer (not image-only) — verifiable with PyMuPDF
- [ ] PDF has ≥10 pages and >5000 chars
- [ ] Existing RAG app can ingest this PDF without errors

**Deliverables:**

```text
scripts/convert_md_to_pdf.py
data/sample_handbook.pdf
```

**Code Skeleton — `scripts/convert_md_to_pdf.py`:**

```python
"""
Convert handbook Markdown files to a single PDF using reportlab or md2pdf.
"""
from pathlib import Path
import json
import subprocess
import sys

HANDBOOK = "hshadab"  # primary target
RAW_DIR = Path(f"data/raw/{HANDBOOK}")
OUTPUT_PDF = Path("data/sample_handbook.pdf")


def convert_with_pandoc():
    """Convert markdown to PDF using pandoc (preferred if installed)."""
    md_files = sorted(RAW_DIR.rglob("*.md"))
    combined_md = Path("data/temp_combined.md")

    with open(combined_md, "w", encoding="utf-8") as f:
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            f.write(f"\n\n# {md_file.stem}\n\n")
            f.write(content)
            f.write("\n\n---\n\n")

    result = subprocess.run(
        ["pandoc", str(combined_md), "-o", str(OUTPUT_PDF),
         "--pdf-engine=xelatex", "-V", "geometry:margin=1in"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"pandoc error: {result.stderr}")
        return False

    combined_md.unlink()
    return True


def convert_with_reportlab():
    """Fallback: use reportlab to create a simple text PDF."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        print("Install reportlab: pip install reportlab")
        sys.exit(1)

    md_files = sorted(RAW_DIR.rglob("*.md"))
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUTPUT_PDF), pagesize=letter)
    story = []

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        # Simple: add content as paragraphs (strip markdown)
        import re
        text = re.sub(r"[#*`\[\]()!]", "", content)
        for line in text.split("\n"):
            line = line.strip()
            if line:
                try:
                    story.append(Paragraph(line, styles["Normal"]))
                    story.append(Spacer(1, 6))
                except Exception:
                    pass  # skip lines with problematic chars

    doc.build(story)
    return True


def verify_pdf(path: Path) -> bool:
    import fitz
    doc = fitz.open(str(path))
    total_chars = sum(len(page.get_text()) for page in doc)
    pages = len(doc)
    print(f"  Pages: {pages}, Total chars: {total_chars}")
    return pages >= 5 and total_chars >= 3000


def main():
    print(f"Converting {HANDBOOK} handbook to PDF...")

    # Try pandoc first
    try:
        result = subprocess.run(["pandoc", "--version"],
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("Using pandoc...")
            if convert_with_pandoc():
                print(f"✓ PDF created via pandoc: {OUTPUT_PDF}")
                verify_pdf(OUTPUT_PDF)
                return
    except FileNotFoundError:
        pass

    # Fallback to reportlab
    print("pandoc not found, using reportlab...")
    convert_with_reportlab()
    print(f"✓ PDF created via reportlab: {OUTPUT_PDF}")
    verify_pdf(OUTPUT_PDF)


if __name__ == "__main__":
    main()
```

**Action Steps:**

1. Install reportlab (primary on Windows):
   ```powershell
   pip install reportlab
   ```
2. Run conversion:
   ```powershell
   python scripts/convert_md_to_pdf.py
   ```
3. Verify PDF readable:
   ```powershell
   python -c "import fitz; doc=fitz.open('data/sample_handbook.pdf'); print(f'Pages: {len(doc)}, Chars: {sum(len(p.get_text()) for p in doc)}')"
   ```

4. If char count < 5000: check that markdown files were read correctly in Task 1

**Time Estimate:** ~20–30 min

---

### Task 7: Dataset Validation (0.5 hours)

**Owner:** Both developers  
**Objective:** Programmatically validate all output files; confirm everything is ready for Phase 5 + 6

**Acceptance Criteria:**

- [ ] All JSONL files parse without error
- [ ] QA pair count ≥ 1500 total
- [ ] Train split ≥ 1200 rows (each format)
- [ ] No duplicate QA pairs (check by question text)
- [ ] Source distribution: all 3 handbooks represented in test set
- [ ] Vietnamese text confirmed in questions (detect diacritics or common Vietnamese words)

**Deliverables:**

```text
scripts/validate_dataset.py
data/validation_report.txt
```

**Code Skeleton — `scripts/validate_dataset.py`:**

```python
"""Validate all Phase 4 outputs before handing off to Phase 5 + 6."""
import json
from pathlib import Path

CHECKS = {
    "data/qa_pairs_filtered.jsonl": {"min_rows": 1500, "required_keys": ["question", "answer", "source"]},
    "data/embedding_train.jsonl": {"min_rows": 1500, "required_keys": ["anchor", "positive", "hard_negative"]},
    "data/llm_train.jsonl": {"min_rows": 1500, "required_keys": ["system", "instruction", "output"]},
    "data/splits/embedding_train_split.jsonl": {"min_rows": 900, "required_keys": ["anchor", "positive"]},
    "data/splits/embedding_test_split.jsonl": {"min_rows": 100, "required_keys": ["anchor", "positive"]},
    "data/splits/llm_train_split.jsonl": {"min_rows": 900, "required_keys": ["instruction", "output"]},
    "data/splits/llm_test_split.jsonl": {"min_rows": 100, "required_keys": ["instruction", "output"]},
}

VN_CHARS = set("àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")

results = []
all_passed = True

for filepath, rules in CHECKS.items():
    path = Path(filepath)
    if not path.exists():
        results.append(f"FAIL: {filepath} — file not found")
        all_passed = False
        continue

    try:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    except json.JSONDecodeError as e:
        results.append(f"FAIL: {filepath} — JSON parse error: {e}")
        all_passed = False
        continue

    # Row count check
    if len(rows) < rules["min_rows"]:
        results.append(f"FAIL: {filepath} — {len(rows)} rows (need ≥{rules['min_rows']})")
        all_passed = False
        continue

    # Required keys check
    missing = [k for k in rules["required_keys"] if k not in rows[0]]
    if missing:
        results.append(f"FAIL: {filepath} — missing keys: {missing}")
        all_passed = False
        continue

    results.append(f"PASS: {filepath} — {len(rows)} rows, keys OK")

# Vietnamese text check on QA pairs
qa_path = Path("data/qa_pairs_filtered.jsonl")
if qa_path.exists():
    pairs = [json.loads(l) for l in open(qa_path, encoding="utf-8")]
    vi_count = sum(1 for p in pairs if any(c in VN_CHARS for c in p.get("question", "")))
    pct = vi_count / len(pairs) * 100 if pairs else 0
    if pct < 80:
        results.append(f"WARN: Only {pct:.0f}% of questions contain Vietnamese diacritics (expected ≥80%)")
    else:
        results.append(f"PASS: Vietnamese check — {pct:.0f}% questions have diacritics")

    # Source distribution
    sources = {}
    for p in pairs:
        sources[p["source"]] = sources.get(p["source"], 0) + 1
    results.append(f"INFO: Source distribution: {sources}")

# Write report
report = "\n".join(results)
print(report)
Path("data/validation_report.txt").write_text(report)

if all_passed:
    print("\n✓ ALL CHECKS PASSED — Phase 5 + Phase 6 can start")
else:
    print("\n✗ SOME CHECKS FAILED — fix before proceeding")
```

**Action Steps:**

1. Run validation:
   ```powershell
   python scripts/validate_dataset.py
   ```
2. Fix any failures before proceeding
3. Keep `data/validation_report.txt` as evidence of Phase 4 completion

**Time Estimate:** ~20–30 min

---

### Task 8: Notebook Finalization & Commit (1 hour)

**Owner:** Both developers  
**Objective:** Transfer all scripts into `notebooks/FINETUNING_A0-DataPreparation.ipynb`; create final atomic commit

**Acceptance Criteria:**

- [ ] Notebook runs cell-by-cell from top to bottom without errors
- [ ] Each cell has a 1-line comment explaining what it does
- [ ] Notebook reads clean data (no hardcoded local paths)
- [ ] Notebook outputs match `data/validation_report.txt`
- [ ] All Phase 4 files committed atomically

**Deliverables:**

```text
notebooks/FINETUNING_A0-DataPreparation.ipynb  (updated, runnable)
scripts/ingest_handbooks.py                    (committed)
scripts/generate_qa.py                         (committed)
scripts/filter_qa.py                           (committed)
scripts/convert_to_training_format.py          (committed)
scripts/convert_md_to_pdf.py                   (committed)
scripts/validate_dataset.py                    (committed)
data/qa_pairs_filtered.jsonl                   (committed — or .gitignore if >100MB)
data/splits/                                   (committed)
data/validation_report.txt                     (committed)
data/sample_handbook.pdf                       (committed)
```

**Note on data files:** If `qa_pairs.jsonl` is large (>50MB), add to `.gitignore` and document how to regenerate. Always commit the validation report and split files.

**Action Steps:**

1. Open `notebooks/FINETUNING_A0-DataPreparation.ipynb`
2. Structure cells:
   - Cell 1: Setup imports + config
   - Cell 2: Handbook ingestion (call `ingest_handbooks.py` logic)
   - Cell 3: QA generation (small test on 5 chunks)
   - Cell 4: Filter + quality review
   - Cell 5: Training format conversion
   - Cell 6: Validation report
3. Run all cells, fix any errors
4. Commit everything:
   ```powershell
   git add scripts/ data/splits/ data/validation_report.txt data/sample_handbook.pdf notebooks/FINETUNING_A0-DataPreparation.ipynb
   git commit -m "feat(phase4): data synthesis pipeline — 1500+ Vietnamese HR QA pairs from 3 GitHub handbooks"
   ```

**Time Estimate:** ~45–60 min

---

## Wave Planning & Execution Order

All tasks sequential (LLM generation must complete before format conversion):

```text
Wave 1: Task 1 (Ingestion) — 1h
   ↓
Wave 2: Task 2 (Quality Check) — 0.5h
   ↓
Wave 3: Task 3 (QA Generation) — 0.5h setup + 1-3h unattended LLM calls
   ↓
Wave 4: Task 4 (Filter) — 0.5h
   ↓
Wave 5: Task 5 (Format Conversion) — 1h  ← UNBLOCKS Phase 5 + Phase 6
   ↓ (parallel with Phase 5 + 6 starting on H100)
Wave 6: Task 6 (PDF) — 0.5h
   ↓
Wave 7: Task 7 (Validation) — 0.5h
   ↓
Wave 8: Task 8 (Notebook + Commit) — 1h
```

**Total Developer Time:** ~6 hours  
**Total Clock Time:** ~8–10 hours (Task 3 runs unattended)

---

## Gate Checks

### Gate 1: After Task 1 (Handbook Ingestion)

```powershell
python -c "
data = open('data/raw_chunks.jsonl').readlines()
assert len(data) >= 400, f'Too few chunks: {len(data)}'
print(f'✓ Gate 1 passed: {len(data)} chunks')
"
```

### Gate 2: After Task 3 (QA Generation)

```powershell
python -c "
import json
pairs = [json.loads(l) for l in open('data/qa_pairs.jsonl', encoding='utf-8')]
assert len(pairs) >= 1500, f'Too few pairs: {len(pairs)}'
print(f'✓ Gate 2 passed: {len(pairs)} QA pairs')
"
```

### Gate 3: After Task 5 (Format Conversion) — Phase 5 + 6 Handoff Gate

```powershell
python scripts/validate_dataset.py
```

Pass criteria: All `PASS` lines, no `FAIL` lines.

### Gate 4: After Task 8 (Notebook + Commit)

```powershell
jupyter nbconvert --to notebook --execute notebooks/FINETUNING_A0-DataPreparation.ipynb --output notebooks/FINETUNING_A0-executed.ipynb
```

Pass criteria: No exceptions; output notebook created.

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| GitHub repos unavailable | High | Use `--depth=1` clone for speed; cache locally |
| LLM API rate limit | Medium | Task 3 script auto-retries with backoff; checkpoint saves progress |
| Poor Vietnamese quality | High | Test on 10 chunks before full run; adjust prompt if needed |
| QA count < 1500 | Medium | Lower chunk size to 400 to create more chunks; re-run Task 3 |
| PDF conversion fails | Low | reportlab fallback covers 99% of cases |
| Anthropic API key missing | High | Set env var before Task 3; verify with test call first |

---

## Commit Strategy

**Single atomic commit after all 8 tasks complete:**

```powershell
git add scripts/ data/splits/ data/validation_report.txt data/sample_handbook.pdf notebooks/FINETUNING_A0-DataPreparation.ipynb .planning/phases/04-data-synthesis/
git commit -m "feat(phase4): data synthesis pipeline — Vietnamese HR QA dataset from 3 GitHub handbooks"
```

**Data file handling:**

If `data/qa_pairs.jsonl` > 50MB, add to `.gitignore`:

```gitignore
# Large generated data files (regenerate with scripts/generate_qa.py)
data/qa_pairs.jsonl
data/raw/
data/qa_checkpoint.json
```

Always commit: `data/splits/`, `data/validation_report.txt`, `data/sample_handbook.pdf`

---

## Execution Checklist

### Pre-Execution

- [ ] Python 3.10+ installed
- [ ] `git` CLI installed (for cloning handbooks)
- [ ] `ANTHROPIC_API_KEY` environment variable set
- [ ] `pip install anthropic fitz reportlab` verified
- [ ] H100 environment ready to receive `data/splits/` after Task 5

### During Execution

- [ ] **Task 1 (1h)**: Handbook ingestion
  - [ ] 3 repos cloned to `data/raw/`
  - [ ] `data/raw_chunks.jsonl` created
  - [ ] ≥400 chunks confirmed
  - [ ] ✓ Gate 1 passed

- [ ] **Task 2 (0.5h)**: Quality check
  - [ ] 20 random chunks reviewed manually
  - [ ] No garbage chunks found
  - [ ] Chunk count documented

- [ ] **Task 3 (2–4h)**: QA generation
  - [ ] Test run on 10 chunks approved
  - [ ] Full run started (unattended)
  - [ ] ≥1500 pairs confirmed in output
  - [ ] ✓ Gate 2 passed

- [ ] **Task 4 (0.5h)**: Quality filter
  - [ ] `filter_qa.py` run; count still ≥1500
  - [ ] 20 pairs manually reviewed
  - [ ] Review notes written

- [ ] **Task 5 (1h)**: Training format conversion
  - [ ] Embedding triplets created and split
  - [ ] SFT JSONL created and split
  - [ ] `validate_dataset.py` passes all checks
  - [ ] ✓ Gate 3 passed — HANDOFF TO PHASE 5 + 6

- [ ] **Task 6 (0.5h)**: PDF conversion
  - [ ] `data/sample_handbook.pdf` created
  - [ ] PDF verifiable with PyMuPDF

- [ ] **Task 7 (0.5h)**: Dataset validation
  - [ ] `validation_report.txt` all PASS
  - [ ] Vietnamese quality confirmed

- [ ] **Task 8 (1h)**: Notebook + commit
  - [ ] Notebook runs end-to-end
  - [ ] Atomic commit created
  - [ ] ✓ Gate 4 passed

### Post-Execution

- [ ] All Gate checks passed
- [ ] `data/splits/` accessible from H100 environment
- [ ] Phase 5 and Phase 6 can start in parallel
- [ ] Phase 4 exit criteria met (all 7 success criteria)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `git clone` fails (no network) | Use `pip install requests` and download zip via GitHub API |
| Anthropic API returns 429 | Task 3 script has exponential backoff; just re-run |
| JSON parse error from LLM | Script skips and logs; regenerate those chunks manually |
| Chunk count < 600 | Lower `CHUNK_SIZE` to 400 in `ingest_handbooks.py` and re-run Task 1 |
| pandoc PDF fails | Script auto-falls back to reportlab |
| Notebook kernel crashes | Check import errors; ensure all deps installed in notebook kernel |

---

## Success Summary

**Phase 4 is complete when:**

1. ≥1500 Vietnamese QA pairs generated and filtered
2. Train/dev/test splits created in `data/splits/`
3. Both `embedding_train.jsonl` and `llm_train.jsonl` pass validation
4. `data/sample_handbook.pdf` created and verified
5. `notebooks/FINETUNING_A0-DataPreparation.ipynb` runs end-to-end
6. All 4 gate checks passed
7. Atomic git commit created

**Handoff to Phase 5 + 6:** After Gate 3 (Task 5), upload `data/splits/` to H100 and start both embedding and LLM fine-tuning in parallel.

---

**PLAN.md created:** 2026-06-01  
**Status:** Ready for execution  
**Owner:** Both developers  
**Next:** Execute Task 1 → Task 8 sequentially; kick off Phase 5 + 6 after Task 5
