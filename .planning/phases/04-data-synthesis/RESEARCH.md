# Phase 4: Data Synthesis Pipeline — Research

**Researched:** 2026-06-01
**Domain:** Synthetic Vietnamese QA generation, hard negative mining, vLLM+ngrok serving, embedding/LLM training data formats
**Confidence:** MEDIUM-HIGH (core findings verified via official docs and libraries; vLLM/ngrok specifics verified against official sources)

---

## Summary

Phase 4 synthesizes a Vietnamese HR Q&A dataset from three public English GitHub handbooks using a teacher LLM served on the team's own H100 via vLLM + ngrok. No paid external API is required. The output feeds Phase 5 (embedding fine-tuning) and Phase 6 (QLoRA LLM fine-tuning).

The key architectural decision — English source text, Vietnamese Q&A output — is correct and well-supported by Qwen2.5-72B's documented Vietnamese capability. Direct generation (not chain-of-thought) is the recommended prompt strategy for grounded HR Q&A pairs. Hard negative mining using `sentence_transformers.util.mine_hard_negatives` is the cleanest path for the ~600-900 chunk corpus. TRL's `SFTTrainer` consumes the conversational `messages` format natively.

**Primary recommendation:** Use a single-pass direct generation prompt (English chunk in, Vietnamese Q+A out), structured JSON output, checkpoint-based generation loop, and `mine_hard_negatives` with `output_format="triplet"` for embedding data.

**Critical note on teacher LLM:** The existing CONTEXT.md names Claude Haiku as teacher. The user's request replaces that with self-hosted Qwen2.5-72B via vLLM + ngrok. This research covers the vLLM+ngrok path. The PLAN.md author must reconcile this decision — the two approaches have different prompt interfaces and cost profiles.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
| Decision | Rationale |
|----------|-----------|
| English source → Vietnamese Q&A | Source handbooks are English; we want Vietnamese RAG output |
| Hard negatives from same handbook | Cross-source negatives are harder than same-source; better training signal |
| Checkpoint-based generation | Task 3 runs 1–3 hours; checkpoints allow resume on failure |
| 80/10/10 split | Standard; test set reserved for evaluation after Phase 5+6 |

### Output Formats (Locked)
**Embedding Training (Phase 5 input):**
```json
{"anchor": "Nhân viên được nghỉ bao nhiêu ngày?", "positive": "<handbook chunk>", "hard_negative": "<different chunk>"}
```
**LLM SFT Training (Phase 6 input):**
```json
{"system": "Bạn là trợ lý nhân sự...", "instruction": "<Vietnamese question>", "input": "<handbook context>", "output": "<Vietnamese answer>"}
```

### Claude's Discretion
- Exact prompt template wording
- Number of QA pairs per chunk (2 or 3)
- Hard negative mining strategy (BM25 vs. embedding-based)
- Chunking parameters (target ~500 chars per chunk)

### Deferred Ideas (OUT OF SCOPE)
- Translation of handbook text (only Q&A output is Vietnamese)
- Manual annotation or labeling
- Web scraping beyond the 3 specified GitHub repos
- Vietnamese handbooks as source
- Model training (Phase 5 + 6)

### Teacher LLM — Conflict Between CONTEXT.md and User Request
CONTEXT.md specifies **Claude Haiku** as teacher. The user's research request specifies **Qwen2.5-72B via vLLM + ngrok on H100**. This research covers the vLLM+ngrok path. PLAN.md author must finalize this choice before execution.
</user_constraints>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Handbook ingestion (git clone + markdown parse) | CPU / local script | — | Pure file I/O, no GPU needed |
| Text cleaning and chunking | CPU / local script | — | String processing; ~600-900 chunks fits in RAM |
| Vietnamese QA generation | Remote GPU (H100 via vLLM+ngrok) | Anthropic API (fallback) | LLM inference requires GPU for throughput |
| Hard negative mining | CPU / local script | — | sentence-transformers mine_hard_negatives runs on CPU for 600-900 chunks |
| Training format conversion | CPU / local script | — | Pure data transformation |
| Markdown → PDF | CPU / local script | — | reportlab/weasyprint; no GPU |
| Dataset validation and quality checks | CPU / local script | — | count, JSON validity, spot review |

---

## Section 1: Synthetic QA Generation Best Practices

### 1.1 Prompt Strategy: Direct Generation vs. Chain-of-Thought

**Finding:** For HR policy content (factual, document-grounded, structured), direct generation outperforms chain-of-thought for QA pair synthesis. CoT adds latency and token cost without measurable quality gain when the source text is already coherent prose. [VERIFIED: RAG synthetic data generation literature, NVIDIA Technical Blog]

**Self-instruct** (generate questions independently, then generate answers) is useful for diversity but risks hallucination because answers are generated without grounding to the specific chunk. **Direct generation** (chunk-in → Q+A-out in one call) is preferred because:
- The answer is grounded to the provided chunk by instruction
- Lower token cost (single pass)
- Easier to enforce "answer only from the provided text" constraint

**Recommended approach:** Direct generation with explicit grounding instruction.

### 1.2 Proven Prompt Template

```python
SYSTEM_PROMPT = """Bạn là chuyên gia nhân sự. Hãy đọc đoạn văn bản từ sổ tay nhân viên (bằng tiếng Anh) và tạo ra {n_pairs} cặp hỏi-đáp bằng tiếng Việt.

Yêu cầu:
1. Câu hỏi phải là câu hỏi thực tế mà nhân viên có thể đặt ra.
2. Câu trả lời phải DỰA HOÀN TOÀN vào đoạn văn bản được cung cấp — không thêm thông tin bên ngoài.
3. Câu trả lời phải rõ ràng, ngắn gọn, bằng tiếng Việt chuẩn.
4. Đầu ra phải là JSON hợp lệ theo định dạng sau, không thêm văn bản khác.

Định dạng JSON:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""

USER_PROMPT = """Đoạn văn bản:
{chunk_text}

Hãy tạo {n_pairs} cặp hỏi-đáp từ đoạn văn bản trên."""
```

**Why this works:**
- Vietnamese system prompt primes the model for Vietnamese output [ASSUMED: based on LLM prompting practice]
- Explicit "dựa hoàn toàn" (based entirely) instruction reduces hallucination
- JSON-only output format enables programmatic parsing
- `n_pairs=2` for chunks <300 chars, `n_pairs=3` for chunks >=300 chars

### 1.3 Hallucination Prevention

| Technique | Implementation |
|-----------|---------------|
| Source constraint in prompt | "Câu trả lời phải DỰA HOÀN TOÀN vào đoạn văn bản" |
| JSON-only output format | Parse failure = discard the pair |
| Answer length check | Discard if answer > 3× the chunk length |
| Answer substring overlap | Soft check: ≥1 named entity or key phrase from chunk in answer |
| Temperature ≤ 0.3 | Lower temperature = more grounded, less creative |

**Recommended:** `temperature=0.2`, `top_p=0.9`, `max_tokens=512` per call.

### 1.4 Existing Tools

| Tool | Verdict for This Project |
|------|--------------------------|
| **Ragas** `TestsetGenerator` | Designed for evaluation, not training data. Overkill for this use case. Skip. |
| **LlamaIndex** `RagDatasetGenerator` | GPT-4 only in practice; doesn't support custom vLLM endpoint well. Skip. |
| **distilabel** `ClientvLLM` | Good fit — connects to vLLM via OpenAI-compatible API. But adds dependency weight. Use only if generation loop needs pipeline features. |
| **Custom loop (recommended)** | Simple `openai` SDK loop with checkpoint file. Minimal dependencies. Full control. |

**Recommended approach:** Custom Python generation loop using `openai` SDK pointing at vLLM. ~80 lines of code. Distilabel is a valid alternative if the team wants pipeline retries and monitoring.

### 1.5 Pairs per Chunk

| Chunk length | Recommended pairs | Rationale |
|---|---|---|
| < 200 chars | 1 | Too short for 2 distinct questions |
| 200-500 chars | 2 | Standard; good diversity |
| > 500 chars | 3 | Rich enough content for 3 angles |

Expected output from ~750 chunks avg: 750 × 2.3 avg ≈ **1,725 raw pairs** before filtering. [ASSUMED: estimate based on handbook size range from REQUIREMENTS_M2.md]

---

## Section 2: Vietnamese NLP Specifics

### 2.1 Existing Vietnamese HR/Legal QA Datasets

| Dataset | HuggingFace ID | Domain | Usability |
|---------|---------------|--------|-----------|
| Vietnamese Legal QA | `nqdhocai/vietnamese-legal-qa` | Law/Ethics MCQ | Style mismatch (MCQ not HR Q&A) |
| Vietnamese Legal Chat | `luanngo/Vietnamese-Legal-Chat-Dataset` | Legal reasoning | VLSP 2025 challenge data; NLI/MCQ format |
| ViBidLQA | (custom, bidding law) | Bidding law | Too domain-specific |
| ViLQA | [github.com/ntphuc149/ViLQA](https://github.com/ntphuc149/ViLQA) | Legal MRC | MRC format, not HR |

**Verdict:** No existing Vietnamese HR Q&A dataset in the right format exists. [VERIFIED: HuggingFace dataset search] The legal datasets use MCQ/NLI formats incompatible with the embedding triplet and SFT JSONL targets. Supplement only if the legal Q&A can be reformatted — not recommended for this phase.

**Recommendation:** Do not supplement with existing datasets. Generate all 1500+ pairs synthetically from the 3 handbooks.

### 2.2 Qwen2.5-72B Vietnamese Diacritics Quality

**Finding:** Qwen2.5 explicitly supports Vietnamese in its multilingual capability set (29+ languages). The Qwen team validated Vietnamese via translated IFEval benchmarks with human post-editing. [CITED: qwenlm.github.io/blog/qwen2.5/]

**Diacritics reliability:** Qwen2.5-72B produces correct Vietnamese diacritics reliably for formal/instructional text when prompted in Vietnamese. At 72B scale, diacritics errors are rare in structured output (JSON format further constrains the output). [ASSUMED: based on scale-quality correlation; no Vietnamese diacritics error rate benchmark found]

**Risk mitigation:** Run a 5-pair pilot test before launching the full generation loop. Check output for common diacritics errors (e.g., "nghi" vs. "nghỉ", "nhan vien" vs. "nhân viên"). If errors appear, add explicit instruction: "Sử dụng đầy đủ dấu thanh tiếng Việt (ă, â, đ, ê, ô, ơ, ư và các dấu hỏi, ngã, nặng, sắc, huyền)."

### 2.3 Vietnamese Questions + Vietnamese Answers (Not English)

**Decision rationale (confirmed correct):** The RAG system serves Vietnamese-speaking users asking Vietnamese questions and expecting Vietnamese answers. Both Q and A must be Vietnamese. The source chunk can remain English (it provides factual grounding for the answer — the model translates and paraphrases while generating the answer).

**Do NOT generate:** Vietnamese questions + English answers. The LLM SFT training (Phase 6) would teach the model to answer in English, which breaks the system's Vietnamese output requirement.

**Embedding triplet note:** The `positive` in the embedding triplet is the English handbook chunk (the actual retrieval target). The `anchor` is the Vietnamese question. This cross-lingual pairing is intentional — it trains the embedding model to match Vietnamese queries to English passages. [VERIFIED: matches Phase 5 requirements in REQUIREMENTS_M2.md]

---

## Section 3: Hard Negative Mining

### 3.1 Approach Comparison for 600-900 Chunk Corpus

| Approach | Quality | Speed | Ease | Verdict for This Scale |
|----------|---------|-------|------|------------------------|
| Random negatives | Low | Fast | Trivial | Too easy; model won't learn hard distinctions |
| BM25-based (rank_bm25) | Medium | Fast | Easy | Good baseline; lexical overlap = plausible negatives |
| Embedding-based (`mine_hard_negatives`) | High | Medium | Easy (one function call) | Best quality; semantic similarity finds true hard negatives |
| Cross-encoder reranking | Highest | Slow | Complex | Overkill for <1000 chunks |

**For 600-900 chunks:** Embedding-based mining via `mine_hard_negatives` is the clear winner. The corpus fits entirely in RAM and the FAISS index build is fast at this scale (~5-10 seconds). BM25 is a viable fallback if sentence-transformers is not available, but it is already installed in this environment. [VERIFIED: `sentence-transformers==5.5.1` confirmed installed locally]

### 3.2 `mine_hard_negatives` — Complete API

**Module:** `sentence_transformers.util.mine_hard_negatives` [VERIFIED: importable in project env]

**Full signature:**
```python
mine_hard_negatives(
    dataset: Dataset,           # HuggingFace Dataset with anchor + positive columns
    model: SentenceTransformer, # Model used to compute similarities
    anchor_column_name: str | None = None,
    positive_column_name: str | None = None,
    corpus: list[str] | None = None,      # If None, uses positives as corpus
    range_min: int = 0,                   # Skip top-N most similar (avoids false negatives)
    range_max: int | None = None,
    max_score: float | None = None,
    min_score: float | None = None,
    relative_margin: float | None = None, # Neg must be ≤ (1-margin) × positive similarity
    num_negatives: int = 3,               # Negatives per anchor
    sampling_strategy: Literal['random', 'top'] = 'top',
    output_format: Literal['triplet', 'n-tuple', 'labeled-pair', 'labeled-list'] = 'triplet',
    use_faiss: bool = False,              # Use FAISS for large corpora (>10k)
    batch_size: int = 32,
    verbose: bool = True
) -> Dataset
```

**Recommended call for this project:**
```python
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

# Build dataset from qa_pairs.jsonl
# anchor = Vietnamese question, positive = English chunk it was generated from
dataset = Dataset.from_list([
    {"anchor": pair["question"], "positive": pair["chunk"]}
    for pair in qa_pairs
])

# Use the same model being fine-tuned as the mining model
# (or multilingual-e5-small if training hasn't started yet)
model = SentenceTransformer("intfloat/multilingual-e5-small")

dataset_with_negatives = mine_hard_negatives(
    dataset=dataset,
    model=model,
    relative_margin=0.1,    # Negatives must be ≤ 90% as similar as positives
    num_negatives=1,        # 1 hard negative per pair (matches target format)
    sampling_strategy="top",
    output_format="triplet",  # Produces (anchor, positive, negative) columns
    use_faiss=False,          # At 900 chunks, brute-force is fine
    batch_size=64,
    verbose=True,
)
# Result columns: "anchor", "positive", "negative"
# Rename "negative" → "hard_negative" to match Phase 5 format
dataset_with_negatives = dataset_with_negatives.rename_column("negative", "hard_negative")
```

**Output format matches CONTEXT.md target:** `{"anchor": "...", "positive": "...", "hard_negative": "..."}` [VERIFIED: column names match after rename]

### 3.3 BM25 Fallback (if embedding-based mining fails)

Install: `pip install rank-bm25`

```python
from rank_bm25 import BM25Okapi

# Tokenize chunks
tokenized_corpus = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)

def get_bm25_hard_negative(query_chunk: str, positive_chunk: str, all_chunks: list[str]) -> str:
    """Return top BM25 result that is not the positive chunk."""
    tokenized_query = query_chunk.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for idx in ranked_indices:
        if all_chunks[idx] != positive_chunk:
            return all_chunks[idx]
    return all_chunks[ranked_indices[-1]]  # last resort
```

**Note:** `rank-bm25` is NOT installed in the current environment. Must be added to requirements if used as fallback.

### 3.4 Cross-handbook vs. Same-handbook Negatives

CONTEXT.md specifies "hard negatives from same handbook." This is correct for embedding training: same-handbook negatives share topic domain (employee policies) but differ in specific content, creating the semantically challenging contrast the model needs to learn. Cross-handbook negatives would be too easy (different company = more obvious contrast).

---

## Section 4: vLLM + ngrok Setup

### 4.1 Memory Analysis: Qwen2.5-72B on Single H100 96GB

| Precision | Model weight size | Fits single H100 96GB |
|-----------|-----------------|----------------------|
| BF16 | ~146 GB | No — exceeds 96GB |
| FP8 | ~73 GB | Yes — with headroom for KV cache |
| AWQ INT4 | ~36 GB | Yes — comfortable |

**Conclusion:** Full BF16/FP16 Qwen2.5-72B does NOT fit on a single H100 96GB. [VERIFIED: Qwen docs state "impossible to serve on a single GPU" for 72B at full precision]

**Two viable paths for single H100 96GB:**

**Option A (Recommended): FP8 quantized checkpoint**
```bash
# Use the community FP8 checkpoint (fits single H100 96GB)
# Several FP8 variants exist from RedHatAI and others
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --dtype fp8 \
  --quantization fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000
```

**Option B: Tensor parallel across 2 GPUs (if 2× H100 available)**
```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000
```

**Option C: Use a smaller model (if H100 is single GPU BF16 only)**
```bash
# Qwen2.5-32B-Instruct fits single H100 96GB in BF16 (~64GB)
vllm serve Qwen/Qwen2.5-32B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000
```

**The PLAN.md author must confirm which GPU configuration is available on the H100 machine before finalizing the launch command.**

### 4.2 ngrok Tunnel Command

```bash
# Basic tunnel (free tier — no session timeout, 1GB/month transfer cap)
ngrok http 8000

# With custom domain (if registered)
ngrok http 8000 --url https://your-domain.ngrok-free.app
```

**Free tier limits relevant to this project:** [VERIFIED: ngrok.com/docs/pricing-limits/free-plan-limits]
- No endpoint timeout (long LLM requests will not be killed by ngrok)
- 1 GB/month data transfer
- 20,000 requests/month
- 3 concurrent online endpoints

**Data transfer estimate:** 1500 QA pairs × ~2KB per request/response ≈ **~3 MB total**. Well within the 1GB monthly cap.

**Potential issue:** ngrok free tier assigns a random subdomain on each `ngrok http` restart. The Python client must be updated with the new URL after each restart. Solution: note the URL from `ngrok` output, or use a paid static domain.

### 4.3 Python OpenAI SDK Client

```python
import httpx
from openai import OpenAI

# Extend timeout for long LLM inference (72B model can take 30-60s per request)
long_timeout = httpx.Timeout(
    connect=10.0,
    read=300.0,    # 5 minutes read timeout
    write=30.0,
    pool=10.0,
)

client = OpenAI(
    base_url="https://xxxx-xxxx.ngrok-free.app/v1",  # Replace with actual ngrok URL
    api_key="fake",                                     # vLLM does not validate API key
    http_client=httpx.Client(timeout=long_timeout),
)

def generate_qa_pairs(chunk: str, n_pairs: int = 2) -> list[dict]:
    """Generate n_pairs Vietnamese QA pairs from an English handbook chunk."""
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",  # Must match --served-model-name or model path
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(n_pairs=n_pairs)},
            {"role": "user", "content": USER_PROMPT.format(chunk_text=chunk, n_pairs=n_pairs)},
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    try:
        pairs = json.loads(raw)
        return [p for p in pairs if "question" in p and "answer" in p]
    except json.JSONDecodeError:
        return []  # Discard malformed output
```

**Timeout note:** [VERIFIED: vllm GitHub issue #14792] Default OpenAI SDK timeout is 10 minutes. For 72B models generating 512 tokens, single requests can take 30-90 seconds. Setting `read=300.0` provides enough headroom.

### 4.4 Known Issues and Mitigations

| Issue | Root Cause | Mitigation |
|-------|-----------|------------|
| Random ngrok URL on restart | Free tier assigns ephemeral domain | Note URL from ngrok output; update `base_url` manually |
| vLLM OOM at 72B BF16 | 146GB model > 96GB VRAM | Use FP8 or tensor-parallel-size 2 |
| Slow throughput at 72B | Large model | Set `--max-model-len 4096` to reduce KV cache; use `temperature=0.2` to reduce sampling overhead |
| JSON parse failure in output | Model outputs markdown fences | Strip ` ```json ` / ` ``` ` before parsing |
| vLLM context window limit | Default `max-model-len` can be huge | Set `--max-model-len 4096` explicitly; chunks are ~500 chars (~150 tokens) so this is safe |
| `distilabel.ClientvLLM` timeout=120 | Default 2-min timeout | Pass `timeout=300` to `ClientvLLM` constructor if using distilabel |

---

## Section 5: Data Format Best Practices

### 5.1 Embedding Training Format (Phase 5)

**sentence-transformers v5.5.1** (installed) uses HuggingFace `Dataset` objects, not `InputExample`. [VERIFIED: sentence-transformers==5.5.1 confirmed installed; API verified via sbert.net docs]

**Column convention:** Column names are irrelevant — only column order matters.

```python
# Correct format for TripletLoss or MultipleNegativesRankingLoss with hard negatives
from datasets import Dataset

embedding_dataset = Dataset.from_list([
    {
        "anchor": row["anchor"],       # Vietnamese question
        "positive": row["positive"],   # English chunk (retrieval target)
        "hard_negative": row["hard_negative"],  # Different English chunk
    }
    for row in triplets
])

# For MultipleNegativesRankingLoss without explicit negatives (uses in-batch negatives):
# Only 2 columns needed: anchor + positive
# For TripletLoss: 3 columns (anchor, positive, negative)
```

**JSONL file format for storage:**
```jsonl
{"anchor": "Nhân viên được nghỉ bao nhiêu ngày phép?", "positive": "Employees receive 15 days of paid vacation...", "hard_negative": "The company provides health insurance covering..."}
```

**Save/load:**
```python
dataset.to_json("data/embedding_train.jsonl", orient="records", lines=True)
# Load: Dataset.from_json("data/embedding_train.jsonl")
```

### 5.2 LLM SFT Training Format (Phase 6)

**TRL SFTTrainer** (not installed locally — runs on H100) natively supports three formats. [VERIFIED: huggingface.co/docs/trl/v0.19.1/sft_trainer]

**Best format for this project: Conversational (`messages` column)**

```jsonl
{
  "messages": [
    {"role": "system", "content": "Bạn là trợ lý nhân sự chuyên nghiệp. Hãy trả lời câu hỏi của nhân viên dựa trên thông tin từ sổ tay công ty được cung cấp."},
    {"role": "user", "content": "Ngữ cảnh:\nEmployees receive 15 days of paid vacation per year...\n\nCâu hỏi: Nhân viên được nghỉ bao nhiêu ngày phép mỗi năm?"},
    {"role": "assistant", "content": "Theo quy định của công ty, nhân viên được nghỉ 15 ngày phép có lương mỗi năm."}
  ]
}
```

**Why conversational format over Alpaca:** The `messages` format is directly consumed by `SFTTrainer` without a `formatting_func`, is chat-template aware, and matches the Qwen2.5-3B-Instruct training format (which uses ChatML). Alpaca requires a formatting function and doesn't handle system prompts cleanly.

**Should the SFT dataset include the handbook chunk as context?** YES. [ASSUMED: best practice for RAG-specific SFT; supported by multiple SFT-for-RAG sources]

Including the chunk as context in the `user` message teaches the LLM to:
1. Read the provided context (the retrieved chunk in the RAG pipeline)
2. Answer grounded in that context (not from parametric memory)
3. Answer in Vietnamese regardless of chunk language

**JSONL field mapping:**

| CONTEXT.md field | TRL `messages` mapping |
|------------------|----------------------|
| `system` | `messages[0].content` (role: system) |
| `instruction` (Vietnamese question) | Part of `messages[1].content` (role: user) — after "Câu hỏi:" |
| `input` (English chunk context) | Part of `messages[1].content` — before "Câu hỏi:", under "Ngữ cảnh:" |
| `output` (Vietnamese answer) | `messages[2].content` (role: assistant) |

**Conversion script:**
```python
def to_sft_format(pair: dict) -> dict:
    """Convert qa_pair dict to TRL SFTTrainer messages format."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "Bạn là trợ lý nhân sự chuyên nghiệp. Hãy trả lời câu hỏi của nhân viên dựa trên thông tin từ sổ tay công ty được cung cấp. Chỉ sử dụng thông tin trong ngữ cảnh — không thêm thông tin bên ngoài.",
            },
            {
                "role": "user",
                "content": f"Ngữ cảnh:\n{pair['chunk']}\n\nCâu hỏi: {pair['question']}",
            },
            {
                "role": "assistant",
                "content": pair["answer"],
            },
        ]
    }
```

### 5.3 Alpaca vs. ShareGPT vs. Conversational — Final Verdict

| Format | TRL native | Chat-template aware | System prompt | Multi-turn | Verdict |
|--------|-----------|--------------------|--------------|-----------| --------|
| Alpaca (`instruction`, `input`, `output`) | Via formatting_func | No | Awkward | No | Avoid |
| ShareGPT (`conversations` key, `from`/`value`) | Via formatting_func | Partial | Yes | Yes | Avoid for simplicity |
| **Conversational (`messages`, role/content)** | **Native** | **Yes** | **Yes** | **Yes** | **Use this** |

### 5.4 multilingual-e5-small Prefix Requirement

**Critical:** `intfloat/multilingual-e5-small` requires query-time prefixes. [CITED: arxiv.org/pdf/2402.05672]

- Queries: prefix `"query: "` → `"query: Nhân viên được nghỉ bao nhiêu ngày?"` 
- Passages: prefix `"passage: "` → `"passage: Employees receive 15 days..."`

When using this model via `SentenceTransformer`, set prompts:
```python
model = SentenceTransformer("intfloat/multilingual-e5-small")
query_embedding = model.encode("query: " + question)
passage_embedding = model.encode("passage: " + chunk)
```

During training with `mine_hard_negatives`, pass `query_prompt="query: "` and `corpus_prompt="passage: "` if anchors are questions and positives are passages.

---

## Section 6: Environment Availability

| Dependency | Required By | Available (local) | Version | Notes |
|------------|------------|-------------------|---------|-------|
| `sentence-transformers` | Hard negative mining, embedding training | Yes | 5.5.1 | `mine_hard_negatives` importable |
| `datasets` | Dataset I/O | Yes (via HF ecosystem) | — | Verify: `pip show datasets` |
| `openai` (Python SDK) | vLLM client | **No** | — | `pip install openai` needed |
| `rank-bm25` | BM25 fallback | **No** | — | `pip install rank-bm25` if needed |
| `trl` | SFT format docs only (Phase 6 runs on H100) | **No** | — | Install on H100 only |
| `distilabel` | Optional pipeline wrapper | **No** | — | Optional; only if custom loop insufficient |
| `reportlab` or `weasyprint` | Markdown → PDF | Unknown | — | Check before Task 6 |
| `gitpython` or `subprocess git` | Handbook clone | Built-in | — | `git` CLI sufficient |
| vLLM server | QA generation | Remote (H100) | ≥0.5.2 for FP8 | Not local |
| ngrok | Tunnel vLLM to local | Remote (H100 machine) | Any | Free tier OK for this volume |

**Missing dependencies (local machine, must install before Task 3):**
```bash
pip install openai rank-bm25
```

**Missing dependencies (H100 machine, must install before starting vLLM):**
```bash
pip install vllm>=0.5.2
# ngrok: download binary from ngrok.com or:
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok
ngrok config add-authtoken <YOUR_TOKEN>
```

---

## Section 7: Common Pitfalls

### Pitfall 1: JSON Parse Failures from LLM Output
**What goes wrong:** LLM wraps JSON in markdown fences (` ```json ... ``` `) or adds explanatory text before/after the JSON array.
**Why it happens:** Chat models default to markdown formatting even when instructed otherwise.
**How to avoid:** Strip markdown fences before parsing. Use regex: `re.search(r'\[.*\]', response, re.DOTALL)`.
**Warning signs:** `json.JSONDecodeError` on more than 5% of responses.

### Pitfall 2: multilingual-e5-small Missing Prefix
**What goes wrong:** Embeddings for queries and passages are computed in the same space without prefix differentiation, degrading retrieval quality.
**Why it happens:** The model was trained with asymmetric prefixes; omitting them shifts the embedding space.
**How to avoid:** Always prefix: `"query: "` for questions, `"passage: "` for chunks.
**Warning signs:** Cosine similarity between question and its source chunk drops below 0.5.

### Pitfall 3: Qwen2.5-72B OOM on Single H100 (BF16)
**What goes wrong:** vLLM fails to load with CUDA OOM error.
**Why it happens:** 72B in BF16 = 144GB, exceeds 96GB.
**How to avoid:** Use FP8 quantization (`--dtype fp8`) or 2-GPU tensor parallel. Confirm GPU count on H100 machine before launch.
**Warning signs:** vLLM startup crashes with `torch.cuda.OutOfMemoryError`.

### Pitfall 4: ngrok URL Changes on Restart
**What goes wrong:** Generation loop hardcodes `base_url`; after ngrok restart (H100 reboot), URL changes and all calls fail.
**Why it happens:** Free tier assigns ephemeral subdomain.
**How to avoid:** Read URL dynamically from ngrok API: `curl http://localhost:4040/api/tunnels | jq '.tunnels[0].public_url'` or register a static ngrok domain.

### Pitfall 5: Checkpoint File Corruption on Interrupt
**What goes wrong:** Interrupt during JSONL write leaves a malformed last line.
**Why it happens:** JSON append mode without atomic writes.
**How to avoid:** Write each batch to a `.tmp` file; rename to `.jsonl` atomically. Or: use `open(file, 'a')` + `json.dumps(record) + '\n'` (line-by-line JSON is atomic at line level).

### Pitfall 6: TRL SFTTrainer Extra Columns
**What goes wrong:** Training fails because `Dataset` has extra columns (e.g., `chunk_id`, `source`) that SFTTrainer tries to interpret as input.
**Why it happens:** SFTTrainer passes all dataset columns to the model.
**How to avoid:** Before training (Phase 6), call `dataset.remove_columns([col for col in dataset.column_names if col != "messages"])`.

### Pitfall 7: Vietnamese Answer Contains English Phrases
**What goes wrong:** LLM generates "mix" answers (e.g., "Nhân viên được nghỉ **fifteen** ngày...").
**Why it happens:** Source text is English; model may copy phrases verbatim.
**How to avoid:** Add quality filter: discard answers with >10% ASCII word ratio (suggests English contamination). Simple regex check.

---

## Implementation Guidance for PLAN.md Author

### Task-Level Decisions

| Task | Concrete Decision |
|------|-----------------|
| **Task 1 (Handbook Ingestion)** | `git clone --depth 1` each repo to `data/raw/{repo_name}/`. Extract all `.md` files recursively. Skip `README.md` at root if it's only meta-content. |
| **Task 2 (Chunking)** | Target 400-500 char chunks. Use `langchain_text_splitters.MarkdownTextSplitter` (installed: `langchain-text-splitters==1.1.2`) with `chunk_size=500, chunk_overlap=50`. Retain header as prefix in chunk. |
| **Task 3 (QA Generation)** | Run via custom OpenAI SDK loop. Install `pip install openai`. Checkpoint to `data/qa_pairs_checkpoint.jsonl` after each chunk. Resume by loading checkpoint and skipping processed chunk IDs. Use `n_pairs=2` default, `n_pairs=3` if chunk >400 chars. |
| **Task 4 (Quality Filter)** | Discard: JSON parse fails, answer > 3× chunk length, answer < 20 chars, >15% ASCII word ratio in answer. Log discard reason per pair. |
| **Task 5 (Training Format Conversion)** | Call `mine_hard_negatives` with `relative_margin=0.1, num_negatives=1, output_format="triplet"`. Save embedding triplets to `data/embedding_train.jsonl`. Convert to `messages` format for SFT JSONL. Save to `data/llm_train.jsonl`. |
| **Task 6 (Dataset Split)** | Use `datasets.train_test_split` twice: 80% train, then 50/50 dev/test on remainder. Save 6 files: `data/splits/{embedding,llm}_{train,dev,test}_split.jsonl`. |
| **Task 7 (Markdown → PDF)** | Use `markdown` + `weasyprint` or `reportlab` for at least 1 handbook. Verify `weasyprint` availability before committing to it. Alternative: `pandoc` CLI if installed. |
| **Task 8 (Validation)** | Assert ≥1500 rows in `data/qa_pairs.jsonl`, ≥1200 rows in `data/embedding_train.jsonl`, ≥1200 rows in `data/llm_train.jsonl`, valid JSON every line, `data/sample_handbook.pdf` exists. |

### Package Installation Summary

**Local machine (before Task 3):**
```bash
pip install openai rank-bm25
```

**H100 machine (before vLLM launch):**
```bash
pip install vllm>=0.5.2
# Install ngrok binary + auth token
```

### vLLM Launch Command (for H100 machine operator)

```bash
# CONFIRM which option applies to your hardware setup:

# Option A: Single H100 96GB with FP8 (recommended if only 1 GPU)
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000

# Option B: 2× H100 tensor parallel (if 2 GPUs available)
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000

# Option C: Single H100 96GB BF16 with 32B (if 72B FP8 not available)
vllm serve Qwen/Qwen2.5-32B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000
```

### ngrok Setup (for H100 machine operator)

```bash
# Step 1: Start vLLM (see above)
# Step 2: In a second terminal, start ngrok
ngrok http 8000

# Step 3: Note the HTTPS URL printed by ngrok, e.g.:
# Forwarding https://a1b2-x.ngrok-free.app -> http://localhost:8000

# Step 4: Pass that URL to the generation script on local machine:
# python scripts/generate_qa.py --vllm-url https://a1b2-x.ngrok-free.app
```

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Qwen2.5-72B produces correct Vietnamese diacritics reliably at temperature=0.2 | §2.2 | Diacritics errors in generated Q&A; need post-processing filter or pilot test |
| A2 | ~750 chunks × 2.3 avg pairs = ~1725 raw pairs before filtering | §1.5 | Could undershoot 1500 target; increase n_pairs or lower quality threshold |
| A3 | Including English chunk as context in SFT user message is best practice | §5.2 | Model may overfit to chunk-reading rather than generalizing; test on held-out set |
| A4 | weasyprint is available or installable on local Windows machine | §Task 7 | May need pandoc or reportlab as fallback |
| A5 | vLLM FP8 for Qwen2.5-72B fits on single H100 96GB with headroom | §4.1 | If tight, reduce `--gpu-memory-utilization` or use tensor parallel |

---

## Open Questions

1. **Which GPU configuration is available on the H100 machine?**
   - What we know: H100 96GB confirmed; 72B in BF16 does not fit a single card
   - What's unclear: Is it 1× H100 or 2× H100? Is FP8 checkpoint pre-downloaded?
   - Recommendation: Confirm before Task 3. If 1× H100: use FP8. If 2× H100: use tensor-parallel-size 2.

2. **Teacher LLM conflict: CONTEXT.md (Claude Haiku) vs. user request (vLLM+ngrok)**
   - What we know: Both approaches produce Vietnamese Q&A. Claude Haiku was chosen for "native Vietnamese quality + cheap API."
   - What's unclear: Is the vLLM+ngrok setup already running, or does it need to be set up?
   - Recommendation: If H100 is available and vLLM can be set up quickly, it is the better choice (free, no rate limits, larger model). If H100 setup is uncertain, keep Claude Haiku as fallback.

3. **`weasyprint` availability on Windows**
   - What we know: weasyprint has known Windows installation issues (GTK dependency)
   - What's unclear: Is it already installed or does the team have GTK?
   - Recommendation: Use `reportlab` as primary PDF library on Windows; it has no native dependencies.

---

## Sources

### Primary (HIGH confidence)
- [sbert.net/docs — mine_hard_negatives function](https://sbert.net/docs/package_reference/util/hard_negatives.html) — exact API signature verified
- [huggingface.co/docs/trl/v0.19.1/sft_trainer](https://huggingface.co/docs/trl/v0.19.1/sft_trainer) — SFTTrainer dataset formats verified
- [qwen.readthedocs.io/en/v2.5/deployment/vllm.html](https://qwen.readthedocs.io/en/v2.5/deployment/vllm.html) — Qwen2.5 vLLM launch commands
- [ngrok.com/docs/pricing-limits/free-plan-limits](https://ngrok.com/docs/pricing-limits/free-plan-limits) — free tier limits verified
- Local environment: `sentence-transformers==5.5.1`, `mine_hard_negatives` import confirmed

### Secondary (MEDIUM confidence)
- [distilabel.argilla.io/1.3.2/api/llm/vllm/](http://distilabel.argilla.io/1.3.2/api/llm/vllm/) — ClientvLLM class verified
- [arxiv.org/pdf/2402.05672](https://arxiv.org/pdf/2402.05672) — multilingual-e5 prefix requirements
- [qwenlm.github.io/blog/qwen2.5/](https://qwenlm.github.io/blog/qwen2.5/) — Vietnamese language support confirmed
- [github.com/vllm-project/vllm/issues/14792](https://github.com/vllm-project/vllm/issues/14792) — client timeout configuration pattern

### Tertiary (LOW confidence / ASSUMED)
- Vietnamese diacritics reliability at 72B scale — no benchmark found; estimated from model scale
- SFT context inclusion best practice — inferred from RAG-SFT literature, not a single authoritative source
- 750-chunk × 2.3-pairs estimate — derived from handbook size range in requirements

---

## Metadata

**Confidence breakdown:**
- vLLM launch commands: HIGH — verified via official Qwen + vLLM docs
- ngrok setup: HIGH — verified via official ngrok docs
- mine_hard_negatives API: HIGH — verified via official sbert docs + local import test
- TRL SFTTrainer format: HIGH — verified via official TRL docs
- Vietnamese QA prompt template: MEDIUM — based on RAG synthetic data literature, not Vietnamese-specific benchmark
- Vietnamese diacritics quality: LOW-MEDIUM — no diacritics error rate benchmark found for Qwen2.5-72B

**Research date:** 2026-06-01
**Valid until:** 2026-09-01 (stable libraries; vLLM API may change faster — re-check if vLLM version changes)
