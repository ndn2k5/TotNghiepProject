/gsd-new-milestone i am now having access to 1.5 bucks of Vast.ai for training. We can use up to H200 for 20 minute (141GB vram) to either trying to finetune model, or generate bulk tons of datasets for him.
The user has $1.50 of Vast.ai credit, which gives ~20 minutes on an H200 (141GB VRAM). They want to use this for either:

Fine-tuning a model
Generating bulk datasets
This is for their friend's thesis project. Let me think about what makes sense here.

Context from the project:

Previous QLoRA fine-tuning FAILED because training data was contaminated (63% off-domain)
They have 20 Vietnamese HR docs in the knowledge base
The thesis already documents the negative fine-tuning result
The friend needs to submit thesis by 2026-06-24 (3 days from now) I'm weighing two approaches: generating synthetic Vietnamese HR Q&A pairs from the existing documents using a large model like Qwen2.5-72B to fix the data quality issues that derailed the previous fine-tuning attempt, or attempting another fine-tuning run with cleaner data. The generation route would produce usable training material and demonstrate an improved data pipeline for the thesis, while fine-tuning is risky given the tight timeline and the fact that 20 minutes might not be enough to complete a meaningful training run.
Before I create a concrete plan, I should check with the friend about what matters most given their three-day deadline.

$1.50 on Vast.ai, H200 141GB VRAM, ~20 min window. Big opportunity but tight. Let me scope this right.

AskUserQuestion
OUT
Your questions have been answered: "With 20 min on H200, what's the priority? Previous QLoRA failed due to garbage training data (63% off-domain). Clean data generation is the bottleneck."="Generate datasets (Recommended)", "Which large model do you want to run on the H200 for data generation?"="Qwen2.5-72B-Instruct". You can now continue with these answers in mind.
The user wants to:

Focus on generating clean Vietnamese HR Q&A datasets
Use Qwen2.5-72B-Instruct on the H200
Now let me create a milestone for this. This is Milestone 5 — Clean Dataset Generation.

Key planning:

20 min on H200 with Qwen2.5-72B-Instruct
Goal: Generate thousands of clean Vietnamese HR Q&A pairs from the 20 existing HR docs
Input: 20 Vietnamese HR policy documents in data/viet_labor_docs/
Output: Clean Q&A pairs in CSV/JSONL format
Strategy: Feed each document as context to Qwen2.5-72B and ask it to generate diverse Q&A pairs
I need to prepare the generation script locally before renting the GPU to save money
Breaking this into phases: first I'll prepare the scripts and set up Vast.ai locally without any GPU cost, then execute the bulk generation on the H200, and finally clean and validate the output.

Let me check the existing roadmap to make sure I'm numbering the next phase correctly, then I'll create the milestone documentation.

Good. Dataset generation with Qwen2.5-72B-Instruct. Let me build the milestone.

Key insight: script prep happens locally BEFORE renting GPU — every second on H200 counts.

Read d:\PROJEct\AI MODELS\TotNghiepProject\.planning\ROADMAP.md (lines 381-400)
Now creating Milestone 5 and updating planning docs. Let me prepare everything.

Edit d:\PROJEct\AI MODELS\TotNghiepProject\.planning\ROADMAP.md
Added 1 line
Just markdown lint warnings about table formatting in ROADMAP.md — not relevant. Let me continue building the milestone. I need to:

Add Milestone 5 section to ROADMAP.md
Update STATE.md
Create the generation script that will run on H200
Let me think about the strategy:

Budget: $1.50, ~20 min on H200
Model: Qwen2.5-72B-Instruct (needs ~140GB for full precision, or ~40GB quantized)
Input: 20 Vietnamese HR docs in data/viet_labor_docs/
Goal: Generate as many clean Vietnamese HR Q&A pairs as possible
I'll load a quantized version of the model for faster initialization, then use vLLM for batched inference to maximize throughput across all 20 documents, aiming for 1000-2000 Q&A pairs total.
Now I need to prepare the generation script locally before renting the H200 instance, along with all dependencies and requirements.

Lint warnings, ignoring. Continuing milestone setup — adding Milestone 5 phases to ROADMAP and creating the generation script.

Read d:\PROJEct\AI MODELS\TotNghiepProject\.planning\ROADMAP.md (lines 491-505)
Edit d:\PROJEct\AI MODELS\TotNghiepProject\.planning\ROADMAP.md
Added 80 lines
Now update STATE.md.

Edit d:\PROJEct\AI MODELS\TotNghiepProject\.planning\STATE.md
Modified
Just markdown lint warnings again, not relevant. Let me now create the actual generation script and Vast.ai setup script. These are the critical deliverables — the user needs these ready BEFORE renting the GPU.

Let me now write the generation script. Key design decisions:

Use vLLM for batched inference (much faster than HuggingFace sequential)
Read all 20 docs from data/viet_labor_docs/
For each doc, generate diverse Q&A pairs using Qwen2.5-72B-Instruct
Output JSONL format
Need to maximize throughput in 20 minutes I should check what existing data generation scripts look like in this project to match the style and patterns.
Lint noise, ignoring. Now creating the actual scripts — this is what matters before renting.

Read d:\PROJEct\AI MODELS\TotNghiepProject\data\viet_labor_docs\chinh_sach_nghi_phep.txt
Good, docs are rich. Now creating the generation script.

Write d:\PROJEct\AI MODELS\TotNghiepProject\scripts\vast_generate_qa.py
258 lines
#!/usr/bin/env python3
"""
Bulk Vietnamese HR Q&A generation using Qwen2.5-72B-Instruct on Vast.ai H200.

Reads all 20 Vietnamese HR docs, generates diverse Q&A pairs via vLLM batched inference.
Designed for ~20 min on H200 141GB VRAM.

Usage:
    # On Vast.ai H200:
    python scripts/vast_generate_qa.py

    # Local dry-run (no GPU, mock outputs):
    python scripts/vast_generate_qa.py --dry-run

Output: data/generated_qa_h200.jsonl
"""

import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "viet_labor_docs"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "generated_qa_h200.jsonl"

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"

QA_PER_DOC_PER_BATCH = 10
BATCHES_PER_DOC = 5  # 5 batches × 10 Q&A = 50 pairs per doc × 20 docs = 1000 pairs

GENERATION_PROMPT = """Bạn là chuyên gia nhân sự Việt Nam. Dựa vào đoạn tài liệu nội bộ dưới đây, hãy tạo ra {n} cặp câu hỏi-trả lời đa dạng bằng tiếng Việt.

YÊU CẦU:
- Câu hỏi phải tự nhiên, giống như nhân viên thực sự hỏi (ngắn gọn, đời thường)
- Trả lời phải chính xác, dựa hoàn toàn trên nội dung tài liệu
- Đa dạng: câu hỏi có/không, câu hỏi số liệu, câu hỏi quy trình, câu hỏi so sánh
- Độ dài câu hỏi: 5-20 từ. Độ dài trả lời: 1-4 câu.
- KHÔNG bịa thông tin ngoài tài liệu
- Mỗi cặp Q&A trên một dòng, định dạng JSON: {{"q": "...", "a": "..."}}

TÀI LIỆU:
{document}

Tạo {n} cặp Q&A (chỉ output JSON, mỗi dòng một cặp):"""

DIVERSE_PROMPT = """Bạn là chuyên gia nhân sự Việt Nam. Dựa vào tài liệu dưới đây, tạo {n} câu hỏi-trả lời MỚI, KHÁC với các câu đã có.

CÂU HỎI ĐÃ CÓ (không được lặp lại):
{existing}

YÊU CẦU:
- Hỏi về các khía cạnh CHƯA được hỏi
- Câu hỏi tự nhiên, ngắn gọn (5-20 từ)
- Trả lời chính xác theo tài liệu (1-4 câu)
- Đa dạng loại câu hỏi: tình huống, so sánh, điều kiện, quy trình
- Định dạng JSON: {{"q": "...", "a": "..."}} mỗi dòng

TÀI LIỆU:
{document}

Tạo {n} cặp Q&A mới:"""


def load_documents() -> List[Dict]:
    """Load all Vietnamese HR docs from data/viet_labor_docs/."""
    docs = []
    for txt_file in sorted(DATA_DIR.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8").strip()
        if content:
            docs.append({
                "filename": txt_file.stem,
                "content": content,
                "char_count": len(content),
            })
    logger.info(f"Loaded {len(docs)} documents from {DATA_DIR}")
    return docs


def parse_qa_output(text: str) -> List[Dict]:
    """Parse model output into Q&A pairs. Tolerant of formatting issues."""
    pairs = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading number/bullet if present
        for prefix in ["- ", "* ", ". "]:
            if prefix in line[:5]:
                line = line[line.index(prefix) + len(prefix):]
        # Try to find JSON object
        start = line.find("{")
        end = line.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                obj = json.loads(line[start:end])
                q = obj.get("q", "").strip()
                a = obj.get("a", "").strip()
                if q and a and len(q) > 5 and len(a) > 5:
                    pairs.append({"question": q, "answer": a})
            except json.JSONDecodeError:
                continue
    return pairs


def generate_with_vllm(docs: List[Dict]) -> List[Dict]:
    """Generate Q&A pairs using vLLM batched inference."""
    from vllm import LLM, SamplingParams

    logger.info(f"Loading model: {MODEL_ID}")
    t0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    sampling = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=2048,
        repetition_penalty=1.1,
    )

    all_pairs = []

    for batch_idx in range(BATCHES_PER_DOC):
        logger.info(f"=== Batch {batch_idx + 1}/{BATCHES_PER_DOC} ===")
        prompts = []
        doc_map = []  # track which doc each prompt belongs to

        for doc in docs:
            if batch_idx == 0:
                prompt = GENERATION_PROMPT.format(
                    n=QA_PER_DOC_PER_BATCH,
                    document=doc["content"][:3000],
                )
            else:
                # Show existing questions to avoid duplicates
                existing_qs = [
                    p["question"] for p in all_pairs
                    if p.get("source_doc") == doc["filename"]
                ]
                existing_str = "\n".join(f"- {q}" for q in existing_qs[-20:])
                prompt = DIVERSE_PROMPT.format(
                    n=QA_PER_DOC_PER_BATCH,
                    existing=existing_str or "(không có)",
                    document=doc["content"][:3000],
                )
            prompts.append(prompt)
            doc_map.append(doc["filename"])

        t1 = time.time()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.time() - t1
        logger.info(f"Batch {batch_idx + 1} generated in {elapsed:.1f}s")

        batch_count = 0
        for output, doc_name in zip(outputs, doc_map):
            text = output.outputs[0].text
            pairs = parse_qa_output(text)
            for pair in pairs:
                pair["source_doc"] = doc_name
                pair["batch"] = batch_idx
            all_pairs.extend(pairs)
            batch_count += len(pairs)

        logger.info(f"Batch {batch_idx + 1}: {batch_count} pairs extracted (total: {len(all_pairs)})")

    return all_pairs


def generate_dry_run(docs: List[Dict]) -> List[Dict]:
    """Mock generation for local testing."""
    pairs = []
    for doc in docs:
        for i in range(3):
            pairs.append({
                "question": f"[DRY-RUN] Câu hỏi {i+1} về {doc['filename']}?",
                "answer": f"[DRY-RUN] Trả lời dựa trên tài liệu {doc['filename']}.",
                "source_doc": doc["filename"],
                "batch": 0,
            })
    return pairs


def save_results(pairs: List[Dict], output_path: Path):
    """Save Q&A pairs as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(pairs)} pairs to {output_path}")


def print_stats(pairs: List[Dict]):
    """Print generation statistics."""
    from collections import Counter
    doc_counts = Counter(p["source_doc"] for p in pairs)
    avg_q_len = sum(len(p["question"]) for p in pairs) / max(len(pairs), 1)
    avg_a_len = sum(len(p["answer"]) for p in pairs) / max(len(pairs), 1)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Q&A pairs: {len(pairs)}")
    print(f"Documents covered: {len(doc_counts)}")
    print(f"Avg question length: {avg_q_len:.0f} chars")
    print(f"Avg answer length: {avg_a_len:.0f} chars")
    print(f"\nPer-document breakdown:")
    for doc, count in sorted(doc_counts.items()):
        print(f"  {doc}: {count} pairs")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Bulk Vietnamese HR Q&A generation")
    parser.add_argument("--dry-run", action="store_true", help="Mock generation (no GPU)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    parser.add_argument("--batches", type=int, default=BATCHES_PER_DOC)
    parser.add_argument("--per-batch", type=int, default=QA_PER_DOC_PER_BATCH)
    args = parser.parse_args()

    global BATCHES_PER_DOC, QA_PER_DOC_PER_BATCH
    BATCHES_PER_DOC = args.batches
    QA_PER_DOC_PER_BATCH = args.per_batch

    docs = load_documents()
    if not docs:
        logger.error(f"No documents found in {DATA_DIR}")
        return

    t_start = time.time()

    if args.dry_run:
        logger.info("DRY-RUN mode — no GPU, mock outputs")
        pairs = generate_dry_run(docs)
    else:
        pairs = generate_with_vllm(docs)

    elapsed = time.time() - t_start

    output_path = Path(args.output)
    save_results(pairs, output_path)
    print_stats(pairs)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
Write d:\PROJEct\AI MODELS\TotNghiepProject\scripts\vast_setup.sh
48 lines
#!/bin/bash
# ============================================================
# Vast.ai H200 Setup — Run this IMMEDIATELY after SSH into instance
# Budget: $1.50 (~20 min). Every second counts.
# ============================================================

set -e

echo "=========================================="
echo "  VAST.AI H200 — QA GENERATION SETUP"
echo "=========================================="
START=$(date +%s)

# Step 1: Install vLLM (pre-compiled wheels, fastest path)
echo "[1/4] Installing vLLM..."
pip install vllm --quiet 2>&1 | tail -1

# Step 2: Clone repo (or upload manually)
echo "[2/4] Cloning project repo..."
if [ ! -d "TotNghiepProject" ]; then
    git clone --depth 1 https://github.com/ndn2k5/TotNghiepProject.git
fi
cd TotNghiepProject

# Step 3: Pre-download model (vLLM will cache it)
echo "[3/4] Model will download on first run via vLLM..."
echo "       Using: Qwen/Qwen2.5-72B-Instruct-AWQ"

# Step 4: Run generation
echo "[4/4] Starting Q&A generation..."
python scripts/vast_generate_qa.py \
    --batches 5 \
    --per-batch 10 \
    --output data/generated_qa_h200.jsonl

END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "=========================================="
echo "  DONE in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo "  1. Download: data/generated_qa_h200.jsonl"
echo "  2. Run validation: python scripts/validate_qa.py"
echo "  3. TERMINATE THIS INSTANCE NOW to save money!"
echo ""
Write d:\PROJEct\AI MODELS\TotNghiepProject\scripts\validate_qa.py
182 lines
#!/usr/bin/env python3
"""
Validate generated Q&A pairs for quality and domain relevance.
Flags off-domain content (the problem that killed previous QLoRA attempt).

Usage:
    python scripts/validate_qa.py [--input data/generated_qa_h200.jsonl]
"""

import json
import argparse
import re
from pathlib import Path
from collections import Counter

DEFAULT_INPUT = Path(__file__).parent.parent / "data" / "generated_qa_h200.jsonl"

# Vietnamese HR keywords — if a Q&A pair contains none of these, it's suspicious
HR_KEYWORDS_VI = {
    "nghỉ phép", "nghỉ việc", "lương", "hợp đồng", "bảo hiểm",
    "nhân viên", "công ty", "lao động", "kỷ luật", "phúc lợi",
    "tuyển dụng", "đào tạo", "thai sản", "chế độ", "quy định",
    "thử việc", "phụ cấp", "trợ cấp", "nội quy", "bảo mật",
    "đánh giá", "hiệu suất", "công tác", "nghỉ ốm", "BHXH",
    "quản lý", "phòng nhân sự", "hành vi", "vi phạm", "sa thải",
    "làm thêm", "tăng ca", "nghỉ lễ", "phép năm", "hưởng lương",
    "an toàn", "sức khỏe", "quấy rối", "tham nhũng", "bàn giao",
    "từ xa", "remote", "di chuyển", "đi công tác",
}

# Off-domain red flags
RED_FLAGS = {
    "medicare", "social security (US)", "election", "voting",
    "japanese", "labor law japan", "日本", "選挙",
}


def load_pairs(path: Path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_line"] = line_num
                pairs.append(obj)
            except json.JSONDecodeError:
                print(f"  WARNING: Invalid JSON on line {line_num}")
    return pairs


def check_domain_relevance(pair):
    """Check if Q&A pair is Vietnamese HR-relevant."""
    text = (pair.get("question", "") + " " + pair.get("answer", "")).lower()

    # Check for red flags
    for flag in RED_FLAGS:
        if flag.lower() in text:
            return "RED_FLAG", flag

    # Check for HR keywords
    for kw in HR_KEYWORDS_VI:
        if kw.lower() in text:
            return "RELEVANT", kw

    return "UNCLEAR", None


def validate(pairs):
    """Run all validation checks."""
    stats = {
        "total": len(pairs),
        "relevant": 0,
        "unclear": 0,
        "red_flag": 0,
        "empty_q": 0,
        "empty_a": 0,
        "short_q": 0,
        "short_a": 0,
        "duplicate_q": 0,
    }

    doc_counts = Counter()
    seen_questions = set()
    flagged = []

    for pair in pairs:
        q = pair.get("question", "").strip()
        a = pair.get("answer", "").strip()
        doc = pair.get("source_doc", "unknown")
        doc_counts[doc] += 1

        if not q:
            stats["empty_q"] += 1
        if not a:
            stats["empty_a"] += 1
        if len(q) < 10:
            stats["short_q"] += 1
        if len(a) < 10:
            stats["short_a"] += 1

        q_norm = re.sub(r"\s+", " ", q.lower().strip())
        if q_norm in seen_questions:
            stats["duplicate_q"] += 1
        seen_questions.add(q_norm)

        status, detail = check_domain_relevance(pair)
        if status == "RELEVANT":
            stats["relevant"] += 1
        elif status == "RED_FLAG":
            stats["red_flag"] += 1
            flagged.append((pair["_line"], q[:60], detail))
        else:
            stats["unclear"] += 1

    # Print report
    print(f"\n{'='*60}")
    print(f"  Q&A VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Total pairs:        {stats['total']}")
    print(f"Domain-relevant:    {stats['relevant']} ({100*stats['relevant']/max(stats['total'],1):.1f}%)")
    print(f"Unclear relevance:  {stats['unclear']}")
    print(f"RED FLAGS:          {stats['red_flag']}")
    print(f"Empty questions:    {stats['empty_q']}")
    print(f"Empty answers:      {stats['empty_a']}")
    print(f"Short questions:    {stats['short_q']} (<10 chars)")
    print(f"Short answers:      {stats['short_a']} (<10 chars)")
    print(f"Duplicate questions:{stats['duplicate_q']}")

    avg_q = sum(len(p.get("question", "")) for p in pairs) / max(len(pairs), 1)
    avg_a = sum(len(p.get("answer", "")) for p in pairs) / max(len(pairs), 1)
    print(f"\nAvg question len:   {avg_q:.0f} chars")
    print(f"Avg answer len:     {avg_a:.0f} chars")

    print(f"\nPer-document breakdown:")
    for doc, count in sorted(doc_counts.items()):
        print(f"  {doc}: {count} pairs")

    if flagged:
        print(f"\n⚠️  RED-FLAGGED PAIRS ({len(flagged)}):")
        for line_num, q_preview, flag in flagged[:10]:
            print(f"  Line {line_num}: [{flag}] {q_preview}...")

    # Verdict
    relevance_pct = 100 * stats["relevant"] / max(stats["total"], 1)
    print(f"\n{'='*60}")
    if relevance_pct >= 90 and stats["red_flag"] == 0:
        print(f"  ✅ PASS — {relevance_pct:.0f}% domain-relevant, 0 red flags")
    elif relevance_pct >= 80:
        print(f"  ⚠️  WARN — {relevance_pct:.0f}% relevant, review unclear pairs")
    else:
        print(f"  ❌ FAIL — only {relevance_pct:.0f}% relevant, data needs cleaning")
    print(f"{'='*60}\n")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        print("Run vast_generate_qa.py first, or use --input to specify file.")
        return

    pairs = load_pairs(path)
    if not pairs:
        print("No valid Q&A pairs found.")
        return

    print(f"Loaded {len(pairs)} pairs from {path}")
    validate(pairs)


if __name__ == "__main__":
    main()
Now let me test the dry-run to make sure the script works locally.

Bash Test generation script in dry-run mode
IN
cd "d:\PROJEct\AI MODELS\TotNghiepProject" && python scripts/vast_generate_qa.py --dry-run 2>&1

OUT
Exit code 1
  File "D:\PROJEct\AI MODELS\TotNghiepProject\scripts\vast_generate_qa.py", line 231
    global BATCHES_PER_DOC, QA_PER_DOC_PER_BATCH
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: name 'BATCHES_PER_DOC' is used prior to global declaration
Edit d:\PROJEct\AI MODELS\TotNghiepProject\scripts\vast_generate_qa.py
Removed 1 line
Need to also update the functions to accept these as params. Let me fix the architecture — pass config through instead of using globals.

Edit d:\PROJEct\AI MODELS\TotNghiepProject\scripts\vast_generate_qa.py
Modified
Edit d:\PROJEct\AI MODELS\TotNghiepProject\scripts\vast_generate_qa.py
Removed 1 line
You've hit your monthly spend limit · raise it at claude.ai/settings/usage