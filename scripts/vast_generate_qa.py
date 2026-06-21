#!/usr/bin/env python3
"""
Bulk Vietnamese HR Q&A generation using Qwen2.5-72B-Instruct on Vast.ai H200.

Reads all 20 Vietnamese HR docs, generates diverse Q&A pairs via vLLM batched inference.
Designed for ~20 min on H200 141GB VRAM (or any large-VRAM GPU).

Usage:
    # On Vast.ai H200:
    python scripts/vast_generate_qa.py

    # High-volume run (10 batches × 15 pairs = 150 per doc → 3000 total):
    python scripts/vast_generate_qa.py --batches 10 --per-batch 15

    # Resume interrupted run:
    python scripts/vast_generate_qa.py --resume

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

# Default: 5 batches × 10 pairs × 20 docs = 1000 pairs total
DEFAULT_BATCHES = 5
DEFAULT_PER_BATCH = 10

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


# ─────────────────────────── data loading ──────────────────────────────────

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
                "doc_topic": txt_file.stem.replace("_", " ").title(),
            })
    logger.info(f"Loaded {len(docs)} documents from {DATA_DIR}")
    return docs


def load_existing_pairs(output_path: Path) -> List[Dict]:
    """Load already-generated pairs to support resumption."""
    if not output_path.exists():
        return []
    pairs = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON on line {line_no}")
    logger.info(f"Resume: loaded {len(pairs)} existing pairs from {output_path}")
    return pairs


# ─────────────────────────── parsing ───────────────────────────────────────

def parse_qa_output(text: str) -> List[Dict]:
    """Parse model output into Q&A pairs. Tolerant of common formatting issues."""
    pairs = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading number/bullet markers (e.g. "1. ", "- ")
        for prefix in ["- ", "* "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
        import re
        line = re.sub(r"^\d+\.\s*", "", line)

        # Extract JSON object
        start = line.find("{")
        end = line.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                obj = json.loads(line[start:end])
                q = str(obj.get("q", "")).strip()
                a = str(obj.get("a", "")).strip()
                if len(q) > 5 and len(a) > 5:
                    pairs.append({"question": q, "answer": a})
            except json.JSONDecodeError:
                continue
    return pairs


# ─────────────────────────── generation ────────────────────────────────────

def generate_with_vllm(
    docs: List[Dict],
    batches: int = DEFAULT_BATCHES,
    per_batch: int = DEFAULT_PER_BATCH,
    output_path: Path = OUTPUT_FILE,
    resume: bool = False,
) -> List[Dict]:
    """Generate Q&A pairs using vLLM batched inference.

    Args:
        docs:        List of document dicts with filename/content/doc_topic keys.
        batches:     Number of generation passes per document.
        per_batch:   Q&A pairs requested per document per batch.
        output_path: Streaming output path (results written as they finish).
        resume:      If True, load existing output and skip completed (doc, batch) pairs.
    """
    from vllm import LLM, SamplingParams
    import torch

    # Tensor parallelism: use all available GPUs, capped at 8 (Qwen2.5-72B has 64 heads)
    n_gpus = max(1, torch.cuda.device_count())
    tp_size = 1
    for candidate in [8, 4, 2, 1]:
        if n_gpus >= candidate:
            tp_size = candidate
            break

    logger.info(f"GPU count: {n_gpus} | tensor_parallel_size: {tp_size}")
    logger.info(f"Loading model: {MODEL_ID}")
    t0 = time.time()

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=tp_size,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=False,  # enable CUDA graph for throughput
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    sampling = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=2048,
        repetition_penalty=1.1,
    )

    # Load existing pairs for resume logic
    existing: List[Dict] = load_existing_pairs(output_path) if resume else []
    all_pairs: List[Dict] = list(existing)

    # Track which (doc, batch) combos already have data
    done_keys = set()
    if resume:
        for p in existing:
            done_keys.add((p.get("source_doc", ""), p.get("batch", -1)))

    # Open output file: append if resuming, overwrite otherwise
    file_mode = "a" if resume else "w"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, file_mode, encoding="utf-8") as out_file:
        for batch_idx in range(batches):
            logger.info(f"=== Batch {batch_idx + 1}/{batches} ===")

            # Build prompts only for docs not already done in this batch
            active_docs = []
            active_prompts = []

            for doc in docs:
                key = (doc["filename"], batch_idx)
                if key in done_keys:
                    logger.info(f"  Skip {doc['filename']} batch {batch_idx} (already done)")
                    continue

                if batch_idx == 0:
                    prompt = GENERATION_PROMPT.format(
                        n=per_batch,
                        document=doc["content"][:3000],
                    )
                else:
                    existing_qs = [
                        p["question"] for p in all_pairs
                        if p.get("source_doc") == doc["filename"]
                    ]
                    existing_str = "\n".join(f"- {q}" for q in existing_qs[-20:])
                    prompt = DIVERSE_PROMPT.format(
                        n=per_batch,
                        existing=existing_str or "(không có)",
                        document=doc["content"][:3000],
                    )

                active_docs.append(doc)
                active_prompts.append(prompt)

            if not active_prompts:
                logger.info(f"  Batch {batch_idx + 1}: nothing to generate, skipping")
                continue

            logger.info(f"  Generating for {len(active_docs)} docs...")
            t1 = time.time()
            outputs = llm.generate(active_prompts, sampling)
            logger.info(f"  Batch {batch_idx + 1} inference done in {time.time() - t1:.1f}s")

            batch_count = 0
            for output, doc in zip(outputs, active_docs):
                text = output.outputs[0].text
                pairs = parse_qa_output(text)
                for pair in pairs:
                    pair["source_doc"] = doc["filename"]
                    pair["doc_topic"] = doc.get("doc_topic", "")
                    pair["batch"] = batch_idx
                    out_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                all_pairs.extend(pairs)
                batch_count += len(pairs)
                done_keys.add((doc["filename"], batch_idx))

            out_file.flush()
            logger.info(
                f"  Batch {batch_idx + 1}: {batch_count} new pairs "
                f"(running total: {len(all_pairs)})"
            )

    return all_pairs


def generate_dry_run(docs: List[Dict]) -> List[Dict]:
    """Mock generation for local testing without GPU."""
    pairs = []
    for doc in docs:
        for i in range(3):
            pairs.append({
                "question": f"[DRY-RUN] Câu hỏi {i+1} về {doc['filename']}?",
                "answer": f"[DRY-RUN] Trả lời dựa trên tài liệu {doc['filename']}.",
                "source_doc": doc["filename"],
                "doc_topic": doc.get("doc_topic", ""),
                "batch": 0,
            })
    return pairs


# ─────────────────────────── output helpers ────────────────────────────────

def save_results(pairs: List[Dict], output_path: Path):
    """Save Q&A pairs as JSONL (for dry-run — overwrites)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(pairs)} pairs to {output_path}")


def print_stats(pairs: List[Dict]):
    """Print generation statistics."""
    from collections import Counter
    if not pairs:
        print("No pairs generated.")
        return
    doc_counts = Counter(p.get("source_doc", "unknown") for p in pairs)
    avg_q = sum(len(p.get("question", "")) for p in pairs) / len(pairs)
    avg_a = sum(len(p.get("answer", "")) for p in pairs) / len(pairs)

    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Q&A pairs:    {len(pairs)}")
    print(f"Documents covered:  {len(doc_counts)}")
    print(f"Avg question len:   {avg_q:.0f} chars")
    print(f"Avg answer len:     {avg_a:.0f} chars")
    print(f"\nPer-document breakdown:")
    for doc, count in sorted(doc_counts.items()):
        print(f"  {doc}: {count} pairs")
    print(f"{'='*60}\n")


# ─────────────────────────── main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bulk Vietnamese HR Q&A generation via vLLM (Vast.ai H200)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock generation — no GPU required")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                        help="Output JSONL path")
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES,
                        help=f"Generation passes per document (default: {DEFAULT_BATCHES})")
    parser.add_argument("--per-batch", type=int, default=DEFAULT_PER_BATCH,
                        help=f"Q&A pairs per doc per batch (default: {DEFAULT_PER_BATCH})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip completed batches)")
    args = parser.parse_args()

    docs = load_documents()
    if not docs:
        logger.error(f"No .txt documents found in {DATA_DIR}")
        logger.error("Make sure the viet_labor_docs folder is populated.")
        return

    target = len(docs) * args.batches * args.per_batch
    logger.info(
        f"Plan: {len(docs)} docs × {args.batches} batches × {args.per_batch} pairs "
        f"= ~{target} target pairs"
    )

    output_path = Path(args.output)
    t_start = time.time()

    if args.dry_run:
        logger.info("DRY-RUN mode — skipping model load")
        pairs = generate_dry_run(docs)
        save_results(pairs, output_path)
    else:
        pairs = generate_with_vllm(
            docs,
            batches=args.batches,
            per_batch=args.per_batch,
            output_path=output_path,
            resume=args.resume,
        )

    elapsed = time.time() - t_start
    print_stats(pairs)
    print(f"Output: {output_path}")
    print(f"Time:   {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
