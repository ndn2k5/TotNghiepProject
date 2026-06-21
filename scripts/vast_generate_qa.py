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


def generate_with_vllm(docs: List[Dict], batches: int = 5, per_batch: int = 10) -> List[Dict]:
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

    for batch_idx in range(batches):
        logger.info(f"=== Batch {batch_idx + 1}/{batches} ===")
        prompts = []
        doc_map = []

        for doc in docs:
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

    batches = args.batches
    per_batch = args.per_batch

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
