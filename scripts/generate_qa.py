"""
Generate Vietnamese Q&A pairs from handbook chunks using
Qwen2.5-72B served via vLLM + ngrok tunnel.
Checkpoint-safe: re-run after failure to resume.

Usage:
    python scripts/generate_qa.py --vllm-url https://xxxx.ngrok-free.app
"""
import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INPUT_FILE = Path("data/raw_chunks.jsonl")
OUTPUT_FILE = Path("data/qa_pairs.jsonl")

LOG_FILE = Path("data/qa_generation_log.txt")
CHECKPOINT_FILE = Path("data/qa_checkpoint.json")

SYSTEM_PROMPT = """Bạn là chuyên gia nhân sự. Hãy đọc đoạn văn bản từ sổ tay nhân viên \
và tạo ra {n_pairs} cặp hỏi-đáp bằng tiếng Việt.

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
                        help="API base URL — ngrok (https://xxxx.ngrok-free.app) or Groq (https://api.groq.com/openai)")
    parser.add_argument("--api-key", default="fake",
                        help="API key — 'fake' for local vLLM, real key for Groq/OpenRouter")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct",
                        help="Model name — e.g. llama-3.1-70b-versatile for Groq")
    parser.add_argument("--input", default=str(INPUT_FILE),
                        help="Input JSONL chunks file (default: data/raw_chunks.jsonl)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: only process first 10 chunks")
    parser.add_argument("--target-qa-pairs", type=int, default=0,
                        help="Stop early once the output file contains at least this many Q&A pairs")
    parser.add_argument("--max-new-chunks", type=int, default=0,
                        help="Stop after processing this many new chunks in the current run")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print progress every N newly processed chunks")
    args = parser.parse_args()

    input_file = Path(args.input)

    # Derive output + checkpoint from input stem when using non-default input
    global OUTPUT_FILE, CHECKPOINT_FILE, LOG_FILE
    if str(input_file) != str(INPUT_FILE):
        stem = input_file.stem.replace("raw_chunks", "qa_pairs").replace("raw_chunks_", "qa_pairs_")
        if stem == input_file.stem:
            stem = "qa_pairs_" + input_file.stem
        OUTPUT_FILE = input_file.parent / f"{stem}.jsonl"
        CHECKPOINT_FILE = input_file.parent / f"qa_checkpoint_{input_file.stem}.json"
        LOG_FILE = input_file.parent / f"qa_generation_log_{input_file.stem}.txt"
        print(f"Output:     {OUTPUT_FILE}")
        print(f"Checkpoint: {CHECKPOINT_FILE}")

    # Extend timeout — 72B model can take 30-90s per request
    base_url = args.vllm_url if args.vllm_url.endswith("/v1") else f"{args.vllm_url}/v1"
    client = OpenAI(
        base_url=base_url,
        api_key=args.api_key,
        http_client=httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10)),
    )

    chunks = [json.loads(l) for l in open(input_file, encoding="utf-8")]
    if args.test:
        chunks = chunks[:10]
        print("TEST MODE: processing first 10 chunks only")

    processed = load_checkpoint()
    existing_qa = sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8")) if OUTPUT_FILE.exists() else 0
    print(f"Chunks: {len(chunks)} | Already processed: {len(processed)} | Existing QA: {existing_qa}")

    total_qa = existing_qa
    errors = 0
    start = time.time()
    new_chunks_processed = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, \
         open(LOG_FILE, "a", encoding="utf-8") as log_f:

        for idx, chunk in enumerate(chunks):
            if args.target_qa_pairs and total_qa >= args.target_qa_pairs:
                print(f"Reached target QA pair budget: {total_qa}/{args.target_qa_pairs}. Stopping early.")
                break
            if args.max_new_chunks and new_chunks_processed >= args.max_new_chunks:
                print(f"Reached max new chunk budget: {new_chunks_processed}/{args.max_new_chunks}. Stopping early.")
                break
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
                new_chunks_processed += 1

                if args.progress_every > 0 and new_chunks_processed % args.progress_every == 0:
                    elapsed = time.time() - start
                    rate = max(new_chunks_processed / elapsed, 1e-9)
                    remaining_budget_chunks = None
                    if args.max_new_chunks:
                        remaining_budget_chunks = max(args.max_new_chunks - new_chunks_processed, 0)
                    remaining_unprocessed_chunks = sum(1 for chunk_idx in range(len(chunks)) if chunk_idx not in processed)
                    remaining_chunks = remaining_unprocessed_chunks if remaining_budget_chunks is None else min(remaining_unprocessed_chunks, remaining_budget_chunks)
                    eta_min = remaining_chunks / rate / 60
                    print(
                        f"[chunk {idx+1}/{len(chunks)} | new {new_chunks_processed}] "
                        f"QA: {total_qa} | ETA: {eta_min:.1f}min"
                    )
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
    print(f"\nDone. QA pairs: {total_qa} | Errors: {errors} | Time: {(time.time()-start)/60:.1f}min")


if __name__ == "__main__":
    main()