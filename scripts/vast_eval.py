#!/usr/bin/env python3
"""
Evaluate fine-tuned model on Vast.ai H200 — runs right after training.

Compares base Phi-3-Mini vs fine-tuned on 20 Vietnamese HR test questions.
Produces structured eval report + scoring CSV.

Usage:
    # After vast_finetune.py:
    python scripts/vast_eval.py

    # Custom paths:
    python scripts/vast_eval.py --adapter models/phi3-mini-hr-finetuned --data data/generated_qa_h200.jsonl

    # Dry-run:
    python scripts/vast_eval.py --dry-run
"""

import json
import time
import argparse
import logging
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
ADAPTER_DIR = REPO_ROOT / "models" / "phi3-mini-hr-finetuned"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
EVAL_OUTPUT = REPO_ROOT / "data" / "eval_report.txt"
EVAL_CSV = REPO_ROOT / "data" / "eval_scores.csv"

TEST_QUESTIONS = [
    ("Nhân viên được nghỉ phép bao nhiêu ngày mỗi năm?", "chinh_sach_nghi_phep"),
    ("Quy trình xin nghỉ phép như thế nào?", "chinh_sach_nghi_phep"),
    ("Lương thử việc là bao nhiêu phần trăm?", "hop_dong_lao_dong"),
    ("Nghỉ thai sản được bao lâu?", "nghi_thai_san_va_phu_nu"),
    ("Bảo hiểm xã hội trích bao nhiêu phần trăm?", "bao_hiem_xa_hoi"),
    ("Chính sách làm thêm giờ như thế nào?", "lam_them_gio"),
    ("Kỷ luật lao động gồm những hình thức gì?",  "ky_luat_lao_dong"),
    ("Công ty có chính sách làm việc từ xa không?", "chinh_sach_lam_viec_tu_xa"),
    ("Lương tháng 13 có không?", "chinh_sach_luong"),
    ("Quy trình nghỉ việc như thế nào?", "nghi_viec_va_ban_giao"),
    ("Nhân viên mới có thời gian thử việc bao lâu?", "hop_dong_lao_dong"),
    ("Chế độ ốm đau được hưởng bao nhiêu phần trăm lương?", "chinh_sach_nghi_phep"),
    ("Chính sách phúc lợi nhân viên gồm những gì?", "phuc_loi_nhan_vien"),
    ("Tuyển dụng nhân viên mới qua mấy vòng?", "tuyen_dung_dao_tao"),
    ("Quy định về bảo mật thông tin IT là gì?", "bao_mat_thong_tin_it"),
    ("Công tác phí được tính như thế nào?", "cong_tac_va_di_chuyen"),
    ("Quy trình đánh giá hiệu suất nhân viên?", "danh_gia_hieu_suat"),
    ("Chính sách chống quấy rối tại nơi làm việc?", "quay_roi_phan_biet_doi_xu"),
    ("Phụ cấp ăn trưa là bao nhiêu?", "phu_cap_va_tro_cap"),
    ("An toàn lao động quy định những gì?", "suc_khoe_an_toan_lao_dong"),
]

HR_KEYWORDS = {
    "nghỉ phép", "nghỉ việc", "lương", "hợp đồng", "bảo hiểm",
    "nhân viên", "công ty", "lao động", "kỷ luật", "phúc lợi",
    "tuyển dụng", "đào tạo", "thai sản", "chế độ", "quy định",
    "thử việc", "phụ cấp", "trợ cấp", "nội quy", "nghỉ ốm",
    "quản lý", "bảo mật", "đánh giá", "hiệu suất", "an toàn",
}

HALLUCINATION_FLAGS = [
    "medicare", "social security", "election", "voting", "japan",
    "i'm sorry", "as an ai", "i cannot", "i don't have",
]


def score_answer(question: str, answer: str, topic: str) -> Dict:
    """Score a single answer on multiple dimensions."""
    answer_lower = answer.lower()

    hr_relevant = any(kw in answer_lower for kw in HR_KEYWORDS)
    hallucinated = any(flag in answer_lower for flag in HALLUCINATION_FLAGS)
    is_vietnamese = sum(1 for c in answer if ord(c) > 127) > len(answer) * 0.05
    is_empty = len(answer.strip()) < 10
    is_refusal = "không tìm thấy" in answer_lower or "i cannot" in answer_lower

    score = 0
    if hr_relevant and not hallucinated and is_vietnamese and not is_empty:
        score = 1
    elif hr_relevant and not hallucinated:
        score = 0.5

    return {
        "hr_relevant": hr_relevant,
        "hallucinated": hallucinated,
        "vietnamese": is_vietnamese,
        "empty": is_empty,
        "refusal": is_refusal,
        "score": score,
    }


def generate_answer(model, tokenizer, question: str, max_tokens: int = 256) -> str:
    """Generate answer from model."""
    import torch
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def load_finetuned_model(adapter_path: Path):
    """Load base model + LoRA adapter via Unsloth."""
    from unsloth import FastLanguageModel
    logger.info(f"Loading fine-tuned model from {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_base_model():
    """Load base Phi-3-Mini for comparison."""
    from unsloth import FastLanguageModel
    logger.info(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_eval(model, tokenizer, label: str) -> List[Dict]:
    """Run evaluation on all test questions."""
    results = []
    for i, (question, topic) in enumerate(TEST_QUESTIONS):
        logger.info(f"  [{i+1}/{len(TEST_QUESTIONS)}] {question[:50]}...")
        t0 = time.time()
        answer = generate_answer(model, tokenizer, question)
        elapsed = time.time() - t0
        scores = score_answer(question, answer, topic)
        results.append({
            "model": label,
            "question": question,
            "topic": topic,
            "answer": answer[:500],
            "latency_s": round(elapsed, 2),
            **scores,
        })
    return results


def write_report(base_results: List[Dict], ft_results: Optional[List[Dict]], output_path: Path):
    """Write human-readable evaluation report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  VIETNAMESE HR CHATBOT — EVALUATION REPORT")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    for label, results in [("BASE (Phi-3-Mini)", base_results), ("FINE-TUNED", ft_results)]:
        if results is None:
            continue
        lines.append(f"\n{'='*70}")
        lines.append(f"  MODEL: {label}")
        lines.append(f"{'='*70}")

        total = len(results)
        avg_score = sum(r["score"] for r in results) / total
        hr_count = sum(1 for r in results if r["hr_relevant"])
        hall_count = sum(1 for r in results if r["hallucinated"])
        avg_latency = sum(r["latency_s"] for r in results) / total

        lines.append(f"  Overall score:    {avg_score:.2f} / 1.00")
        lines.append(f"  HR-relevant:      {hr_count}/{total}")
        lines.append(f"  Hallucinations:   {hall_count}/{total}")
        lines.append(f"  Avg latency:      {avg_latency:.1f}s")
        lines.append("")

        for i, r in enumerate(results, 1):
            status = "[OK]" if r["score"] >= 0.5 else "[FAIL]"
            if r["hallucinated"]:
                status = "[HALL]"
            lines.append(f"Q{i}: {r['question']}")
            lines.append(f"  {status} ({r['latency_s']:.1f}s) {r['answer'][:200]}")
            lines.append("")

    # Comparison summary
    if ft_results and base_results:
        lines.append("=" * 70)
        lines.append("  COMPARISON: BASE vs FINE-TUNED")
        lines.append("=" * 70)
        base_score = sum(r["score"] for r in base_results) / len(base_results)
        ft_score = sum(r["score"] for r in ft_results) / len(ft_results)
        base_hall = sum(1 for r in base_results if r["hallucinated"])
        ft_hall = sum(1 for r in ft_results if r["hallucinated"])
        lines.append(f"  {'Metric':<25} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
        lines.append(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")
        lines.append(f"  {'Score':.<25} {base_score:>10.2f} {ft_score:>12.2f} {ft_score-base_score:>+10.2f}")
        lines.append(f"  {'Hallucinations':.<25} {base_hall:>10} {ft_hall:>12} {ft_hall-base_hall:>+10}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved to {output_path}")
    print("\n".join(lines[-15:]))


def write_csv(all_results: List[Dict], csv_path: Path):
    """Write scoring CSV for downstream analysis."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "question", "topic", "score", "hr_relevant",
              "hallucinated", "vietnamese", "empty", "refusal", "latency_s", "answer"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fields})
    logger.info(f"CSV saved to {csv_path}")


def dry_run():
    logger.info("DRY-RUN mode")
    results = []
    for q, topic in TEST_QUESTIONS:
        results.append({
            "model": "dry-run", "question": q, "topic": topic,
            "answer": f"[DRY-RUN] Tra loi ve {topic}",
            "latency_s": 0.0, "hr_relevant": False, "hallucinated": False,
            "vietnamese": False, "empty": False, "refusal": False, "score": 0,
        })
    write_report(results, None, EVAL_OUTPUT)
    write_csv(results, EVAL_CSV)
    print("\n[OK] Dry-run eval complete\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned model")
    parser.add_argument("--adapter", type=str, default=str(ADAPTER_DIR))
    parser.add_argument("--base-only", action="store_true", help="Only eval base model")
    parser.add_argument("--ft-only", action="store_true", help="Only eval fine-tuned model")
    parser.add_argument("--output", type=str, default=str(EVAL_OUTPUT))
    parser.add_argument("--csv", type=str, default=str(EVAL_CSV))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    base_results = None
    ft_results = None

    if not args.ft_only:
        logger.info("=== Evaluating BASE model ===")
        base_model, base_tok = load_base_model()
        base_results = run_eval(base_model, base_tok, "base")
        del base_model, base_tok
        import torch; torch.cuda.empty_cache()

    adapter_path = Path(args.adapter)
    if not args.base_only and adapter_path.exists():
        logger.info("=== Evaluating FINE-TUNED model ===")
        ft_model, ft_tok = load_finetuned_model(adapter_path)
        ft_results = run_eval(ft_model, ft_tok, "finetuned")
        del ft_model, ft_tok
        import torch; torch.cuda.empty_cache()
    elif not args.base_only:
        logger.warning(f"Adapter not found at {adapter_path} — skipping fine-tuned eval")

    write_report(base_results, ft_results, Path(args.output))
    all_results = (base_results or []) + (ft_results or [])
    write_csv(all_results, Path(args.csv))

    print(f"\nFiles saved:")
    print(f"  Report: {args.output}")
    print(f"  CSV:    {args.csv}")


if __name__ == "__main__":
    main()
