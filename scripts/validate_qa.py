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
        print(f"\n[!] RED-FLAGGED PAIRS ({len(flagged)}):")
        for line_num, q_preview, flag in flagged[:10]:
            print(f"  Line {line_num}: [{flag}] {q_preview}...")

    # Verdict
    relevance_pct = 100 * stats["relevant"] / max(stats["total"], 1)
    print(f"\n{'='*60}")
    if relevance_pct >= 90 and stats["red_flag"] == 0:
        print(f"  [PASS] {relevance_pct:.0f}% domain-relevant, 0 red flags")
    elif relevance_pct >= 80:
        print(f"  [WARN] {relevance_pct:.0f}% relevant, review unclear pairs")
    else:
        print(f"  [FAIL] only {relevance_pct:.0f}% relevant, data needs cleaning")
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
