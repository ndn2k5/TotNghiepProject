# -*- coding: utf-8 -*-
"""Generate chunk size distribution histogram for thesis."""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def chunk_txt_by_paragraph(text, max_chars=300, overlap=50):
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in raw_paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            for sentence in sentences:
                if current and len(current) + len(sentence) + 1 > max_chars:
                    chunks.append(current)
                    if overlap > 0 and current:
                        overlap_text = current[-overlap:].strip()
                        current = (overlap_text + " " + sentence).strip()
                    else:
                        current = sentence
                else:
                    current = (current + " " + sentence).strip() if current else sentence
            if current:
                chunks.append(current)
    return [c for c in chunks if len(c) > 20]


doc_dir = Path("data/viet_labor_docs")
all_lens = []
for f in sorted(doc_dir.glob("*.txt")):
    text = f.read_text(encoding="utf-8")
    chunks = chunk_txt_by_paragraph(text, max_chars=300, overlap=50)
    all_lens.extend([len(c) for c in chunks])

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = range(0, max(all_lens) + 50, 25)
ax.hist(all_lens, bins=bins, color="#4A90D9", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Chunk size (characters)", fontsize=12)
ax.set_ylabel("Number of chunks", fontsize=12)
ax.set_title("Distribution of Chunk Sizes (258 chunks, 20 documents)", fontsize=13)
ax.axvline(x=np.mean(all_lens), color="red", linestyle="--", linewidth=1.5,
           label=f"Mean = {np.mean(all_lens):.0f} chars")
ax.axvline(x=300, color="orange", linestyle=":", linewidth=1.5,
           label="Max chunk size = 300")
ax.legend(fontsize=10)
ax.set_xlim(0, 600)
plt.tight_layout()
out = Path("report/figures/chunk_distribution.png")
plt.savefig(str(out), dpi=300)
print(f"Saved {out} ({len(all_lens)} chunks, mean={np.mean(all_lens):.0f})")
