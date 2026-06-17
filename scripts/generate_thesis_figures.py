#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all figures required by report/thesis.tex.

Run from project root:
    python scripts/generate_thesis_figures.py

Output files in report/figures/:
    usth_logo.png           -- USTH logo placeholder (circular seal)
    architecture.png        -- End-to-end RAG pipeline diagram
    retrieval_flowchart.png -- Hybrid BM25+Vector RRF flowchart
    kb_topics.png           -- Knowledge base topic distribution bar chart
    retrieval_comparison.png-- Pure vector vs hybrid accuracy comparison
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)

DPI = 200
FONT = "DejaVu Sans"


# ── Colour palette ────────────────────────────────────────────────────
C_BLUE   = "#1565C0"
C_LBLUE  = "#42A5F5"
C_TEAL   = "#00695C"
C_LTEAL  = "#4DB6AC"
C_ORANGE = "#E65100"
C_LORANGE= "#FFB74D"
C_GREY   = "#546E7A"
C_LGREY  = "#ECEFF1"
C_WHITE  = "#FFFFFF"
C_DARK   = "#212121"


# ════════════════════════════════════════════════════════════════════════
# 1. USTH LOGO PLACEHOLDER
# ════════════════════════════════════════════════════════════════════════
def make_usth_logo():
    """
    Approximate the actual USTH logo:
      - Blue 'USTH' text (large, bold)
      - Wing / leaf shape above-right of the 'Y' (red left wing, blue right wing)
      - 'VIETNAM FRANCE UNIVERSITY' in small blue text below
    This is a close approximation for thesis use; replace with official logo if available.
    """
    from matplotlib.patches import Wedge, Polygon
    from matplotlib.path import Path as MPath
    import matplotlib.patches as mpatch

    fig, ax = plt.subplots(figsize=(3.5, 2.0), facecolor="white")
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2.0)
    ax.axis("off")

    # ── 'USTH' bold blue text ─────────────────────────────────────────
    ax.text(1.30, 0.90, "USTH", fontsize=46, fontweight="bold",
            color=C_BLUE, ha="center", va="center", zorder=3,
            fontfamily="DejaVu Sans")

    # ── Wing design above the 'Y' (upper-right of USTH) ──────────────
    # Red left wing — swept upward-left shape
    red_wing = mpatch.FancyArrowPatch(
        posA=(1.72, 1.22), posB=(1.35, 1.75),
        arrowstyle=mpatch.ArrowStyle("simple", head_width=0, tail_width=18),
        color="#CC0000", zorder=4,
        connectionstyle="arc3,rad=-0.35",
    )
    # Use bezier polygons instead for more faithful wing shape
    # Red wing (left, curves up-left)
    red_pts = np.array([
        [1.60, 1.18],
        [1.38, 1.55],
        [1.22, 1.82],
        [1.42, 1.65],
        [1.58, 1.38],
        [1.75, 1.22],
    ])
    red_wing_poly = mpatch.Polygon(red_pts, closed=True,
                                   facecolor="#CC0000", edgecolor="none", zorder=4)
    ax.add_patch(red_wing_poly)

    # Blue wing (right, curves up-right)
    blue_pts = np.array([
        [1.72, 1.18],
        [1.95, 1.50],
        [2.10, 1.82],
        [1.90, 1.65],
        [1.80, 1.40],
        [1.62, 1.20],
    ])
    blue_wing_poly = mpatch.Polygon(blue_pts, closed=True,
                                    facecolor=C_BLUE, edgecolor="none", zorder=4)
    ax.add_patch(blue_wing_poly)

    # Small white gap between wings (centre spine)
    spine_pts = np.array([
        [1.65, 1.20],
        [1.66, 1.55],
        [1.68, 1.80],
        [1.70, 1.55],
        [1.72, 1.20],
    ])
    spine = mpatch.Polygon(spine_pts, closed=True,
                           facecolor=C_WHITE, edgecolor="none", zorder=5)
    ax.add_patch(spine)

    # ── 'VIETNAM FRANCE UNIVERSITY' subtitle ─────────────────────────
    ax.text(1.30, 0.34, "VIETNAM FRANCE UNIVERSITY",
            fontsize=7, color=C_BLUE, ha="center", va="center",
            fontweight="normal")

    path = OUT / "usth_logo.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path}")


# ════════════════════════════════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE DIAGRAM
# ════════════════════════════════════════════════════════════════════════
def _box(ax, x, y, w, h, label, sublabel="", color=C_BLUE, fontsize=9):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.04",
                         facecolor=color, edgecolor=C_DARK,
                         linewidth=0.8, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.08 if sublabel else 0), label,
            fontsize=fontsize, fontweight="bold", color=C_WHITE,
            ha="center", va="center", zorder=4)
    if sublabel:
        ax.text(x, y - 0.14, sublabel,
                fontsize=6.5, color="#CFD8DC",
                ha="center", va="center", zorder=4, style="italic")


def _arrow(ax, x1, y1, x2, y2, color=C_DARK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.2, mutation_scale=12),
                zorder=5)


def make_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=C_LGREY)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Title
    ax.text(5.5, 6.15, "End-to-End RAG Pipeline Architecture",
            fontsize=13, fontweight="bold", color=C_DARK,
            ha="center", va="center")

    # ── OFFLINE phase (left side) ─────────────────────────────────
    offline_bg = FancyBboxPatch((0.15, 2.2), 3.0, 3.9,
                                boxstyle="round,pad=0.1",
                                facecolor="#E3F2FD", edgecolor=C_BLUE,
                                linewidth=1.2, linestyle="--", zorder=1)
    ax.add_patch(offline_bg)
    ax.text(1.65, 5.85, "OFFLINE  (Indexing)", fontsize=8,
            color=C_BLUE, ha="center", fontweight="bold")

    _box(ax, 1.65, 5.30, 2.6, 0.52, "HR Documents (×20)",
         "20 Vietnamese policy texts", color=C_TEAL, fontsize=8)
    _arrow(ax, 1.65, 5.04, 1.65, 4.56)

    _box(ax, 1.65, 4.30, 2.6, 0.52, "Text Chunker",
         "600 chars / 100 overlap", color=C_TEAL, fontsize=8)
    _arrow(ax, 1.65, 4.04, 1.65, 3.56)

    _box(ax, 1.65, 3.30, 2.6, 0.52, "Multilingual Embedder",
         "paraphrase-multilingual-MiniLM-L12-v2", color=C_TEAL, fontsize=8)
    _arrow(ax, 1.65, 3.04, 1.65, 2.56)

    _box(ax, 1.65, 2.30, 2.6, 0.52, "ChromaDB  (Vector Store)",
         "./chroma_db  |  384-dim  |  cosine", color="#1B5E20", fontsize=8)

    # ── ONLINE phase (right side) ─────────────────────────────────
    online_bg = FancyBboxPatch((3.5, 0.3), 7.3, 5.8,
                               boxstyle="round,pad=0.1",
                               facecolor="#FFF8E1", edgecolor=C_ORANGE,
                               linewidth=1.2, linestyle="--", zorder=1)
    ax.add_patch(online_bg)
    ax.text(7.15, 5.85, "ONLINE  (Inference)", fontsize=8,
            color=C_ORANGE, ha="center", fontweight="bold")

    # User query
    _box(ax, 5.5, 5.30, 2.6, 0.52, "User Question",
         "Vietnamese natural language query", color=C_ORANGE, fontsize=8)

    _arrow(ax, 5.5, 5.04, 5.5, 4.56)

    _box(ax, 5.5, 4.30, 2.6, 0.52, "Query Embedder",
         "same model as indexing", color=C_ORANGE, fontsize=8)

    # Split to BM25 and Vector
    _arrow(ax, 4.3, 4.04, 3.9, 3.56)   # left branch → BM25
    _arrow(ax, 6.7, 4.04, 7.1, 3.56)   # right branch → Vector

    _box(ax, 3.6, 3.30, 2.2, 0.52, "BM25 Search",
         "BM25Okapi + tokenize_vi()", color=C_GREY, fontsize=8)
    _box(ax, 7.4, 3.30, 2.2, 0.52, "Vector Search",
         "ChromaDB cosine similarity", color=C_GREY, fontsize=8)

    # Merge to RRF
    _arrow(ax, 3.6, 3.04, 4.8, 2.56)
    _arrow(ax, 7.4, 3.04, 6.2, 2.56)

    _box(ax, 5.5, 2.30, 3.2, 0.52,
         "Reciprocal Rank Fusion (RRF)",
         r"score = α·rrf_vector + (1−α)·rrf_bm25   k=60, α=0.5",
         color=C_BLUE, fontsize=8)

    _arrow(ax, 5.5, 2.04, 5.5, 1.56)

    _box(ax, 5.5, 1.30, 3.2, 0.52, "Phi-3-Mini  (GGUF Q4)",
         "llama-cpp-python  |  <|user|>...<|end|>  |  top-5 chunks",
         color=C_BLUE, fontsize=8)

    _arrow(ax, 5.5, 1.04, 5.5, 0.60)

    # Answer
    ans_box = FancyBboxPatch((3.9, 0.35), 3.2, 0.46,
                             boxstyle="round,pad=0.04",
                             facecolor="#E8F5E9", edgecolor=C_TEAL,
                             linewidth=1.2, zorder=3)
    ax.add_patch(ans_box)
    ax.text(5.5, 0.58, "Vietnamese Answer  +  Source Attribution",
            fontsize=8.5, fontweight="bold", color=C_TEAL,
            ha="center", va="center", zorder=4)

    # Cross-link: ChromaDB → Vector Search
    ax.annotate("", xy=(7.1, 3.56), xytext=(2.95, 2.30),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL,
                                lw=1.0, linestyle="dashed",
                                mutation_scale=10,
                                connectionstyle="arc3,rad=-0.3"),
                zorder=5)
    ax.text(5.1, 2.85, "query", fontsize=6.5, color=C_TEAL, style="italic")

    path = OUT / "architecture.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C_LGREY)
    plt.close(fig)
    print(f"  [OK] {path}")


# ════════════════════════════════════════════════════════════════════════
# 3. HYBRID RETRIEVAL FLOWCHART
# ════════════════════════════════════════════════════════════════════════
def make_retrieval_flowchart():
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(5.0, 5.25, "Hybrid BM25 + Vector Retrieval with Reciprocal Rank Fusion",
            fontsize=12, fontweight="bold", color=C_DARK,
            ha="center", va="center")

    # Query box (top centre)
    _box(ax, 5.0, 4.65, 3.0, 0.55, "User Query (Vietnamese text)",
         color=C_ORANGE, fontsize=9)

    # Split arrows
    _arrow(ax, 3.7, 4.37, 2.5, 3.87)
    _arrow(ax, 6.3, 4.37, 7.5, 3.87)

    # BM25 branch (left)
    _box(ax, 2.5, 3.60, 2.8, 0.55, "tokenize_vi(query)",
         "lowercase + remove punct + split", color=C_GREY, fontsize=8)
    _arrow(ax, 2.5, 3.33, 2.5, 2.83)
    _box(ax, 2.5, 2.55, 2.8, 0.55, "BM25Okapi.get_scores()",
         "score each of N corpus chunks", color=C_GREY, fontsize=8)
    _arrow(ax, 2.5, 2.27, 2.5, 1.77)
    _box(ax, 2.5, 1.50, 2.8, 0.55, "BM25 Rank List",
         "top-20 by keyword score", color=C_GREY, fontsize=8)

    # Vector branch (right)
    _box(ax, 7.5, 3.60, 2.8, 0.55, "Embed query",
         "paraphrase-multilingual-MiniLM", color=C_TEAL, fontsize=8)
    _arrow(ax, 7.5, 3.33, 7.5, 2.83)
    _box(ax, 7.5, 2.55, 2.8, 0.55, "ChromaDB cosine similarity",
         "approximate nearest neighbours", color=C_TEAL, fontsize=8)
    _arrow(ax, 7.5, 2.27, 7.5, 1.77)
    _box(ax, 7.5, 1.50, 2.8, 0.55, "Vector Rank List",
         "top-20 by cosine distance", color=C_TEAL, fontsize=8)

    # Converge to RRF
    _arrow(ax, 3.85, 1.50, 4.45, 0.97)
    _arrow(ax, 6.15, 1.50, 5.55, 0.97)

    _box(ax, 5.0, 0.72, 4.0, 0.62,
         "Reciprocal Rank Fusion (RRF)",
         "score(d) = α·rrf_vec(d) + (1-α)·rrf_bm25(d)   |   k=60, α=0.5",
         color=C_BLUE, fontsize=9)

    # Result
    ax.annotate("", xy=(5.0, 0.28), xytext=(5.0, 0.44),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.2,
                                mutation_scale=12), zorder=5)
    res_box = FancyBboxPatch((3.3, 0.05), 3.4, 0.38,
                             boxstyle="round,pad=0.04",
                             facecolor="#E8F5E9", edgecolor=C_TEAL,
                             linewidth=1.3, zorder=3)
    ax.add_patch(res_box)
    ax.text(5.0, 0.24, "Top-5 Chunks  (by RRF score)",
            fontsize=9, fontweight="bold", color=C_TEAL,
            ha="center", va="center", zorder=4)

    # Branch labels
    ax.text(2.5, 4.10, "BM25 branch", fontsize=8, color=C_GREY,
            ha="center", style="italic")
    ax.text(7.5, 4.10, "Vector branch", fontsize=8, color=C_TEAL,
            ha="center", style="italic")

    path = OUT / "retrieval_flowchart.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path}")


# ════════════════════════════════════════════════════════════════════════
# 4. KNOWLEDGE BASE TOPIC DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════
def make_kb_topics():
    topics = [
        # (label, category)
        ("Leave Policy",              "Manual"),
        ("Employment Contract",        "Manual"),
        ("Salary & Payroll",           "Manual"),
        ("Company Regulations",        "Manual"),
        ("Labor Discipline",           "Manual"),
        ("Employee Benefits",          "Manual"),
        ("Recruitment & Training",     "Manual"),
        ("Social Insurance",           "Manual"),
        ("Overtime Policy",            "AI-Generated"),
        ("Maternity / Parental Leave", "AI-Generated"),
        ("Remote Work Policy",         "AI-Generated"),
        ("Business Travel",            "AI-Generated"),
        ("Performance Review",         "AI-Generated"),
        ("IT Security",                "AI-Generated"),
        ("Special Public Holidays",    "AI-Generated"),
        ("Anti-Harassment",            "AI-Generated"),
        ("Resignation & Handover",     "AI-Generated"),
        ("Anti-Corruption",            "AI-Generated"),
        ("Workplace Safety",           "AI-Generated"),
        ("Allowances & Subsidies",     "AI-Generated"),
    ]

    labels = [t[0] for t in topics]
    cats   = [t[1] for t in topics]
    colors = [C_TEAL if c == "Manual" else C_BLUE for c in cats]
    y      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    bars = ax.barh(y, [1]*len(labels), color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.6)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.invert_yaxis()
    ax.set_title("Vietnamese HR Knowledge Base — 20 Documents by Topic",
                 fontsize=11, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Category labels on bars
    for i, (bar, cat) in enumerate(zip(bars, cats)):
        ax.text(1.02, i, cat, va="center", ha="left",
                fontsize=8, color=C_DARK, style="italic")

    # Legend
    legend_els = [
        mpatches.Patch(facecolor=C_TEAL, label="Manually authored (8)"),
        mpatches.Patch(facecolor=C_BLUE, label="AI-generated via Minimax-M2.7 (12)"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=8.5,
              framealpha=0.9)

    ax.text(1.55, 19.7, "Total: 20 documents",
            fontsize=8, color=C_GREY, ha="right", style="italic")

    fig.tight_layout()
    path = OUT / "kb_topics.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path}")


# ════════════════════════════════════════════════════════════════════════
# 5. RETRIEVAL ACCURACY COMPARISON
# ════════════════════════════════════════════════════════════════════════
def make_retrieval_comparison():
    # 10-query evaluation results (from REPORT_OUTLINE.md)
    queries = [
        "1 tháng\nnghỉ bao ngày",
        "Nghỉ thai\nsản bao lâu",
        "Lương\ntháng 13",
        "Vi phạm\nnội quy",
        "Bảo hiểm\ny tế",
        "Làm thêm\ngiờ",
        "Thử việc\nbao lâu",
        "Phúc lợi\nnhân viên",
        "Quy trình\nnghỉ việc",
        "BHXH\n% đóng",
    ]
    vector_correct  = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]  # 7/10 = 70%
    hybrid_correct  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 10/10 = 100%

    x = np.arange(len(queries))
    w = 0.35

    fig, (ax_bar, ax_acc) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [3, 1]},
        facecolor="white"
    )

    # Per-query correctness bars
    b1 = ax_bar.bar(x - w/2, vector_correct, w, label="Pure Vector",
                    color=C_LORANGE, edgecolor="white", linewidth=0.6)
    b2 = ax_bar.bar(x + w/2, hybrid_correct, w, label="Hybrid BM25+Vector (RRF)",
                    color=C_LBLUE, edgecolor="white", linewidth=0.6)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(queries, fontsize=7.5)
    ax_bar.set_yticks([0, 1])
    ax_bar.set_yticklabels(["Incorrect", "Correct"], fontsize=9)
    ax_bar.set_ylabel("Retrieval Result", fontsize=10)
    ax_bar.set_title("Per-Query Retrieval Correctness\n(Top-1 Source Document)",
                     fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=8.5)
    ax_bar.set_ylim(0, 1.35)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_facecolor("#FAFAFA")

    # Mark failures
    for i, v in enumerate(vector_correct):
        if v == 0:
            ax_bar.text(i - w/2, 0.06, "✗", ha="center", fontsize=11,
                        color=C_ORANGE, fontweight="bold")
    for i, v in enumerate(hybrid_correct):
        ax_bar.text(i + w/2, 0.06, "✓", ha="center", fontsize=11,
                    color=C_BLUE, fontweight="bold")

    # Overall accuracy summary bar
    accs   = [70, 100]
    labels = ["Pure\nVector", "Hybrid\nBM25+Vector\n(RRF)"]
    col    = [C_LORANGE, C_LBLUE]
    bars   = ax_acc.bar([0, 1], accs, color=col,
                        edgecolor="white", linewidth=0.8, width=0.55)
    ax_acc.set_xticks([0, 1])
    ax_acc.set_xticklabels(labels, fontsize=8.5)
    ax_acc.set_ylim(0, 115)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.set_title("Overall\nAccuracy", fontsize=10, fontweight="bold")
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    ax_acc.set_facecolor("#FAFAFA")
    for bar, acc in zip(bars, accs):
        ax_acc.text(bar.get_x() + bar.get_width()/2,
                    acc + 2, f"{acc}%",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=C_DARK)

    # Improvement annotation
    ax_acc.annotate("", xy=(1, 102), xytext=(0, 72),
                    arrowprops=dict(arrowstyle="-|>", color=C_TEAL,
                                    lw=1.5, mutation_scale=12))
    ax_acc.text(0.5, 88, "+30%", fontsize=10, color=C_TEAL,
                fontweight="bold", ha="center")

    fig.suptitle("Retrieval Accuracy: Pure Vector vs. Hybrid BM25+Vector with RRF",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "retrieval_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path}")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating thesis figures -> report/figures/\n")

    make_usth_logo()
    make_architecture()
    make_retrieval_flowchart()
    make_kb_topics()
    make_retrieval_comparison()

    print(f"\nDone. 5 figures saved to {OUT.resolve()}")
    print("\nStill needed (manual):")
    print("  report/figures/cli_screenshot.png — take a terminal screenshot of cli_demo.py")
