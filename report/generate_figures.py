"""
Generate all thesis figures for the Vietnamese HR RAG chatbot thesis.
Run: python generate_figures.py
Output: figures/*.png (overwrites existing)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Output directory
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Common styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# Color palette
COLORS = {
    'blue_dark': '#1a365d',
    'blue_mid': '#2b6cb0',
    'blue_light': '#4299e1',
    'teal': '#0d7377',
    'teal_dark': '#065f63',
    'orange': '#dd6b20',
    'orange_light': '#ed8936',
    'green': '#276749',
    'green_light': '#48bb78',
    'gray': '#4a5568',
    'gray_light': '#a0aec0',
    'red': '#e53e3e',
    'bg_light': '#f7fafc',
    'bg_blue': '#ebf4ff',
    'bg_cream': '#fffff0',
    'white': '#ffffff',
}


def draw_box(ax, x, y, w, h, text, subtext='', color='#2b6cb0',
             text_color='white', fontsize=11, subsize=9, radius=0.02):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor='none',
                         zorder=3)
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.012, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
        ax.text(x, y - 0.018, subtext, ha='center', va='center',
                fontsize=subsize, fontstyle='italic', color=text_color,
                alpha=0.85, zorder=4)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color='#2d3748', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=2)


# ============================================================
# FIGURE 1: architecture.png
# ============================================================
def gen_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.97, 'End-to-End RAG Pipeline Architecture',
            ha='center', va='top', fontsize=18, fontweight='bold',
            color=COLORS['blue_dark'])

    # --- OFFLINE section ---
    offline_box = FancyBboxPatch((0.03, 0.78), 0.94, 0.15,
                                 boxstyle="round,pad=0.01",
                                 facecolor=COLORS['bg_blue'], edgecolor=COLORS['blue_mid'],
                                 linestyle='--', linewidth=1.5, zorder=1)
    ax.add_patch(offline_box)
    ax.text(0.06, 0.915, 'OFFLINE (Indexing)', fontsize=12, fontweight='bold',
            color=COLORS['orange'], zorder=4)

    # Offline boxes
    bw, bh = 0.18, 0.065
    offline_y = 0.84
    boxes_offline = [
        (0.14, 'HR Documents', '20 Vietnamese texts', COLORS['teal_dark']),
        (0.36, 'Text Chunker', '600 chars / 100 overlap', COLORS['gray']),
        (0.58, 'Multilingual Embedder', 'paraphrase-MiniLM-L12-v2', COLORS['orange']),
        (0.82, 'ChromaDB', '384-dim | cosine', COLORS['teal_dark']),
    ]
    for x, t, st, c in boxes_offline:
        draw_box(ax, x, offline_y, bw, bh, t, st, c, fontsize=10, subsize=8)

    # Offline arrows
    for i in range(len(boxes_offline) - 1):
        x1 = boxes_offline[i][0] + bw/2
        x2 = boxes_offline[i+1][0] - bw/2
        draw_arrow(ax, x1, offline_y, x2, offline_y)

    # --- ONLINE section ---
    online_box = FancyBboxPatch((0.03, 0.04), 0.94, 0.70,
                                 boxstyle="round,pad=0.01",
                                 facecolor='#fefcf3', edgecolor=COLORS['orange'],
                                 linestyle='--', linewidth=1.5, zorder=1)
    ax.add_patch(online_box)
    ax.text(0.06, 0.715, 'ONLINE (Inference)', fontsize=12, fontweight='bold',
            color=COLORS['orange'], zorder=4)

    # User Question
    draw_box(ax, 0.5, 0.65, 0.22, 0.06, 'User Question',
             'Vietnamese natural language query', COLORS['orange'], fontsize=11, subsize=8)

    # Query Embedder
    draw_box(ax, 0.5, 0.55, 0.22, 0.06, 'Query Embedder',
             'same multilingual model as indexing', COLORS['orange_light'], fontsize=11, subsize=8)
    draw_arrow(ax, 0.5, 0.62, 0.5, 0.58)

    # BM25 Search
    draw_box(ax, 0.28, 0.44, 0.20, 0.06, 'BM25 Search',
             'BM25Okapi + tokenize_vi()', COLORS['gray'], fontsize=10, subsize=8)

    # Vector Search
    draw_box(ax, 0.72, 0.44, 0.20, 0.06, 'Vector Search',
             'ChromaDB cosine similarity', COLORS['teal'], fontsize=10, subsize=8)

    # Arrows from Embedder to both searches
    draw_arrow(ax, 0.42, 0.52, 0.32, 0.47)
    draw_arrow(ax, 0.58, 0.52, 0.68, 0.47)

    # Arrow from ChromaDB to Vector Search (curved connection)
    ax.annotate('', xy=(0.82, 0.47), xytext=(0.82, 0.805),
                arrowprops=dict(arrowstyle='->', color=COLORS['teal'],
                                connectionstyle='arc3,rad=0.3', lw=1.2, linestyle='--'),
                zorder=2)

    # RRF
    draw_box(ax, 0.5, 0.32, 0.30, 0.06, 'Reciprocal Rank Fusion (RRF)',
             'score = α·rrf_vec + (1−α)·rrf_bm25  |  k=60',
             COLORS['blue_mid'], fontsize=11, subsize=8)
    draw_arrow(ax, 0.32, 0.41, 0.42, 0.35)
    draw_arrow(ax, 0.68, 0.41, 0.58, 0.35)

    # Phi-3-Mini
    draw_box(ax, 0.5, 0.20, 0.28, 0.06, 'Phi-3-Mini HR (Q4_K_M)',
             'llama-cpp-python  |  top-k context, default = 3',
             COLORS['blue_dark'], fontsize=11, subsize=8)
    draw_arrow(ax, 0.5, 0.29, 0.5, 0.23)

    # Output
    draw_box(ax, 0.5, 0.09, 0.28, 0.055, 'Vietnamese Answer + Source Attribution',
             '', COLORS['green'], fontsize=11)
    draw_arrow(ax, 0.5, 0.17, 0.5, 0.115)

    fig.savefig(os.path.join(FIG_DIR, 'architecture.png'), facecolor='white')
    plt.close(fig)
    print('✓ architecture.png')


# ============================================================
# FIGURE 2: kb_topics.png
# ============================================================
def gen_kb_topics():
    topics_manual = [
        'Leave Policy', 'Employment Contract', 'Salary & Payroll',
        'Company Regulations', 'Labour Discipline', 'Employee Benefits',
        'Recruitment & Training', 'Social Insurance',
    ]
    topics_ai = [
        'Overtime Policy', 'Maternity / Parental Leave', 'Remote Work Policy',
        'Business Travel', 'Performance Review', 'IT Security',
        'Special Public Holidays', 'Anti-Harassment',
        'Resignation & Handover', 'Anti-Corruption',
        'Workplace Safety', 'Allowances & Subsidies',
    ]

    all_topics = topics_manual + topics_ai
    colors = [COLORS['teal']] * len(topics_manual) + [COLORS['blue_mid']] * len(topics_ai)
    labels_right = ['Manual'] * len(topics_manual) + ['AI-Generated'] * len(topics_ai)

    fig, ax = plt.subplots(figsize=(12, 9))
    y_pos = np.arange(len(all_topics))
    bars = ax.barh(y_pos, [1]*len(all_topics), color=colors, height=0.7, edgecolor='none')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_topics, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.35)
    ax.set_xticks([])

    # Add labels on the right
    for i, (bar, label) in enumerate(zip(bars, labels_right)):
        color = COLORS['teal_dark'] if label == 'Manual' else COLORS['blue_dark']
        ax.text(1.03, i, label, va='center', fontsize=9, fontstyle='italic', color=color)

    ax.set_title('Vietnamese HR Knowledge Base — 20 Documents by Topic',
                 fontsize=15, fontweight='bold', pad=15)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS['teal'], label='Manually authored (8)'),
        mpatches.Patch(color=COLORS['blue_mid'], label='AI-generated via Minimax-M2.7 (12)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10, framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'kb_topics.png'), facecolor='white')
    plt.close(fig)
    print('✓ kb_topics.png')


# ============================================================
# FIGURE 3: retrieval_flowchart.png
# ============================================================
def gen_retrieval_flowchart():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.97, 'Hybrid BM25 + Vector Retrieval with Reciprocal Rank Fusion',
            ha='center', va='top', fontsize=16, fontweight='bold',
            color=COLORS['blue_dark'])

    bw = 0.22
    bh = 0.065

    # User Query
    draw_box(ax, 0.5, 0.88, 0.24, 0.06, 'User Query',
             'Vietnamese natural language', COLORS['orange'], fontsize=12, subsize=9)

    # Branch labels
    ax.text(0.25, 0.80, 'BM25 Branch (Lexical)', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['orange'])
    ax.text(0.75, 0.80, 'Vector Branch (Semantic)', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['teal'])

    # Arrows from query to branches
    draw_arrow(ax, 0.42, 0.85, 0.28, 0.78)
    draw_arrow(ax, 0.58, 0.85, 0.72, 0.78)

    # BM25 branch
    draw_box(ax, 0.25, 0.73, bw, bh, 'tokenize_vi(query)',
             'lowercase + remove punct + split', COLORS['gray'], fontsize=10, subsize=8)
    draw_box(ax, 0.25, 0.60, bw, bh, 'BM25Okapi.get_scores()',
             'score each of N corpus chunks', COLORS['gray'], fontsize=10, subsize=8)
    draw_box(ax, 0.25, 0.47, bw, bh, 'BM25 Candidate List',
             'fetch_k = 4 × top_k', COLORS['teal'], fontsize=10, subsize=8)
    draw_arrow(ax, 0.25, 0.695, 0.25, 0.635)
    draw_arrow(ax, 0.25, 0.565, 0.25, 0.505)

    # Vector branch
    draw_box(ax, 0.75, 0.73, bw, bh, 'Embed Query',
             'paraphrase-multilingual-MiniLM-L12-v2', COLORS['teal'], fontsize=10, subsize=8)
    draw_box(ax, 0.75, 0.60, bw, bh, 'ChromaDB Cosine Search',
             'approximate nearest neighbours', COLORS['teal'], fontsize=10, subsize=8)
    draw_box(ax, 0.75, 0.47, bw, bh, 'Vector Candidate List',
             'fetch_k = 4 × top_k', COLORS['teal'], fontsize=10, subsize=8)
    draw_arrow(ax, 0.75, 0.695, 0.75, 0.635)
    draw_arrow(ax, 0.75, 0.565, 0.75, 0.505)

    # Dotted divider
    ax.plot([0.5, 0.5], [0.42, 0.78], color=COLORS['gray_light'],
            linestyle=':', linewidth=1.5, zorder=1)

    # RRF box (larger)
    draw_box(ax, 0.5, 0.32, 0.42, 0.075, 'Reciprocal Rank Fusion (RRF)',
             'score(d) = α/(k+rv+1) + (1−α)/(k+rb+1)    k=60, α=0.5',
             COLORS['blue_mid'], fontsize=12, subsize=9)
    draw_arrow(ax, 0.30, 0.435, 0.40, 0.36)
    draw_arrow(ax, 0.70, 0.435, 0.60, 0.36)

    # Output
    draw_box(ax, 0.5, 0.17, 0.30, 0.065, 'Top-k Context Chunks (default k=3)',
             '', '#e8f5e9', text_color=COLORS['teal_dark'], fontsize=11)
    draw_arrow(ax, 0.5, 0.28, 0.5, 0.205)

    fig.savefig(os.path.join(FIG_DIR, 'retrieval_flowchart.png'), facecolor='white')
    plt.close(fig)
    print('✓ retrieval_flowchart.png')


# ============================================================
# FIGURE 4: retrieval_comparison.png
# ============================================================
def gen_retrieval_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={'width_ratios': [2.2, 1]})

    # --- Left panel: per-query ---
    queries_vi = [
        '1 tháng\nnghỉ bao ngày',
        'Nghỉ thai\nsản bao lâu',
        'Lương\ntháng 13',
        'Vi phạm\nnội quy',
        'Bảo hiểm\ny tế',
        'Làm thêm\ngiờ',
        'Thử việc\nbao lâu',
        'Phúc lợi\nnhân viên',
        'Quy trình\nnghỉ việc',
        'BHXH\n% đóng',
    ]

    n = len(queries_vi)
    x = np.arange(n)
    width = 0.35

    # Dense: 6 correct, 4 incorrect (queries 0,2,3,8 wrong based on thesis)
    dense_correct = [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]  # 6 correct
    hybrid_correct = [1]*10  # all correct

    bars1 = ax1.bar(x - width/2, [1]*n, width, color=COLORS['orange_light'],
                    edgecolor='none', label='Pure Vector', alpha=0.85)
    bars2 = ax1.bar(x + width/2, [1]*n, width, color=COLORS['blue_light'],
                    edgecolor='none', label='Hybrid BM25+Vector (RRF)', alpha=0.85)

    # Mark correct/incorrect with symbols
    for i in range(n):
        # Dense result
        if dense_correct[i]:
            ax1.text(i - width/2, 0.05, 'O', ha='center', va='bottom',
                    fontsize=13, color=COLORS['green'], fontweight='bold',
                    fontfamily='DejaVu Sans')
        else:
            ax1.text(i - width/2, 0.05, 'X', ha='center', va='bottom',
                    fontsize=13, color=COLORS['red'], fontweight='bold',
                    fontfamily='DejaVu Sans')
        # Hybrid result - all correct
        ax1.text(i + width/2, 0.05, 'O', ha='center', va='bottom',
                fontsize=13, color=COLORS['green'], fontweight='bold',
                fontfamily='DejaVu Sans')

    ax1.set_xticks(x)
    ax1.set_xticklabels(queries_vi, fontsize=8.5, ha='center')
    ax1.set_ylim(0, 1.3)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Incorrect', 'Correct'], fontsize=10)
    ax1.set_ylabel('Retrieval Result', fontsize=11)
    ax1.set_title('Per-Query Retrieval Correctness\n(Top-1 Source Document)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper center')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Right panel: overall accuracy ---
    methods = ['Pure\nVector', 'Hybrid\nBM25+Vector\n(RRF)']
    accuracies = [60, 100]
    bar_colors = [COLORS['orange_light'], COLORS['blue_light']]

    bars = ax2.bar(methods, accuracies, color=bar_colors, width=0.6,
                   edgecolor='none', alpha=0.85)

    # Add percentage labels
    for bar, acc in zip(bars, accuracies):
        color = COLORS['blue_dark'] if acc == 100 else COLORS['orange']
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc}%', ha='center', va='bottom', fontsize=14,
                fontweight='bold', color=color)

    # Improvement arrow
    ax2.annotate('+40%', xy=(1, 95), xytext=(0.3, 78),
                fontsize=12, fontweight='bold', color=COLORS['teal'],
                arrowprops=dict(arrowstyle='->', color=COLORS['teal'], lw=2))

    ax2.set_ylim(0, 115)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Overall\nAccuracy', fontsize=13, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Retrieval Accuracy: Pure Vector vs. Hybrid BM25+Vector with RRF',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()

    fig.savefig(os.path.join(FIG_DIR, 'retrieval_comparison.png'), facecolor='white')
    plt.close(fig)
    print('✓ retrieval_comparison.png')


# ============================================================
# FIGURE 5: cli_screenshot.png
# ============================================================
def gen_cli_screenshot():
    """Generate a styled terminal screenshot of the chatbot in action."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Terminal background
    terminal = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                               boxstyle="round,pad=0.015",
                               facecolor='#1a1a2e', edgecolor='#333355',
                               linewidth=2, zorder=1)
    ax.add_patch(terminal)

    # Terminal title bar
    title_bar = FancyBboxPatch((0.02, 0.88), 0.96, 0.10,
                                boxstyle="round,pad=0.01",
                                facecolor='#16213e', edgecolor='none', zorder=2)
    ax.add_patch(title_bar)

    # Window buttons
    for i, color in enumerate(['#ff5f57', '#febc2e', '#28c840']):
        circle = plt.Circle((0.06 + i*0.025, 0.93), 0.008,
                            color=color, zorder=3)
        ax.add_patch(circle)

    ax.text(0.5, 0.93, 'Vietnamese HR RAG Chatbot — Local Terminal',
            ha='center', va='center', fontsize=10, color='#8892b0',
            fontfamily='monospace', zorder=3)

    # Terminal content
    mono = {'fontfamily': 'Consolas', 'fontsize': 9.5, 'zorder': 4}
    lines = [
        ('$ python cli_demo.py models/phi-3-mini.gguf', '#48bb78'),
        ('  Vector store: 258 documents loaded', '#8892b0'),
        ('  Model: phi-3-mini.gguf | Retrieval: hybrid BM25+vector RRF', '#8892b0'),
        ('', ''),
        ('You: Mức lương cơ bản là bao nhiêu', '#48bb78'),
        ('  INFO: BM25 index ready: 258 chunks indexed.', '#4299e1'),
        ('', ''),
        ('Bot: Mức lương cơ bản bao gồm:', '#e2e8f0'),
        ('  - Nhân viên mới / Thực tập sinh: 8.000.000 – 12.000.000 VNĐ/tháng', '#e2e8f0'),
        ('  - Nhân viên cơ bản (dưới 2 năm): 12.000.000 – 18.000.000 VNĐ/tháng', '#e2e8f0'),
        ('  - Chuyên viên cấp cao (trên 5 năm): 30.000.000 – 50.000.000 VNĐ/tháng', '#e2e8f0'),
        ('  - Quản lý (Manager): 50.000.000 – 80.000.000 VNĐ/tháng', '#e2e8f0'),
        ('', ''),
        ('Source: page 0 (chinh_sach_luong.txt), page 0 (bao_hiem_xa_hoi.txt)', '#ed8936'),
        ('Time — Retrieval: 0.07s | Generation: 115.75s | Total: 115.81s', '#a0aec0'),
    ]

    y_start = 0.84
    line_height = 0.048
    for i, (text, color) in enumerate(lines):
        if text:
            ax.text(0.05, y_start - i * line_height, text,
                    va='top', color=color, **mono)

    fig.savefig(os.path.join(FIG_DIR, 'cli_screenshot.png'), facecolor='#0f0f23')
    plt.close(fig)
    print('✓ cli_screenshot.png')


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print('Generating thesis figures...')
    gen_architecture()
    gen_kb_topics()
    gen_retrieval_flowchart()
    gen_retrieval_comparison()
    gen_cli_screenshot()
    print('\nAll 5 figures generated in figures/')
