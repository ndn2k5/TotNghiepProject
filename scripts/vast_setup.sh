#!/bin/bash
# ============================================================
# Vast.ai H200 — Full Pipeline Setup Script
# Budget: ~$1.50 (20-30 min on H200 141GB VRAM)
#
# USAGE: Copy-paste into Vast.ai SSH terminal, then run:
#   bash scripts/vast_setup.sh [mode]
#
# MODES:
#   generate         — only generate Q&A data (default, ~10 min)
#   finetune         — only fine-tune (use existing data)
#   all              — generate then fine-tune (~25 min)
#
# EXIT IMMEDIATELY after done — every minute costs money!
# ============================================================

set -euo pipefail

MODE="${1:-generate}"
START_TIME=$(date +%s)

echo "=========================================="
echo "  VAST.AI H200 PIPELINE — mode: ${MODE}"
echo "  $(date)"
echo "=========================================="

# ── Step 1: Clone repo ──────────────────────────────────────
echo ""
echo "[1/5] Cloning project repo..."
if [ ! -d "TotNghiepProject" ]; then
    git clone --depth 1 https://github.com/ndn2k5/TotNghiepProject.git
else
    echo "  Repo already cloned, pulling latest..."
    git -C TotNghiepProject pull --ff-only
fi
cd TotNghiepProject
echo "  Repo ready ✓"

# ── Step 2: Install dependencies ────────────────────────────
echo ""
echo "[2/5] Installing dependencies..."

if [ "$MODE" = "generate" ] || [ "$MODE" = "all" ]; then
    echo "  Installing vLLM (for Q&A generation)..."
    pip install vllm --quiet 2>&1 | tail -3
fi

if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    echo "  Installing Unsloth + TRL (for fine-tuning)..."
    # Unsloth nightly for H200 Blackwell (sm90+) support
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet 2>&1 | tail -3
    pip install "trl>=0.8.0" "peft>=0.9.0" "transformers>=4.40.0" --quiet 2>&1 | tail -3
fi

echo "  Dependencies installed ✓"

# ── Step 3: Verify GPU ───────────────────────────────────────
echo ""
echo "[3/5] GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>/dev/null || true

# ── Step 4: Run generation (if requested) ───────────────────
if [ "$MODE" = "generate" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[4/5] Running Q&A generation (Qwen2.5-72B-Instruct-AWQ)..."
    echo "  Target: ~3000 Vietnamese HR Q&A pairs from 20 docs"
    echo "  (10 batches × 15 pairs × 20 docs)"
    echo "  This takes ~10-15 min on H200..."
    echo ""

    python3 scripts/vast_generate_qa.py \
        --batches 10 \
        --per-batch 15 \
        --output data/generated_qa_h200.jsonl \
        --resume

    echo ""
    echo "  Generation done ✓"
    echo "  Running validation..."
    python3 scripts/validate_qa.py --input data/generated_qa_h200.jsonl
fi

# ── Step 5: Run fine-tuning (if requested) ──────────────────
if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[5/5] Running QLoRA fine-tuning (Phi-3-Mini)..."
    echo "  This takes ~5-10 min on H200..."
    echo ""

    # Use generated data if available, otherwise fall back to original
    DATA_FILE="data/generated_qa_h200.jsonl"
    if [ ! -f "$DATA_FILE" ]; then
        DATA_FILE="data/qa_training_data.csv"
        echo "  WARNING: generated_qa_h200.jsonl not found, using $DATA_FILE"
    fi

    python3 scripts/vast_finetune.py \
        --data "$DATA_FILE" \
        --epochs 3 \
        --batch-size 8 \
        --grad-accum 2

    echo ""
    echo "  Fine-tuning done ✓"
fi

# ── Step 6: Eval (if fine-tuned) ──────────────────────────
if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[6/7] Running evaluation (base vs fine-tuned)..."
    python3 scripts/vast_eval.py --adapter models/phi3-mini-hr-finetuned || true
    echo "  Eval done"
fi

# ── Step 7: Package everything into one zip ───────────────
echo ""
echo "[7/7] Packaging outputs into single zip..."
bash scripts/vast_package.sh

# ── Done ────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "  COMPLETE in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "=========================================="
echo ""
echo "DOWNLOAD ONE FILE:"
echo "  vast_results.zip"
echo ""
echo "  (contains: data, model GGUF, adapter, eval report, scores CSV)"
echo ""
echo "!!!  TERMINATE THIS INSTANCE NOW  !!!"
echo "   vast destroy \$(cat /vast_instance_id 2>/dev/null || echo '<instance-id>')"
echo ""
