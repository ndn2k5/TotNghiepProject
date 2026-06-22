#!/bin/bash
# ============================================================
# Vast.ai H200 — Full Pipeline Setup Script
# Budget: ~$1.50 (20-30 min on H200 141GB VRAM)
#
# USAGE: In Vast.ai web terminal:
#   git clone --depth 1 https://github.com/ndn2k5/TotNghiepProject.git
#   cd TotNghiepProject
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
export PIP_ROOT_USER_ACTION=ignore

echo "=========================================="
echo "  VAST.AI H200 PIPELINE — mode: ${MODE}"
echo "  $(date)"
echo "=========================================="

# ── Step 1: Pre-flight checks ──────────────────────────────
echo ""
echo "[1/6] Pre-flight checks..."

# Disk
DISK_FREE_GB=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
echo "  Disk free: ${DISK_FREE_GB}GB"
if [ "$DISK_FREE_GB" -lt 50 ]; then
    echo "  [!!!] NEED 50GB+ disk. You have ${DISK_FREE_GB}GB. Resize on Vast.ai!"
fi

# CUDA driver version — vLLM 0.23+ needs CUDA >= 12.4
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" || echo "unknown")
echo "  CUDA driver: ${CUDA_VER}"
if [ "$CUDA_VER" != "unknown" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 4 ]); then
        echo ""
        echo "  [!!!] CUDA ${CUDA_VER} TOO OLD — vLLM needs CUDA >= 12.4"
        echo "  [!!!] KILL THIS INSTANCE. Rent one with CUDA >= 12.4."
        echo "  [!!!] On Vast.ai, filter by 'CUDA Version >= 12.4'"
        echo ""
        echo "  (If you want to try anyway: pip install vllm==0.4.2)"
        exit 1
    fi
fi
echo "  Pre-flight OK ✓"

# ── Step 2: Install deps ───────────────────────────────────
# Vast.ai images have torch/transformers pre-installed.
# Only install what's ACTUALLY missing. One package at a time
# so pip doesn't freeze resolving a giant dep tree.
echo ""
echo "[2/6] Installing dependencies..."

apt-get update -qq && apt-get install -y -qq zip > /dev/null 2>&1 || true

if [ "$MODE" = "generate" ] || [ "$MODE" = "all" ]; then
    echo "  Installing vLLM (may take 1-2 min)..."
    pip install vllm --no-build-isolation 2>&1 | tail -5
    echo "  vLLM done ✓"
fi

if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    echo "  Installing unsloth..."
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -3
    echo "  Installing trl + peft + bitsandbytes..."
    pip install "trl>=0.8.0" 2>&1 | tail -2
    pip install "peft>=0.9.0" 2>&1 | tail -2
    pip install "bitsandbytes>=0.41.0" 2>&1 | tail -2
    echo "  Fine-tune deps done ✓"
fi

# datasets is small, always needed for finetune
if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    pip install "datasets>=2.16.0" "sentencepiece" "protobuf" "accelerate" 2>&1 | tail -2
fi

echo "  All deps installed ✓"

# ── Step 3: Verify ─────────────────────────────────────────
echo ""
echo "[3/6] Verifying setup..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not found"
python3 -c "import torch; print(f'  torch {torch.__version__} CUDA={torch.cuda.is_available()} GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>/dev/null || true

if [ "$MODE" = "generate" ] || [ "$MODE" = "all" ]; then
    python3 -c "import vllm; print(f'  vllm {vllm.__version__} OK')" || echo "  [!] vllm import FAILED"
fi
if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    python3 -c "import unsloth; print('  unsloth OK')" || echo "  [!] unsloth import FAILED"
    python3 -c "import trl; print(f'  trl {trl.__version__} OK')" || echo "  [!] trl import FAILED"
fi

# ── Step 4: Generate Q&A ───────────────────────────────────
if [ "$MODE" = "generate" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[4/6] Generating Q&A (Qwen2.5-72B-Instruct-AWQ)..."
    echo "  Target: ~3000 pairs from 20 docs"
    echo ""

    python3 scripts/vast_generate_qa.py \
        --batches 10 \
        --per-batch 15 \
        --output data/generated_qa_h200.jsonl \
        --resume

    echo "  Generation done ✓"
    python3 scripts/validate_qa.py --input data/generated_qa_h200.jsonl || true
fi

# ── Step 5: Fine-tune ──────────────────────────────────────
if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[5/6] QLoRA fine-tuning (Phi-3-Mini)..."

    DATA_FILE="data/generated_qa_h200.jsonl"
    if [ ! -f "$DATA_FILE" ]; then
        DATA_FILE="data/qa_training_data.csv"
        echo "  WARNING: using fallback $DATA_FILE"
    fi

    python3 scripts/vast_finetune.py \
        --data "$DATA_FILE" \
        --epochs 3 \
        --batch-size 8 \
        --grad-accum 2

    echo "  Fine-tuning done ✓"

    echo "  Running eval..."
    python3 scripts/vast_eval.py --adapter models/phi3-mini-hr-finetuned || true
fi

# ── Step 6: Package ────────────────────────────────────────
echo ""
echo "[6/6] Packaging into zip..."
bash scripts/vast_package.sh

# ── Done ───────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "  DONE in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "=========================================="
echo ""
echo "DOWNLOAD: vast_results.zip"
echo ""
echo "!!!  TERMINATE INSTANCE NOW  !!!"
echo ""
