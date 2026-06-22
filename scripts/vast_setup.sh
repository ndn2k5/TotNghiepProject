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

# ── Step 2: Disk check ──────────────────────────────────────
echo ""
echo "[2/7] Checking disk space..."
DISK_FREE_GB=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
echo "  Free disk: ${DISK_FREE_GB}GB"

if [ "$MODE" = "all" ] && [ "$DISK_FREE_GB" -lt 70 ]; then
    echo "  [!!!] NEED 70GB+ for 'all' mode (model ~40GB + deps ~8GB + output ~5GB)"
    echo "  [!!!] You only have ${DISK_FREE_GB}GB. Resize disk on Vast.ai or use 'generate' mode."
    echo "  Continuing anyway... but may run out of space."
elif [ "$MODE" = "generate" ] && [ "$DISK_FREE_GB" -lt 50 ]; then
    echo "  [!!!] NEED 50GB+ for 'generate' mode (Qwen-72B-AWQ ~40GB)"
    echo "  [!!!] You only have ${DISK_FREE_GB}GB. Resize disk on Vast.ai."
    echo "  Continuing anyway..."
else
    echo "  Disk OK ✓"
fi

# ── Step 3: Install ALL dependencies upfront ────────────────
# Install everything FIRST so no script crashes mid-run from missing libs.
# NO --quiet so you can see what's happening.
echo ""
echo "[3/7] Installing dependencies (all modes)..."
export PIP_ROOT_USER_ACTION=ignore

# --- System packages ---
echo ""
echo "  [3a] System packages (zip)..."
apt-get update -qq && apt-get install -y -qq zip 2>&1 | tail -3 || true

# --- Common Python deps (needed by ALL modes) ---
echo ""
echo "  [3b] Core Python libs..."
pip install --upgrade pip --root-user-action=ignore 2>&1 | tail -2
pip install \
    "torch>=2.1.0" \
    "transformers>=4.40.0" \
    "accelerate>=0.26.0" \
    "datasets>=2.16.0" \
    "sentencepiece>=0.1.99" \
    "protobuf>=4.0" \
    --root-user-action=ignore 2>&1 | tail -5

# --- Generation mode: vLLM ---
if [ "$MODE" = "generate" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "  [3c] vLLM (batched inference)..."
    pip install vllm --root-user-action=ignore 2>&1 | tail -5
fi

# --- Fine-tune + eval mode: Unsloth + TRL + bitsandbytes ---
if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "  [3d] Unsloth + TRL + bitsandbytes..."
    pip install \
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
        "trl>=0.8.0" \
        "peft>=0.9.0" \
        "bitsandbytes>=0.41.0" \
        --root-user-action=ignore 2>&1 | tail -5
fi

# --- Verify critical imports ---
echo "  [2e] Verifying imports..."
IMPORT_FAIL=0
python3 -c "import torch; print(f'    torch {torch.__version__} OK')" || IMPORT_FAIL=1
python3 -c "import transformers; print(f'    transformers {transformers.__version__} OK')" || IMPORT_FAIL=1
python3 -c "import datasets; print(f'    datasets OK')" || IMPORT_FAIL=1

if [ "$MODE" = "generate" ] || [ "$MODE" = "all" ]; then
    python3 -c "import vllm; print(f'    vllm {vllm.__version__} OK')" || IMPORT_FAIL=1
fi
if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    python3 -c "import unsloth; print('    unsloth OK')" || IMPORT_FAIL=1
    python3 -c "import trl; print(f'    trl {trl.__version__} OK')" || IMPORT_FAIL=1
    python3 -c "import peft; print(f'    peft {peft.__version__} OK')" || IMPORT_FAIL=1
    python3 -c "import bitsandbytes; print('    bitsandbytes OK')" || IMPORT_FAIL=1
fi

if [ "$IMPORT_FAIL" -eq 1 ]; then
    echo ""
    echo "  [!] SOME IMPORTS FAILED — check above. Continuing anyway..."
else
    echo "  All imports verified ✓"
fi

# ── Step 4: Verify GPU ───────────────────────────────────────
echo ""
echo "[4/7] GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>/dev/null || true

# ── Step 5: Run generation (if requested) ───────────────────
if [ "$MODE" = "generate" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[5/7] Running Q&A generation (Qwen2.5-72B-Instruct-AWQ)..."
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

# ── Step 6: Run fine-tuning (if requested) ──────────────────
if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[6/7] Running QLoRA fine-tuning (Phi-3-Mini)..."
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

# ── Step 7: Eval (if fine-tuned) ──────────────────────────
if [ "$MODE" = "finetune" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[7/8] Running evaluation (base vs fine-tuned)..."
    python3 scripts/vast_eval.py --adapter models/phi3-mini-hr-finetuned || true
    echo "  Eval done"
fi

# ── Step 8: Package everything into one zip ───────────────
echo ""
echo "[8/8] Packaging outputs into single zip..."
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
