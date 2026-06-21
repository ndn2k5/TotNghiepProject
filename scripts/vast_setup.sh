#!/bin/bash
# ============================================================
# Vast.ai H200 Setup — Run this IMMEDIATELY after SSH into instance
# Budget: $1.50 (~20 min). Every second counts.
# ============================================================

set -e

echo "=========================================="
echo "  VAST.AI H200 — QA GENERATION SETUP"
echo "=========================================="
START=$(date +%s)

# Step 1: Install vLLM (pre-compiled wheels, fastest path)
echo "[1/4] Installing vLLM..."
pip install vllm --quiet 2>&1 | tail -1

# Step 2: Clone repo (or upload manually)
echo "[2/4] Cloning project repo..."
if [ ! -d "TotNghiepProject" ]; then
    git clone --depth 1 https://github.com/ndn2k5/TotNghiepProject.git
fi
cd TotNghiepProject

# Step 3: Pre-download model (vLLM will cache it)
echo "[3/4] Model will download on first run via vLLM..."
echo "       Using: Qwen/Qwen2.5-72B-Instruct-AWQ"

# Step 4: Run generation
echo "[4/4] Starting Q&A generation..."
python scripts/vast_generate_qa.py \
    --batches 5 \
    --per-batch 10 \
    --output data/generated_qa_h200.jsonl

END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "=========================================="
echo "  DONE in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo "  1. Download: data/generated_qa_h200.jsonl"
echo "  2. Run validation: python scripts/validate_qa.py"
echo "  3. TERMINATE THIS INSTANCE NOW to save money!"
echo ""
