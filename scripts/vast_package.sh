#!/bin/bash
# ============================================================
# Package all Vast.ai outputs into ONE zip for fast download.
#
# Run this after vast_setup.sh completes:
#   bash scripts/vast_package.sh
#
# Downloads: scp or Vast.ai file browser → grab vast_results.zip
# ============================================================

set -euo pipefail

OUTZIP="vast_results.zip"
MANIFEST="vast_manifest.txt"

echo "=========================================="
echo "  Packaging Vast.ai outputs..."
echo "=========================================="

# Build file list — only include files that actually exist
FILES_TO_ZIP=""

# --- Generated data ---
for f in \
    data/generated_qa_h200.jsonl \
    data/eval_report.txt \
    data/eval_scores.csv \
    data/eval_results.txt \
; do
    [ -f "$f" ] && FILES_TO_ZIP="$FILES_TO_ZIP $f"
done

# --- Fine-tuned model (GGUF — the main deliverable) ---
for f in models/phi3-mini-hr-q4.gguf; do
    [ -f "$f" ] && FILES_TO_ZIP="$FILES_TO_ZIP $f"
done

# --- LoRA adapter weights (smaller, for HF loading) ---
if [ -d "models/phi3-mini-hr-finetuned" ]; then
    for f in models/phi3-mini-hr-finetuned/adapter_config.json \
             models/phi3-mini-hr-finetuned/adapter_model.safetensors \
             models/phi3-mini-hr-finetuned/tokenizer_config.json \
             models/phi3-mini-hr-finetuned/tokenizer.json \
             models/phi3-mini-hr-finetuned/special_tokens_map.json; do
        [ -f "$f" ] && FILES_TO_ZIP="$FILES_TO_ZIP $f"
    done
fi

# --- Training logs ---
[ -f "finetune_run.log" ] && FILES_TO_ZIP="$FILES_TO_ZIP finetune_run.log"
[ -f "outputs/qlora/training_args.bin" ] && FILES_TO_ZIP="$FILES_TO_ZIP outputs/qlora/training_args.bin"

# --- Validation report ---
# (validate_qa.py prints to stdout, but check for redirected file)
for f in data/validation_report.txt; do
    [ -f "$f" ] && FILES_TO_ZIP="$FILES_TO_ZIP $f"
done

# --- Write manifest ---
echo "Files in this archive:" > "$MANIFEST"
echo "Generated: $(date)" >> "$MANIFEST"
echo "---" >> "$MANIFEST"
for f in $FILES_TO_ZIP; do
    SIZE=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "?")
    echo "  $f  ($SIZE bytes)" >> "$MANIFEST"
done
FILES_TO_ZIP="$FILES_TO_ZIP $MANIFEST"

# --- Zip ---
if [ -z "$FILES_TO_ZIP" ]; then
    echo "[ERROR] No output files found. Did vast_setup.sh run?"
    exit 1
fi

echo ""
echo "Zipping files..."
zip -9 "$OUTZIP" $FILES_TO_ZIP 2>&1

SIZE_MB=$(stat -c%s "$OUTZIP" 2>/dev/null || stat -f%z "$OUTZIP" 2>/dev/null || echo "?")
if [ "$SIZE_MB" != "?" ]; then
    SIZE_MB=$((SIZE_MB / 1024 / 1024))
fi

echo ""
echo "=========================================="
echo "  [DONE] $OUTZIP ($SIZE_MB MB)"
echo "=========================================="
echo ""
echo "Download this ONE file:"
echo "  $OUTZIP"
echo ""
echo "Contents:"
cat "$MANIFEST"
echo ""
rm -f "$MANIFEST"
