#!/usr/bin/env python3
"""
QLoRA Fine-Tuning on Vast.ai H200 — no Jupyter needed.

Trains Phi-3-Mini (or configurable model) on Vietnamese HR Q&A data generated
by vast_generate_qa.py. Exports Q4_K_M GGUF for local deployment.

Usage (on Vast.ai after vast_setup.sh):
    # Fine-tune on generated data:
    python scripts/vast_finetune.py

    # Custom model and paths:
    python scripts/vast_finetune.py \\
        --model unsloth/phi-3-mini-4k-instruct-bnb-4bit \\
        --data data/generated_qa_h200.jsonl \\
        --epochs 3

    # Local dry-run (no GPU):
    python scripts/vast_finetune.py --dry-run

Output:
    models/phi3-mini-hr-finetuned/   — LoRA adapter
    models/phi3-mini-hr-q4.gguf     — quantised GGUF for llama.cpp
"""

import json
import time
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetune_run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

# ─────────────────────── default config ────────────────────────────────────
DEFAULT_MODEL = "unsloth/phi-3-mini-4k-instruct-bnb-4bit"
DEFAULT_DATA = str(REPO_ROOT / "data" / "generated_qa_h200.jsonl")
# Fallback to original QA data if generated file not found
FALLBACK_DATA = str(REPO_ROOT / "data" / "qa_training_data.csv")

DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8        # H200 has 141GB → can afford big batches
DEFAULT_GRAD_ACCUM = 2        # effective batch = 8 × 2 = 16
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_LORA_R = 16
DEFAULT_LR = 2e-4

ADAPTER_DIR = REPO_ROOT / "models" / "phi3-mini-hr-finetuned"
MERGED_DIR = REPO_ROOT / "models" / "phi3-mini-hr-merged"
GGUF_PATH = REPO_ROOT / "models" / "phi3-mini-hr-q4.gguf"
TRAIN_OUTPUT_DIR = REPO_ROOT / "outputs" / "qlora"


# ─────────────────────── data loading ──────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    """Load Q&A pairs from JSONL file."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Support both {question,answer} and {q,a} field names
                q = obj.get("question") or obj.get("q", "")
                a = obj.get("answer") or obj.get("a", "")
                if q and a:
                    pairs.append({"question": str(q).strip(), "answer": str(a).strip()})
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON on line {line_no}, skipping")
    logger.info(f"Loaded {len(pairs)} Q&A pairs from {path}")
    return pairs


def load_csv(path: Path) -> List[Dict]:
    """Load Q&A pairs from CSV (question,answer columns)."""
    import csv
    pairs = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
    logger.info(f"Loaded {len(pairs)} Q&A pairs from {path}")
    return pairs


def find_data(data_arg: str) -> List[Dict]:
    """Find and load training data. Tries data_arg first, then fallback."""
    path = Path(data_arg)
    if path.exists():
        if path.suffix == ".csv":
            return load_csv(path)
        return load_jsonl(path)

    logger.warning(f"Data file not found: {path}")
    fallback = Path(FALLBACK_DATA)
    if fallback.exists():
        logger.info(f"Using fallback data: {fallback}")
        if fallback.suffix == ".csv":
            return load_csv(fallback)
        return load_jsonl(fallback)

    raise FileNotFoundError(
        f"No training data found at {data_arg} or {FALLBACK_DATA}.\n"
        "Run vast_generate_qa.py first, or supply --data path."
    )


# ─────────────────────── prompt formatting ─────────────────────────────────

def format_phi3(question: str, answer: str) -> str:
    """Format as Phi-3 chat template."""
    return f"<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"


def build_dataset(pairs: List[Dict]):
    """Build a HuggingFace Dataset from Q&A pairs."""
    from datasets import Dataset
    texts = [format_phi3(p["question"], p["answer"]) for p in pairs]
    return Dataset.from_dict({"text": texts})


# ─────────────────────── model + trainer ───────────────────────────────────

def run_training(
    pairs: List[Dict],
    model_name: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    lora_r: int = DEFAULT_LORA_R,
    lr: float = DEFAULT_LR,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
):
    """Load model, apply LoRA, train, and save adapter."""
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments

    logger.info(f"Loading base model: {model_name}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,        # auto: bfloat16 on H200/A100, float16 otherwise
        load_in_4bit=True,
    )
    logger.info(f"Base model loaded in {time.time() - t0:.1f}s")

    logger.info(f"Applying LoRA (r={lora_r})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_r,          # alpha = r is a good default
        lora_dropout=0.0,           # 0 = Unsloth fast-path kernels
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        init_lora_weights=True,
    )

    dataset = build_dataset(pairs)
    logger.info(f"Dataset: {len(dataset)} examples")

    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        warmup_ratio=0.03,
        learning_rate=lr,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=str(TRAIN_OUTPUT_DIR),
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        dataset_num_proc=min(4, os.cpu_count() or 1),
        args=training_args,
    )

    logger.info("=" * 60)
    logger.info(f"Starting training: {epochs} epochs, batch={batch_size}×{grad_accum}")
    logger.info("=" * 60)
    t_train = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - t_train
    logger.info(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Final loss: {trainer_stats.training_loss:.4f}")

    # Save LoRA adapter
    logger.info(f"Saving adapter to {ADAPTER_DIR}")
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    logger.info("Adapter saved ✓")

    return model, tokenizer, trainer_stats


# ─────────────────────── GGUF export ───────────────────────────────────────

def export_gguf(model, tokenizer) -> Optional[Path]:
    """Merge adapter and export to Q4_K_M GGUF via Unsloth."""
    try:
        logger.info("Saving merged model (float16) for GGUF conversion...")
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method="merged_16bit")
        logger.info(f"Merged model saved to {MERGED_DIR} ✓")
    except Exception as exc:
        logger.error(f"Merge failed: {exc}")
        return None

    # Try Unsloth's built-in GGUF export
    try:
        logger.info("Exporting Q4_K_M GGUF via Unsloth...")
        GGUF_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(
            str(GGUF_PATH.parent / GGUF_PATH.stem),
            tokenizer,
            quantization_method="q4_k_m",
        )
        if GGUF_PATH.exists():
            size_mb = GGUF_PATH.stat().st_size / 1024 / 1024
            logger.info(f"GGUF exported: {GGUF_PATH} ({size_mb:.0f} MB) ✓")
            return GGUF_PATH
    except Exception as exc:
        logger.warning(f"Unsloth GGUF export failed ({exc}), trying llama.cpp fallback...")

    # Fallback: use llama.cpp convert script
    convert_script = REPO_ROOT / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        logger.warning("llama.cpp not found — GGUF skipped. Download manually.")
        return None

    cmd = [
        sys.executable, str(convert_script),
        str(MERGED_DIR),
        "--outfile", str(GGUF_PATH),
        "--outtype", "q4_k_m",
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        size_mb = GGUF_PATH.stat().st_size / 1024 / 1024
        logger.info(f"GGUF exported: {GGUF_PATH} ({size_mb:.0f} MB) ✓")
        return GGUF_PATH
    else:
        logger.error(f"llama.cpp conversion failed:\n{result.stderr}")
        return None


# ─────────────────────── dry run ───────────────────────────────────────────

def dry_run():
    """Simulate the training pipeline without loading any models."""
    logger.info("DRY-RUN mode — skipping model load")
    logger.info("Would train on: " + DEFAULT_DATA)
    logger.info("Would save adapter to: " + str(ADAPTER_DIR))
    logger.info("Would export GGUF to: " + str(GGUF_PATH))
    print("\n[OK] Dry-run complete -- no GPU required\n")


# ─────────────────────── eval ──────────────────────────────────────────────

def quick_eval(model, tokenizer, num_questions: int = 5):
    """Run a quick post-training inference check on sample HR questions."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    test_questions = [
        "Nhân viên được nghỉ phép bao nhiêu ngày trong năm?",
        "Quy trình xin nghỉ phép như thế nào?",
        "Lương thử việc là bao nhiêu phần trăm?",
        "Công ty có chính sách bảo hiểm y tế không?",
        "Khi nào nhân viên được tăng lương?",
    ][:num_questions]

    inputs_list = tokenizer(
        [f"<|user|>\n{q}<|end|>\n<|assistant|>\n" for q in test_questions],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs_list,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
        )

    print("\n" + "=" * 60)
    print("  POST-TRAINING EVAL SAMPLE")
    print("=" * 60)
    for i, (q, out_ids) in enumerate(zip(test_questions, outputs)):
        full = tokenizer.decode(out_ids, skip_special_tokens=True)
        # Extract answer portion
        answer = full.split("<|assistant|>")[-1].strip() if "<|assistant|>" in full else full
        print(f"\nQ{i+1}: {q}")
        print(f"A:  {answer[:300]}")
    print("=" * 60 + "\n")


# ─────────────────────── main ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune on Vast.ai H200 (no Jupyter)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Unsloth model ID or HF model path")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="Training data: .jsonl or .csv with question/answer columns")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--skip-gguf", action="store_true",
                        help="Skip GGUF export (save time if not needed)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip post-training eval sample")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate run without loading any models")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    logger.info("=" * 60)
    logger.info("  VAST.AI H200 — QLoRA Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Model:      {args.model}")
    logger.info(f"Epochs:     {args.epochs}")
    logger.info(f"Batch size: {args.batch_size} × {args.grad_accum} (effective {args.batch_size * args.grad_accum})")
    logger.info(f"LoRA rank:  {args.lora_r}")
    logger.info(f"LR:         {args.lr}")

    # Load data
    pairs = find_data(args.data)
    if not pairs:
        logger.error("No training pairs loaded — aborting")
        return
    logger.info(f"Training examples: {len(pairs)}")

    # Train
    t_total = time.time()
    model, tokenizer, stats = run_training(
        pairs,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_r=args.lora_r,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
    )

    # Eval
    if not args.skip_eval:
        try:
            quick_eval(model, tokenizer)
        except Exception as exc:
            logger.warning(f"Eval failed: {exc}")

    # GGUF export
    if not args.skip_gguf:
        gguf_path = export_gguf(model, tokenizer)
        if gguf_path:
            logger.info(f"GGUF ready: {gguf_path}")
        else:
            logger.warning("GGUF export skipped or failed")

    total_elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info(f"  COMPLETE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"  Adapter:  {ADAPTER_DIR}")
    if not args.skip_gguf:
        logger.info(f"  GGUF:     {GGUF_PATH}")
    logger.info("=" * 60)
    logger.info("TERMINATE THE INSTANCE NOW to avoid extra charges!")


if __name__ == "__main__":
    main()
