"""
QLoRA fine-tuning for Phi-3-Mini on Windows with 4GB VRAM.
Uses HuggingFace transformers + PEFT + TRL (no Unsloth needed).

Install once:
    pip install transformers peft trl bitsandbytes accelerate datasets torch

Usage:
    python scripts/train_qlora_local.py --csv data/qa_training_data_viet.csv
    python scripts/train_qlora_local.py --csv data/qa_training_data_viet.csv --epochs 3
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 globally — fixes TRL's deepseekv3.jinja read crash on Windows cp1252
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/qa_training_data_viet.csv",
                        help="Training CSV with question,answer columns")
    parser.add_argument("--output", default="models/phi3-mini-viet-adapter",
                        help="Where to save LoRA adapter weights")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Max token length — lower = less VRAM (512 safe for 4GB)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    print("Loading libraries...")
    import torch
    import pandas as pd
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    # ── Check CUDA ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU found — training on CPU will be extremely slow.")
        print("Press Ctrl+C to cancel, or wait to continue on CPU.")
    else:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} Q&A pairs from {csv_path}")

    def format_prompt(row):
        return (f"<|user|>\n{row['question']}<|end|>\n"
                f"<|assistant|>\n{row['answer']}<|end|>")

    dataset = Dataset.from_dict({"text": [format_prompt(r) for _, r in df.iterrows()]})

    # ── 4-bit quantization config ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load base model ───────────────────────────────────────────────────────
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading {model_name} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    # ── LoRA config — tuned for 4GB VRAM ─────────────────────────────────────
    lora_config = LoraConfig(
        r=8,                      # rank — lower = less VRAM
        lora_alpha=16,
        lora_dropout=0.0,         # 0 = fastest
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,       # 1 = safe for 4GB
        gradient_accumulation_steps=8,       # effective batch = 8
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        optim="paged_adamw_8bit",            # saves ~1GB vs standard adamw
        warmup_steps=5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        gradient_checkpointing=True,         # trades speed for ~1GB VRAM saving
        dataloader_pin_memory=False,
        report_to="none",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    print(f"\nStarting training: {len(dataset)} examples x {args.epochs} epochs")
    print(f"Adapter will be saved to: {output_dir}\n")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nDone! Adapter saved to {output_dir}")
    print("Next: run scripts/export_gguf.py to convert to GGUF for the chatbot")


if __name__ == "__main__":
    main()
