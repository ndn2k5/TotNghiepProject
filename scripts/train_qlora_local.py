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
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # ── Check CUDA ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU — training on CPU is extremely slow.")
        print("Press Ctrl+C to cancel.")
    else:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} Q&A pairs from {csv_path}")

    def format_prompt(row):
        return (f"<|user|>\n{row['question']}<|end|>\n"
                f"<|assistant|>\n{row['answer']}<|end|>")

    texts = [format_prompt(r) for _, r in df.iterrows()]

    # ── 4-bit quantization ────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading {model_name} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Tokenize ──────────────────────────────────────────────────────────────
    print("Tokenizing dataset...")
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_seq_len,
        padding="max_length",
        return_tensors="pt",
    )

    class TextDataset(torch.utils.data.Dataset):
        def __getitem__(self, i):
            ids = encodings["input_ids"][i]
            return {"input_ids": ids, "attention_mask": encodings["attention_mask"][i],
                    "labels": ids.clone()}
        def __len__(self): return len(encodings["input_ids"])

    loader = DataLoader(TextDataset(), batch_size=1, shuffle=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    from bitsandbytes.optim import PagedAdamW8bit
    optimizer = PagedAdamW8bit(model.parameters(), lr=2e-4, weight_decay=0.01)
    total_steps = len(loader) * args.epochs // 8  # gradient_accumulation=8
    scheduler = get_linear_schedule_with_warmup(optimizer, 5, total_steps)

    # ── Train loop ────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training: {len(texts)} examples x {args.epochs} epochs")
    print(f"Adapter will be saved to: {output_dir}\n")

    model.train()
    ACCUM = 8
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / ACCUM
            loss.backward()
            total_loss += out.loss.item()

            if (step + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % (ACCUM * 10) == 0:
                avg = total_loss / (step + 1)
                pct = (step + 1) / len(loader) * 100
                print(f"  Epoch {epoch+1}/{args.epochs} | {pct:.0f}% | loss {avg:.4f}")

        print(f"Epoch {epoch+1} done — avg loss {total_loss/len(loader):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nDone! Adapter saved to {output_dir}")
    print("Next: run scripts/export_gguf.py to convert to GGUF for the chatbot")


if __name__ == "__main__":
    main()
