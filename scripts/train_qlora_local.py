"""
QLoRA fine-tuning for Phi-3-Mini on Windows with 4GB VRAM.
Uses HuggingFace transformers + PEFT (no Unsloth/TRL needed).

Install once (exact versions required — newer transformers breaks 4-bit loading):
    pip install "transformers==4.43.4" "accelerate==0.33.0" "peft==0.11.1" "trl==0.9.6" bitsandbytes torch pandas datasets

After installing, delete stale Phi-3 cache ONCE:
    rmdir /s /q %USERPROFILE%\\.cache\\huggingface\\modules\\transformers_modules\\microsoft

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

# Guard: transformers 4.44+ auto-injects device_map for quantized models, which breaks
# bitsandbytes 4-bit loading. Must stay on 4.43.x.
import importlib.metadata as _meta
_tv = _meta.version("transformers")
_major, _minor, *_ = [int(x) for x in _tv.split(".")[:3]]
if (_major, _minor) >= (4, 44):
    print(f"ERROR: transformers {_tv} detected — versions 4.44+ break 4-bit loading on Windows.")
    print("Fix: pip install \"transformers==4.43.4\" \"accelerate==0.28.0\"")
    print("Then: rmdir /s /q %USERPROFILE%\\.cache\\huggingface\\modules\\transformers_modules\\microsoft")
    sys.exit(1)
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
        AutoConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
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

    # Patch config to avoid KeyError: 'type' or ValueError: 'default' in RoPE scaling
    conf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(conf, "rope_scaling") and conf.rope_scaling is not None:
        # Handle 'default' type which causes ValueError in Phi-3 remote code
        if conf.rope_scaling.get("type") == "default" or conf.rope_scaling.get("rope_type") == "default":
            conf.rope_scaling = None
        elif "type" not in conf.rope_scaling and "rope_type" in conf.rope_scaling:
            conf.rope_scaling["type"] = conf.rope_scaling["rope_type"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=conf,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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
        gradient_checkpointing=False,        # handled by prepare_model_for_kbit_training above
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
