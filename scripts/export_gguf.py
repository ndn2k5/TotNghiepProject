"""
Merge LoRA adapter into base model and export to GGUF.
Run AFTER train_qlora_local.py finishes.

Requirements:
    pip install transformers peft torch
    # llama.cpp must be cloned once (script does it automatically)

Usage:
    python scripts/export_gguf.py
    python scripts/export_gguf.py --adapter models/phi3-mini-viet-adapter --out models/phi3-mini-viet.gguf
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="models/phi3-mini-viet-adapter")
    parser.add_argument("--merged", default="models/phi3-mini-viet-merged")
    parser.add_argument("--out", default="models/phi-3-mini.gguf",
                        help="Output GGUF path — replaces existing chatbot model")
    parser.add_argument("--quant", default="q4_k_m")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    merged_path = Path(args.merged)
    out_path = Path(args.out)

    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Run train_qlora_local.py first.")
        sys.exit(1)

    # ── Step 1: Merge adapter into base model ─────────────────────────────────
    print("Step 1/3: Merging LoRA adapter into base model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"  Loading base: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    print("  Applying adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    print(f"  Saving merged model to {merged_path}...")
    model.save_pretrained(str(merged_path), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_path))
    print("  Merge done.")

    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 2: Clone llama.cpp if needed ────────────────────────────────────
    llamacpp_dir = Path("llama.cpp")
    if not llamacpp_dir.exists():
        print("\nStep 2/3: Cloning llama.cpp (one-time, ~100MB)...")
        subprocess.run(["git", "clone", "--depth=1",
                        "https://github.com/ggerganov/llama.cpp",
                        str(llamacpp_dir)], check=True)
    else:
        print("\nStep 2/3: llama.cpp already cloned.")

    # Install llama.cpp Python deps
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "sentencepiece", "numpy"], check=False)

    # ── Step 3: Convert to GGUF ───────────────────────────────────────────────
    print(f"\nStep 3/3: Converting to GGUF ({args.quant})...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    convert_script = llamacpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = llamacpp_dir / "convert-hf-to-gguf.py"

    result = subprocess.run([
        sys.executable, str(convert_script),
        str(merged_path),
        "--outfile", str(out_path.with_suffix("").with_name(out_path.stem + ".BF16.gguf")),
        "--outtype", "bf16",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Conversion error:", result.stderr[-500:])
        sys.exit(1)

    # Quantize BF16 → Q4_K_M using llama-quantize
    bf16_path = out_path.with_suffix("").with_name(out_path.stem + ".BF16.gguf")
    print(f"Quantizing {bf16_path} -> {out_path}...")

    # Try pre-built binary first, fall back to python quant
    quantize_bin = llamacpp_dir / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llamacpp_dir / "llama-quantize.exe"

    if quantize_bin.exists():
        subprocess.run([str(quantize_bin), str(bf16_path), str(out_path),
                        args.quant.upper()], check=True)
    else:
        print("  llama-quantize binary not found — keeping BF16 GGUF as output.")
        bf16_path.rename(out_path)

    if bf16_path.exists() and out_path.exists() and bf16_path != out_path:
        bf16_path.unlink()

    print(f"\nDone! GGUF saved to {out_path}")
    size_gb = out_path.stat().st_size / 1e9
    print(f"File size: {size_gb:.2f} GB")
    print(f"This file replaces models/phi-3-mini.gguf for the chatbot.")


if __name__ == "__main__":
    main()
