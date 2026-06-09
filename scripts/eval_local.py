"""
Evaluate fine-tuned Phi-3-Mini on Vietnamese HR questions.
Loads merged model with 4-bit quantization (no GGUF needed).
Saves results to data/eval_results.txt — send that file back.

Usage:
    python scripts/eval_local.py
    python scripts/eval_local.py --model models/phi3-mini-viet-merged
    python scripts/eval_local.py --model models/phi3-mini-viet-merged --questions my_questions.txt
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_QUESTIONS = [
    "Chính sách nghỉ phép năm của công ty là gì?",
    "Nhân viên mới có thời gian thử việc bao lâu?",
    "Quy trình xin nghỉ phép được thực hiện như thế nào?",
    "Mức lương tối thiểu áp dụng trong công ty là bao nhiêu?",
    "Công ty có những chính sách phúc lợi gì cho nhân viên?",
    "Quy định về trang phục làm việc của công ty như thế nào?",
    "Nhân viên có được phép làm thêm giờ không? Chính sách ra sao?",
    "Quy trình nghỉ ốm và nghỉ bệnh của công ty là gì?",
    "Chính sách khen thưởng và tăng lương của công ty như thế nào?",
    "Quy định về bảo mật thông tin trong công ty là gì?",
]


def generate(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    import torch
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/phi3-mini-viet-merged",
                        help="Path to merged model directory")
    parser.add_argument("--output", default="data/eval_results.txt")
    parser.add_argument("--questions", default=None,
                        help="Optional text file with one question per line")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run scripts/export_gguf.py first (Step 1 creates the merged model).")
        sys.exit(1)

    print("Loading libraries...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
    else:
        print("No GPU — running on CPU (slow but works)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded.\n")

    questions = TEST_QUESTIONS
    if args.questions:
        qfile = Path(args.questions)
        if qfile.exists():
            questions = [l.strip() for l in qfile.read_text(encoding="utf-8").splitlines()
                         if l.strip() and not l.startswith("#")]
            print(f"Loaded {len(questions)} questions from {qfile}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(f"Eval run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    results.append(f"Model: {model_path}")
    results.append("=" * 60)

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}")
        answer = generate(model, tokenizer, q)
        print(f"  → {answer[:120]}{'...' if len(answer) > 120 else ''}\n")
        results.append(f"\nQ{i}: {q}")
        results.append(f"A{i}: {answer}")
        results.append("-" * 40)

    out_path.write_text("\n".join(results), encoding="utf-8")
    print(f"\nDone! Results saved to {out_path}")
    print("Send that file back to review answer quality.")


if __name__ == "__main__":
    main()
