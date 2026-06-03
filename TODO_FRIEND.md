# TODO — Vietnamese QA Generation (Run Overnight)

**Goal:** Generate ~1500 Vietnamese HR Q&A pairs from chunked PDF data using Groq AI (free).  
These pairs will be used to fine-tune Phi-3-Mini in the next step.

**Time needed:** ~2–4 hours unattended (run before sleeping)  
**Cost:** Free (Groq free tier)

---

## Step 0 — Get the code

```bash
git clone https://github.com/ndn2k5/TotNghiepProject
cd TotNghiepProject
```

Or if already cloned:

```bash
git pull
```

---

## Step 1 — Install Python dependencies

Need Python 3.10+ installed. Then:

```bash
pip install openai httpx duckduckgo-search ddgs beautifulsoup4 pymupdf
```

---

## Step 2 — Get a FREE Groq API key (5 minutes)

1. Go to **https://console.groq.com**
2. Sign up with Google or email (free)
3. Click **"API Keys"** in the left sidebar
4. Click **"Create API Key"**
5. Copy the key — looks like `gsk_xxxxxxxxxxxxxxxxxxxx`
6. Save it somewhere safe

> Groq free tier: 14,400 requests/day, no credit card needed.

---

## Step 3 — Download Vietnamese HR PDFs

Run the crawler to download real Vietnamese company handbooks:

```bash
python scripts/crawl_hr_pdfs.py --limit 10
```

This searches for `so-tay-nhan-vien`, `noi-quy-cong-ty` etc. and downloads to `data/raw/pdf/`.

**Check what was downloaded:**

```bash
ls data/raw/pdf/
```

You should see files like:
- `SO-TAY-NHAN-VIEN-THAI-SAN.pdf`
- `HACOM_So-tay-Nhan-vien-2025-2.pdf`
- etc.

If the folder looks empty or has unrelated PDFs (Medicare, road rules, etc.) — delete those and re-run:

```bash
# Windows
del data\raw\pdf\*.pdf
python scripts/crawl_hr_pdfs.py --limit 10
```

---

## Step 4 — Extract and chunk the PDFs

```bash
python scripts/ingest_pdf_handbooks.py
```

Output: `data/raw_chunks_viet.jsonl`

Expected output:
```
Found X PDFs in data\raw\pdf
  Processing: SO-TAY-NHAN-VIEN-THAI-SAN.pdf
    -> 350 chunks
  ...
Total: ~1500-3000 chunks -> data/raw_chunks_viet.jsonl
```

> If a PDF shows `0 chunks` — it is a scanned image PDF with no text layer. Delete it and find another.

---

## Step 5 — Run QA generation overnight

Replace `gsk_YOUR_KEY_HERE` with your actual Groq key from Step 2.

**Windows (cmd):**
```cmd
python scripts/generate_qa.py ^
  --vllm-url https://api.groq.com/openai ^
  --api-key gsk_YOUR_KEY_HERE ^
  --model llama-3.3-70b-versatile ^
  --input data/raw_chunks_viet.jsonl
```

**Linux/Mac:**
```bash
python scripts/generate_qa.py \
  --vllm-url https://api.groq.com/openai \
  --api-key gsk_YOUR_KEY_HERE \
  --model llama-3.3-70b-versatile \
  --input data/raw_chunks_viet.jsonl
```

**Let it run overnight.** Output goes to `data/qa_pairs_viet.jsonl`.

Progress looks like:
```
[chunk 10/1500 | new 10] QA: 25 | ETA: 45.2min
[chunk 20/1500 | new 20] QA: 51 | ETA: 40.1min
...
Done. QA pairs: 3200 | Errors: 2 | Time: 87.3min
```

> Safe to Ctrl+C and resume later — script has auto-checkpoint. Re-run same command to continue.

---

## Step 6 — Verify the output

```bash
python -c "
pairs = open('data/qa_pairs_viet.jsonl', encoding='utf-8').readlines()
print('Total QA pairs:', len(pairs))
import json
sample = json.loads(pairs[0])
print('Sample Q:', sample['question'])
print('Sample A:', sample['answer'][:100])
"
```

Expected: **1000–3000 pairs**. If less than 500, re-run Step 3 with more PDFs.

---

## Step 7 — Convert to CSV for training

```bash
python -c "
import json, csv
pairs = [json.loads(l) for l in open('data/qa_pairs_viet.jsonl', encoding='utf-8')]
with open('data/qa_training_data_viet.csv','w',encoding='utf-8',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['question','answer'])
    w.writeheader()
    w.writerows({'question':p['question'],'answer':p['answer']} for p in pairs)
print(len(pairs), 'pairs written to data/qa_training_data_viet.csv')
"
```

---

## Step 8 — Commit and push

```bash
git add data/raw_chunks_viet.jsonl data/qa_pairs_viet.jsonl data/qa_training_data_viet.csv
git commit -m "Add Vietnamese HR QA dataset from PDF handbooks"
git push
```

---

## Step 9 — Fine-tune (next session, needs GPU)

After push, open **Google Colab** (free T4 GPU):

1. Open `notebooks/FINETUNE QLORA.ipynb`
2. In cell `103b763b`, change CSV path:
   ```python
   csv_path = DATA_DIR / "qa_training_data_viet.csv"  # changed from qa_training_data.csv
   ```
3. Run all cells
4. Download the GGUF file when prompted
5. Replace `models/phi-3-mini.gguf` with the new file

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install openai httpx ddgs pymupdf beautifulsoup4` |
| `0 PDFs found` by crawler | Check internet; try `--dry-run` first to see URLs |
| PDF gives `0 chunks` | Scanned PDF — delete it, find text-based one |
| Groq `rate limit` error | Script auto-retries; or wait 1 min and re-run |
| Script crashes mid-run | Re-run same command — checkpoint auto-resumes |
| Less than 500 QA pairs | Download more PDFs (Step 3 with `--limit 20`) |

---

**Contact:** Push your results and message when done. Next step is Colab training.
