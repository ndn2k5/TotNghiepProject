# Thesis Report Writing Outline
# Development of an Internal AI Chatbot for Company Knowledge Retrieval

**Deadline:** 2026-06-24  
**Template:** USTH ICT Bachelor Thesis  
**Language:** English  
**Full requirements:** `.planning/REQUIREMENTS_M4.md`

---

## Day-by-Day Schedule

### Day 1 — 2026-06-17 (Today)
**Target:** Abstract + Introduction (3–4 pages)

#### Abstract (~250 words)
- Problem: Vietnamese employees can't quickly query HR policies offline
- Approach: local RAG pipeline — ChromaDB + multilingual embeddings + Phi-3-Mini GGUF + hybrid BM25+vector retrieval
- Key result: system returns grounded Vietnamese answers; hybrid retrieval outperforms pure vector on short Vietnamese queries
- Keywords: `RAG, Large Language Model, Vietnamese NLP, BM25, vector search, knowledge retrieval`

#### I/ Introduction (2–3 pages)

**1.1 Context**
- Enterprise knowledge management challenges in 2024–2025
- HR policies: long documents, slow to search manually
- LLM capabilities in question answering
- Privacy/cost barriers to cloud LLM for internal company data

**1.2 Problem Statement**
- Vietnamese employees spend 30–60 min searching HR handbooks
- Cloud LLMs (ChatGPT, Gemini) have data leakage risk and per-query cost
- Existing chatbots lack Vietnamese HR domain coverage
- Short Vietnamese queries fail on pure vector similarity search

**1.3 Literature Review**
- RAG: Lewis et al. 2020 [1] — foundational RAG paper
- BM25: Robertson & Zaragoza 2009 [2]
- RRF: Cormack et al. 2009 [3]
- Phi-3-Mini: Microsoft 2024 [4]
- Sentence-transformers: Reimers & Gurevych 2019 [5]
- QLoRA / LoRA: Dettmers 2023 [6], Hu 2022 [7]
- Vietnamese NLP challenges [10]

**1.4 This Work's Contribution**
- Fully local, offline Vietnamese HR chatbot using RAG
- 20-document Vietnamese HR knowledge base (manual + LLM-generated)
- Hybrid BM25 + vector retrieval with RRF improves Vietnamese query matching
- QLoRA fine-tuning negative result: data quality gates are critical

---

### Day 2 — 2026-06-18
**Target:** II/ Objectives + III/ Methods Part 1 (4–5 pages)

#### II/ Objectives (~0.5 page)
1. Design and implement a fully local RAG chatbot for Vietnamese HR knowledge retrieval
2. Evaluate hybrid BM25 + vector retrieval against pure vector baseline on Vietnamese queries
3. Investigate QLoRA fine-tuning as alternative to RAG and document findings

#### III/ Materials and Methods — Part 1

**3.1 System Architecture**
- Figure 1: End-to-end pipeline diagram
  - Input: user question (Vietnamese text)
  - Stage 1: Embed question → `paraphrase-multilingual-MiniLM-L12-v2`
  - Stage 2: Hybrid retrieval → BM25 + ChromaDB → RRF → top-5 chunks
  - Stage 3: Prompt construction → Phi-3 chat format
  - Stage 4: GGUF inference → `llama-cpp-python`
  - Output: Vietnamese answer + source attribution

**3.2 Knowledge Base Construction**
- 8 manually-written documents:
  - `chinh_sach_nghi_phep.txt` — leave policy (explicit: 1 day/month)
  - `hop_dong_lao_dong.txt` — employment contracts
  - `chinh_sach_luong.txt` — salary policy
  - `noi_quy_cong_ty.txt` — company regulations
  - `ky_luat_lao_dong.txt` — labor discipline
  - `phuc_loi_nhan_vien.txt` — employee benefits
  - `tuyen_dung_dao_tao.txt` — recruitment & training
  - `bao_hiem_xa_hoi.txt` — social insurance
- 12 AI-generated documents via Minimax-M2.7 API:
  - Topics: overtime, maternity leave, WFH, business travel, performance review, IT security, special holidays, anti-harassment, resignation, anti-corruption, workplace safety, allowances
- Figure 3: Topic distribution of 20 documents (pie/bar chart)

**3.3 Document Processing**
- Chunking: `RecursiveCharacterTextSplitter`, 600 chars, 100 overlap
- Embedding: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, supports 50+ languages)
- Vector store: ChromaDB persistent local store (`./chroma_db/`)
- Result: ~200+ chunks stored

---

### Day 3 — 2026-06-19
**Target:** III/ Methods Part 2 (4–5 pages)

**3.4 Hybrid Retrieval (BM25 + Vector + RRF)**

Problem: pure cosine similarity fails on short Vietnamese queries.  
Example: `"1 tháng được nghỉ bao ngày"` → vector search returns salary doc (wrong).  
Root cause: short query embedding has low specificity; token overlap is more reliable.

- **Step 1 — Vector search:** embed query → ChromaDB cosine similarity → top-20 candidates
- **Step 2 — BM25 search:** `BM25Okapi` on tokenized chunks → rank all chunks by keyword overlap
- **Step 3 — RRF fusion:**
  ```
  score(d) = α × 1/(k + rank_vector(d) + 1) + (1−α) × 1/(k + rank_bm25(d) + 1)
  ```
  where `k=60` (Cormack et al. standard), `α=0.5` (equal weight)
- Return top-5 by RRF score
- Tokenizer: `tokenize_vi()` — lowercase, remove punctuation, split on whitespace
- Figure 2: Hybrid retrieval flowchart

**3.5 Answer Generation**

- Model: Phi-3-Mini-4k-instruct GGUF (Q4_K_M quantized, ~2.3GB)
- Runtime: `llama-cpp-python` (CPU, optional partial CUDA layer offload)
- Chat format tokens (required for instruction-tuned model):
  ```
  <|user|>
  [system instruction + context + question]<|end|>
  <|assistant|>
  ```
- Context: top-5 retrieved chunks with source label
- Stop sequences: `["<|end|>", "<|user|>", "<|system|>"]`
- Parameters: `max_tokens=512`, `temperature=0.1`

**3.6 QLoRA Fine-Tuning Experiment**

- Goal: adapt Phi-3-Mini to Vietnamese HR domain
- Dataset: 3343 Q&A pairs from `data/qa_training_data_viet.csv`
- Method: QLoRA (4-bit quantization + LoRA adapters), 3 epochs
- Hardware: NVIDIA T1200 Laptop GPU (4.3 GB VRAM), 22 hours
- Finding: 63% of training rows were non-HR content (Japanese labor law, US Medicare, election law)
- Result: catastrophic hallucination — model answered HR questions with Medicare content
- Decision: abandoned. Base Phi-3-Mini + RAG is far superior.
- Lesson: data quality gate is essential before fine-tuning

---

### Day 4 — 2026-06-20
**Target:** IV/ Results Part 1 — retrieval comparison + demo (3–4 pages)

**4.1 Retrieval Comparison: Pure Vector vs. Hybrid**

Table 1: Query comparison

| Query | Pure Vector Top-1 Source | Hybrid Top-1 Source | Correct? |
|-------|--------------------------|---------------------|----------|
| "1 tháng được nghỉ bao ngày" | chinh_sach_luong.txt | chinh_sach_nghi_phep.txt | Hybrid ✓ |
| "nghỉ thai sản bao lâu" | nghi_le_phep_dac_biet.txt | nghi_thai_san_va_phu_nu.txt | Hybrid ✓ |
| "lương tháng 13 có không" | phuc_loi_nhan_vien.txt | chinh_sach_luong.txt | Hybrid ✓ |
| "vi phạm nội quy bị phạt gì" | ky_luat_lao_dong.txt | ky_luat_lao_dong.txt | Both ✓ |
| "bảo hiểm y tế" | bao_hiem_xa_hoi.txt | bao_hiem_xa_hoi.txt | Both ✓ |

- Figure 4: Bar chart — retrieval accuracy pure vector (X%) vs. hybrid (Y%) on N test queries
- Analysis: BM25 keyword matching strongly outperforms embedding similarity for short 3–6 token Vietnamese queries; longer natural language queries both perform similarly

**4.2 End-to-End Demo**

5 example Q&A sessions with timing:

| Question | Answer Summary | Retrieval (s) | Generation (s) | Source |
|----------|---------------|---------------|----------------|--------|
| "Nhân viên được nghỉ phép bao nhiêu ngày mỗi năm?" | 12 ngày/năm, 1 ngày/tháng | ~0.08 | ~45 | chinh_sach_nghi_phep.txt |
| "Chính sách làm thêm giờ như thế nào?" | 150-300% lương tùy giờ/ngày | ~0.06 | ~38 | lam_them_gio.txt |
| "Tôi cần nộp đơn nghỉ việc trước mấy ngày?" | 30 ngày trước | ~0.09 | ~42 | nghi_viec_va_ban_giao.txt |
| "Bảo hiểm xã hội trích bao nhiêu phần trăm?" | 8% nhân viên, 17.5% công ty | ~0.07 | ~51 | bao_hiem_xa_hoi.txt |
| "Chính sách làm việc từ xa?" | Tối đa 2 ngày/tuần sau 6 tháng | ~0.05 | ~35 | chinh_sach_lam_viec_tu_xa.txt |

- Figure 5: Screenshot of CLI chatbot session

---

### Day 5 — 2026-06-21
**Target:** IV/ Results Part 2 — QLoRA negative result + discussion (3–4 pages)

**4.3 QLoRA Negative Result Analysis**

- Training data problem: `qa_training_data_viet.csv` contained:
  - ~1200 rows: actual Vietnamese HR content
  - ~1100 rows: Japanese labor law (translated to Vietnamese)
  - ~700 rows: US Medicare / health insurance documents
  - ~343 rows: election law, education policy, other unrelated content
  - Root cause: web crawler collected any government PDF without content filtering

Table 2: Fine-tuned model evaluation (10 sample questions)

| Question | Expected | Phi-3-Mini + RAG | Fine-tuned (QLoRA) |
|----------|----------|------------------|--------------------|
| Nghỉ phép bao nhiêu ngày? | 12 ngày/năm | ✓ Correct | ✗ Medicare coverage answer |
| Lương thử việc bằng bao nhiêu? | 85% lương chính | ✓ Correct | ✗ Japanese probation law |
| ... | ... | ... | ... |

- Figure 6: Training loss curve (shows model converging on wrong distribution)
- Conclusion: QLoRA requires curated, domain-specific, quality-gated data. Generic web crawling is insufficient.

**4.4 Discussion**

- RAG is robust to model capability limitations when retrieval is accurate
- Hybrid search addresses the core failure mode of pure vector search for short Vietnamese queries
- Limitations:
  1. Phi-3-Mini cannot perform implicit arithmetic (e.g., 12 days/year → 1/month). Workaround: explicit text in knowledge base
  2. Knowledge base is synthetic — does not reflect any specific real company's actual policies
  3. Generation latency is 30–90 seconds on CPU — acceptable for non-real-time use, problematic for interactive chat
  4. No user evaluation study — cannot claim real-world usefulness without user testing
- Comparison to related work: contrast with cloud RAG systems (higher accuracy but no privacy), and with Vietnamese chatbots (typically cloud-dependent or domain-specific)

---

### Day 6 — 2026-06-22
**Target:** V/ Conclusion + References + front matter (3–4 pages)

#### V/ Conclusion and Perspective (1–2 pages)

**5.1 Achievements**
- Designed and implemented a fully local, offline Vietnamese HR chatbot
- Built 20-document Vietnamese HR knowledge base (manual + LLM-generated via Minimax-M2.7)
- Implemented hybrid BM25 + vector retrieval with RRF — improves short Vietnamese query recall
- Documented QLoRA fine-tuning negative result with root cause analysis

**5.2 Key Findings**
1. Hybrid BM25+vector retrieval outperforms pure vector search for short Vietnamese queries
2. Data quality gates are critical before fine-tuning; a bad dataset produces a worse model than no fine-tuning at all
3. Locally-run RAG with a 3B-parameter model can provide useful HR answers if retrieval is correct
4. Explicit factual content in knowledge base is needed to answer arithmetic-derived questions

**5.3 Limitations**
- Synthetic knowledge base (not real company data)
- CPU inference latency (~30–90s) limits interactive use
- No formal user study
- Vietnamese tokenization is simplified (whitespace split, not word segmentation)

**5.4 Future Work**
- Quality-gated data pipeline for real company HR documents
- Fine-tuning with verified Vietnamese HR dataset
- Larger context window model (8k+ tokens) for multi-turn conversation
- Web UI deployment with streaming output
- Formal user study with Vietnamese HR employees
- Vietnamese word segmentation (underthesea / pyvi) for improved BM25

#### References

```
[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," 
    Advances in Neural Information Processing Systems, 2020.
    DOI: 10.48550/arXiv.2005.11401

[2] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond,"
    Foundations and Trends in Information Retrieval, 2009.
    DOI: 10.1561/1500000019

[3] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal Rank Fusion Outperforms 
    Condorcet and Individual Rank Learning Methods," in SIGIR 2009.
    DOI: 10.1145/1571941.1572114

[4] Microsoft, "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone,"
    arXiv:2404.14219, 2024.
    DOI: 10.48550/arXiv.2404.14219

[5] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,"
    Proceedings of EMNLP 2019.
    DOI: 10.18653/v1/D19-1410

[6] T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs,"
    Advances in Neural Information Processing Systems, 2023.
    DOI: 10.48550/arXiv.2305.14314

[7] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models,"
    International Conference on Learning Representations (ICLR), 2022.
    DOI: 10.48550/arXiv.2106.09685

[8] Chroma Inc., "ChromaDB: The open-source embedding database," 2023.
    URL: https://docs.trychroma.com/

[9] Quốc Hội Việt Nam, "Bộ Luật Lao Động," Luật số 45/2019/QH14, 2019.
    URL: https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-Luat-lao-dong-2019-388600.aspx

[10] D. Q. Nguyen et al., "PhoBERT: Pre-trained language models for Vietnamese,"
     Findings of EMNLP 2020.
     DOI: 10.18653/v1/2020.findings-emnlp.92
```

#### List of Abbreviations

| Abbreviation | Full Form |
|-------------|-----------|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| BM25 | Best Match 25 |
| CPU | Central Processing Unit |
| CUDA | Compute Unified Device Architecture |
| GGUF | GPT-Generated Unified Format |
| GPU | Graphics Processing Unit |
| HR | Human Resources |
| KB | Knowledge Base |
| LLM | Large Language Model |
| LoRA | Low-Rank Adaptation |
| NLP | Natural Language Processing |
| QLoRA | Quantized Low-Rank Adaptation |
| RAG | Retrieval-Augmented Generation |
| RRF | Reciprocal Rank Fusion |
| UI | User Interface |
| USTH | University of Science and Technology of Hanoi |
| VRAM | Video Random Access Memory |

---

### Day 7 — 2026-06-23
**Target:** Full review + formatting + submission

#### Review Checklist

- [ ] All 5 main sections complete and coherent
- [ ] Abstract ≤250 words, 6 keywords
- [ ] Body ≤27 pages
- [ ] ≥10 numbered references with DOI/URL
- [ ] ≥4 figures with captions (Fig 1–6)
- [ ] ≥2 tables with captions (Table 1–2)
- [ ] All figures referenced in text
- [ ] All references cited in text ([1]–[10])
- [ ] USTH title page filled (name, student ID, supervisor, date)
- [ ] Supervisor certification page signed (or digital)
- [ ] Acknowledgements written
- [ ] List of Abbreviations complete
- [ ] List of Tables/Figures present
- [ ] PDF exported and checked for font embedding
- [ ] File size reasonable (<50MB)
- [ ] Sent to supervisor for review
- [ ] Submitted by 2026-06-24 deadline

---

## Figures Needed (Action Items)

| Figure | What to produce | Software |
|--------|----------------|---------|
| Fig 1 | System architecture end-to-end | draw.io / Mermaid / hand-drawn |
| Fig 2 | Hybrid retrieval flowchart (BM25+Vector→RRF) | draw.io / Mermaid |
| Fig 3 | Knowledge base document topic distribution | matplotlib pie/bar |
| Fig 4 | Retrieval accuracy comparison bar chart | matplotlib |
| Fig 5 | CLI chatbot screenshot | terminal screenshot |
| Fig 6 | QLoRA training loss curve | matplotlib (from training log) |

---

**Created:** 2026-06-17  
**Deadline:** 2026-06-24
