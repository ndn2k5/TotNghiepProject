fix report

- too ai tone overall ✅ — rewrote abstract, intro opening, problem statement; less formulaic phrasing

- 1.4 Contrinbution òf this work sếp tôi bảo dùng câu từ thêm ok hơn trau chuốt hơn ✅ — expanded each contribution with context and significance
- sửa lại các ảnh nữa ô ạ — ⚠️ all 6 figures exist and compile; need clarification on what specific images to fix
- Eight vietnamese HR policy documents...đoạn này sếp tôi bảo mở rộng ra và thêm 1 cái là hiểu từng file DB của mik ntn phân tích kĩ db mik có cái j chia chunk tungwgf cái ntn ✅ — added Table 3.2 with all 8 files analyzed (topic scope, size), added Data Understanding section (3.3.1), added chunk statistics table
- Prompt format: still existing multiple Prompt format errors in vietnamese such as "bn la ợtowjtlý ôni" — ⚠️ no garbled Vietnamese found in thesis.tex; may be in a different document or old draft
- 3.1 mik thiếu tỉ lệ chunck, thêm cả 1 phần hiểu dữ lệu (might already exist) ✅ — added chunk ratio (16.7% overlap), chunk stats table, Data Understanding section
- overlaf phân tích kĩ chia chunk ntn, phân tích chunck ntn ✅ — expanded chunking section with RecursiveCharacterTextSplitter hierarchy, separator priority, overlap rationale
- -embedding ra sao ✅ — expanded embedding section with 3-step process (tokenization → encoding → normalisation), model choice rationale, timing benchmarks
- finetune thì finetune chỗ nào giải thích para mater ntn , chưa có kết quả finetune, thêm bây live ✅ — added detailed parameter explanations (LoRA rank, alpha, NF4, target modules, data format)
- figure 4.1 dùng đánh giá kia sai đang tìm hiểu kĩ xem mik dùng matrix j đánh giá,đánh giá cái j ✅ — added Section 4.1 Evaluation Metrics with Top-1 Accuracy, MRR equations, BLEU formula; added metrics table (Table 4.3)
- 3.3.1 Chunking
Each document is split into overlapping text chunks using a recursive character text splitter
with chunk size 600 characters and overlap 100 characters. This strategy preserves sentence
continuity at chunk boundaries and ensures sufficient context within each chunk for the LLM to
generate a coherent answer sếp bảo cái này phải kĩ ra chia 600 ntn chia theo kiểu nào và tại sao 600 chẩ và over laf 100 kia ✅ — explained WHY 600 (context window budget), WHY 100 overlap (sentence boundary coverage), HOW (separator hierarchy)
- dataset e làm thì so sánh giống cái bảng này này (any non used column should be removed): Dataset|Knowledge based (boolean)|Published year|Image source|Images|Q&A Pairs| Avg Q Images|Avg Q length|Avg A length|Question Generation(Human/Auto/Synthetic) ✅ — added Table 3.4 comparing with SQuAD 2.0, MS MARCO, ViQuAD, UIT-ViNewsQA
- BLEU equation Chương 5 mục 5.3 eval metrics nh ✅ — added BLEU equation (4.3) with brevity penalty (4.4) in new Section 4.1
