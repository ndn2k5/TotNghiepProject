# HƯỚNG DẪN — Tạo dữ liệu QA tiếng Việt (Chạy qua đêm)

**Mục tiêu:** Tạo ~1500 cặp hỏi-đáp nhân sự tiếng Việt từ dữ liệu PDF để huấn luyện mô hình AI.  
**Thời gian:** 2–4 tiếng chạy tự động (bật máy trước khi ngủ)  
**Chi phí:** Miễn phí (Groq free tier)  
**Hệ điều hành:** Windows

---

## Bước 0 — Tải code về máy

Mở **Command Prompt** (nhấn `Win + R`, gõ `cmd`, Enter):

```cmd
git clone https://github.com/ndn2k5/TotNghiepProject
cd TotNghiepProject
```

Nếu đã clone rồi thì chỉ cần:

```cmd
git pull
```

---

## Bước 1 — Cài thư viện Python

```cmd
pip install openai httpx ddgs beautifulsoup4 pymupdf
```

Chờ cài xong (1–2 phút).

---

## Bước 1b — Cài thư viện huấn luyện (phiên bản chính xác — QUAN TRỌNG)

> **Lưu ý:** transformers phiên bản 4.44 trở lên bị lỗi khi nạp mô hình 4-bit trên Windows. Phải dùng đúng phiên bản dưới đây.

```cmd
pip install "transformers==4.43.4" "accelerate==0.33.0" "peft==0.11.1" "trl==0.9.6" bitsandbytes torch pandas datasets
```

Sau khi cài xong, xóa cache cũ của Phi-3 (chạy **1 lần duy nhất**):

```cmd
rmdir /s /q %USERPROFILE%\.cache\huggingface\modules\transformers_modules\microsoft
```

Nếu thấy "The system cannot find the path specified" thì bỏ qua — không sao.

---

---

## Bước 2 — Lấy API key Groq miễn phí (5 phút)

1. Vào **[https://console.groq.com](https://console.groq.com)**
2. Đăng ký bằng Gmail hoặc email (miễn phí, không cần thẻ)
3. Sau khi đăng nhập, nhấn **"API Keys"** ở thanh bên trái
4. Nhấn **"Create API Key"**
5. Copy key — trông giống như: `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx`
6. Dán vào Notepad để dùng ở Bước 5

> Groq miễn phí: 14.400 request/ngày, đủ để chạy qua đêm.

---

## Bước 3 — Tải PDF sổ tay nhân viên

```cmd
python scripts/crawl_hr_pdfs.py --limit 10
```

Script tự tìm và tải PDF sổ tay nhân viên Việt Nam về thư mục `data\raw\pdf\`.

**Kiểm tra đã tải được gì:**

```cmd
dir data\raw\pdf\
```

Nên thấy các file như:

- `SO-TAY-NHAN-VIEN-THAI-SAN.pdf`
- `HACOM_So-tay-Nhan-vien-2025-2.pdf`
- v.v.

**Nếu thấy file không liên quan** (Medicare, luật giao thông, v.v.) thì xóa đi:

```cmd
del "data\raw\pdf\tên-file-rác.pdf"
```

Chạy lại nếu cần thêm file:

```cmd
python scripts/crawl_hr_pdfs.py --limit 20
```

---

## Bước 4 — Trích xuất và chia nhỏ nội dung PDF

```cmd
python scripts/ingest_pdf_handbooks.py
```

Kết quả xuất ra file `data\raw_chunks_viet.jsonl`.

Ví dụ output đúng:

```text
Found 8 PDFs in data\raw\pdf
  Processing: SO-TAY-NHAN-VIEN-THAI-SAN.pdf
    -> 350 chunks
  Processing: HACOM_So-tay-Nhan-vien-2025-2.pdf
    -> 280 chunks
  ...
Total: 1500 chunks -> data\raw_chunks_viet.jsonl
```

> Nếu file PDF nào hiện `-> 0 chunks` — đó là file scan ảnh, không đọc được chữ. Xóa đi và tìm file khác.

---

## Bước 5 — Chạy sinh câu hỏi qua đêm

Thay `gsk_KEY_CUA_BAN` bằng key đã copy ở Bước 2.

```powershell
python scripts/generate_qa.py --vllm-url https://api.groq.com/openai --api-key gsk_KEY_CUA_BAN --model llama-3.1-8b-instant --input data/raw_chunks_viet.jsonl --delay 2
```

**Bật lên rồi đi ngủ.** Kết quả lưu vào `data\qa_pairs_viet.jsonl`.

Trong lúc chạy sẽ hiển thị:

```text
[chunk 10/1500 | new 10] QA: 25 | ETA: 45.2min
[chunk 20/1500 | new 20] QA: 51 | ETA: 40.1min
...
Done. QA pairs: 3200 | Errors: 2 | Time: 87.3min
```

> **Bị ngắt giữa chừng không sao** — script tự lưu checkpoint. Chạy lại cùng lệnh là tiếp tục từ chỗ dừng.

---

## Bước 6 — Kiểm tra kết quả sáng hôm sau

```cmd
python -c "import json; pairs=open('data/qa_pairs_viet.jsonl',encoding='utf-8').readlines(); print('Tong so cap QA:', len(pairs)); p=json.loads(pairs[0]); print('Cau hoi mau:', p['question']); print('Tra loi mau:', p['answer'][:100])"
```

**Mục tiêu: trên 1000 cặp.** Nếu ít hơn 500 thì quay lại Bước 3 tải thêm PDF.

---

## Bước 7 — Chuyển sang CSV

```cmd
python -c "import json,csv; pairs=[json.loads(l) for l in open('data/qa_pairs_viet.jsonl',encoding='utf-8')]; f=open('data/qa_training_data_viet.csv','w',encoding='utf-8',newline=''); w=csv.DictWriter(f,fieldnames=['question','answer']); w.writeheader(); w.writerows({'question':p['question'],'answer':p['answer']} for p in pairs); f.close(); print(len(pairs),'cap da luu vao data/qa_training_data_viet.csv')"
```

---

## Bước 8 — Commit và push lên GitHub

```cmd
git add data\qa_pairs_viet.jsonl data\qa_training_data_viet.csv
git commit -m "Add Vietnamese QA dataset from HR PDFs"
git push
```

---

## Bước 9 — Huấn luyện mô hình (bước tiếp theo, cần GPU)

Sau khi push xong, nhắn tin cho Minh để chạy Colab training với file CSV mới.

---

## Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
| --- | --- |
| `ModuleNotFoundError` | Chạy lại: `pip install openai httpx ddgs pymupdf beautifulsoup4` |
| `Found 0 candidate PDF URLs` | Kiểm tra mạng internet; thử `python scripts/crawl_hr_pdfs.py --dry-run` |
| PDF hiện `0 chunks` | File scan ảnh — xóa, tìm file khác có chữ thật |
| Lỗi `rate limit` từ Groq | Script tự thử lại; hoặc chờ 1 phút rồi chạy lại |
| Script bị tắt giữa chừng | Chạy lại cùng lệnh — tự tiếp tục từ chỗ dừng |
| Ít hơn 500 cặp QA | Tải thêm PDF: `python scripts/crawl_hr_pdfs.py --limit 20` |
| Không có Python | Tải tại: [python.org/downloads](https://www.python.org/downloads/) (chọn Python 3.11) |

---

**Xong Bước 8 thì nhắn Minh biết để bước tiếp theo.**
