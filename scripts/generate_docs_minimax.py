# -*- coding: utf-8 -*-
"""
Generate synthetic Vietnamese HR documents using Minimax API.

Produces 12 new HR policy documents covering topics not in the existing 8.
After running this script, run scripts/reindex_chromadb.py to rebuild ChromaDB.

Usage:
    python scripts/generate_docs_minimax.py              # generate HR docs only
    python scripts/generate_docs_minimax.py --qa         # also generate Q&A pairs
    python scripts/generate_docs_minimax.py --qa-only    # only generate Q&A pairs

Requirements:
    pip install openai python-dotenv
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "viet_labor_docs"
QA_OUTPUT = REPO_ROOT / "data" / "qa_minimax.jsonl"

MODEL = "MiniMax-M2.7"

# MiniMax has two endpoints — try both if one fails
MINIMAX_URLS = [
    "https://api.minimax.chat/v1",       # China / default
    "https://api.minimaxi.chat/v1",      # International (extra 'i')
]

# ---------------------------------------------------------------------------
# 12 new HR topics (not covered by the existing 8 hardcoded docs)
# ---------------------------------------------------------------------------
NEW_DOCUMENTS = [
    {
        "filename": "lam_them_gio.txt",
        "topic": "Chính sách làm thêm giờ (残業)",
        "prompt": (
            "Viết tài liệu nội bộ chi tiết về chính sách làm thêm giờ của công ty Việt Nam "
            "theo Bộ Luật Lao Động 2019. Bao gồm: giới hạn giờ làm thêm theo ngày/tháng/năm "
            "(tối đa 40 giờ/tháng, 200 giờ/năm), cách tính lương làm thêm giờ ngày thường "
            "(x1.5), ngày nghỉ tuần (x2.0), ngày lễ (x3.0), quy trình đăng ký làm thêm giờ, "
            "quyền từ chối của nhân viên, và trường hợp ngoại lệ. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "nghi_thai_san_va_phu_nu.txt",
        "topic": "Nghỉ thai sản và chính sách cho lao động nữ",
        "prompt": (
            "Viết tài liệu nội bộ về chính sách nghỉ thai sản và các quyền lợi dành cho lao động nữ "
            "theo Bộ Luật Lao Động Việt Nam 2019 và Luật BHXH. Bao gồm: thời gian nghỉ trước và sau sinh "
            "(tổng 6 tháng), quyền lợi tài chính (100% lương BHXH bình quân 6 tháng), "
            "nghỉ chăm con ốm, không sa thải lao động nữ mang thai, "
            "nghỉ nuôi con nhỏ dưới 12 tháng tuổi 60 phút/ngày, và hỗ trợ của công ty ngoài BHXH. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "chinh_sach_lam_viec_tu_xa.txt",
        "topic": "Chính sách làm việc từ xa (WFH)",
        "prompt": (
            "Viết tài liệu nội bộ về chính sách làm việc từ xa (Work From Home) của công ty. "
            "Bao gồm: điều kiện áp dụng WFH, số ngày WFH tối đa mỗi tuần, "
            "yêu cầu về thiết bị và đường truyền internet, quy định về giờ làm việc và phản hồi, "
            "công cụ họp trực tuyến bắt buộc, bảo mật dữ liệu khi làm từ xa, "
            "đánh giá hiệu suất khi WFH, và trường hợp hủy WFH. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "cong_tac_va_di_chuyen.txt",
        "topic": "Chính sách công tác và di chuyển",
        "prompt": (
            "Viết tài liệu nội bộ về chính sách công tác trong nước và quốc tế của công ty Việt Nam. "
            "Bao gồm: quy trình xin phê duyệt công tác, mức phụ cấp công tác theo tỉnh/thành phố "
            "và quốc tế (USD/ngày), thanh toán vé máy bay và khách sạn, mức phụ cấp ăn uống, "
            "bảo hiểm tai nạn khi công tác, quyết toán chi phí sau chuyến đi, "
            "và phụ cấp đặc biệt cho vùng xa. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "danh_gia_hieu_suat.txt",
        "topic": "Quy trình đánh giá hiệu suất nhân viên",
        "prompt": (
            "Viết tài liệu nội bộ về hệ thống đánh giá hiệu suất công việc (Performance Review) "
            "của công ty. Bao gồm: chu kỳ đánh giá (giữa năm và cuối năm), "
            "thang điểm đánh giá (1-5 hoặc A-E), tiêu chí KPI theo từng cấp bậc, "
            "quy trình self-assessment và manager review, calibration session giữa các phòng ban, "
            "liên kết giữa điểm đánh giá và thưởng/tăng lương, "
            "quy trình cải thiện hiệu suất (PIP) cho nhân viên không đạt. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "bao_mat_thong_tin_it.txt",
        "topic": "Chính sách bảo mật thông tin và an ninh IT",
        "prompt": (
            "Viết tài liệu nội bộ về chính sách bảo mật thông tin và an ninh IT của công ty Việt Nam. "
            "Bao gồm: phân loại dữ liệu (mật, nội bộ, công khai), quy định mật khẩu và xác thực 2 yếu tố, "
            "nghiêm cấm cài phần mềm lậu và truy cập trang web đen, "
            "quy định sử dụng USB và email cá nhân, báo cáo sự cố bảo mật, "
            "chính sách xóa dữ liệu khi thiết bị mất cắp (remote wipe), "
            "và hậu quả vi phạm bảo mật IT. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "nghi_le_phep_dac_biet.txt",
        "topic": "Nghỉ lễ và nghỉ phép đặc biệt",
        "prompt": (
            "Viết tài liệu nội bộ về lịch nghỉ lễ tết và các loại nghỉ phép đặc biệt của công ty "
            "theo quy định Việt Nam. Bao gồm: danh sách 11 ngày nghỉ lễ quốc gia (Tết Dương lịch, "
            "Tết Nguyên Đán 5 ngày, Giỗ Tổ Hùng Vương, 30/4, 1/5, 2/9), "
            "nghỉ phép đặc biệt có lương (kết hôn 3 ngày, con kết hôn 1 ngày, "
            "cha/mẹ/vợ/chồng/con mất 3 ngày, ông bà/anh chị em ruột mất 1 ngày), "
            "và chính sách hoán đổi ngày nghỉ lễ trùng cuối tuần. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "quay_roi_phan_biet_doi_xu.txt",
        "topic": "Phòng chống quấy rối và phân biệt đối xử",
        "prompt": (
            "Viết tài liệu nội bộ về chính sách phòng chống quấy rối tình dục và phân biệt đối xử "
            "tại nơi làm việc của công ty Việt Nam. Bao gồm: định nghĩa quấy rối tình dục "
            "và phân biệt đối xử theo giới tính, độ tuổi, tôn giáo, khuyết tật, "
            "ví dụ các hành vi bị nghiêm cấm, kênh tiếp nhận khiếu nại (HR/đường dây nóng ẩn danh), "
            "quy trình điều tra, biện pháp bảo vệ người tố cáo, và mức kỷ luật cao nhất (sa thải). "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "nghi_viec_va_ban_giao.txt",
        "topic": "Quy trình nghỉ việc và bàn giao",
        "prompt": (
            "Viết tài liệu nội bộ về quy trình nghỉ việc tự nguyện và các thủ tục bàn giao công việc. "
            "Bao gồm: thời gian thông báo trước (30 ngày HĐLĐ xác định thời hạn, 45 ngày không xác định), "
            "mẫu đơn xin thôi việc, quy trình bàn giao tài liệu và thiết bị, "
            "thanh toán các khoản còn lại (lương, phép chưa dùng, thưởng chưa nhận), "
            "chốt sổ BHXH, thủ tục thu hồi thẻ và tài khoản hệ thống, "
            "và phỏng vấn thôi việc (exit interview). "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "phong_chong_tham_nhung.txt",
        "topic": "Phòng chống tham nhũng và xung đột lợi ích",
        "prompt": (
            "Viết tài liệu nội bộ về chính sách phòng chống tham nhũng, hối lộ và xung đột lợi ích "
            "của công ty theo tiêu chuẩn quản trị doanh nghiệp Việt Nam. "
            "Bao gồm: nghiêm cấm nhận quà từ nhà cung cấp/khách hàng vượt quá 500.000 VNĐ, "
            "khai báo xung đột lợi ích (người thân làm việc ở đối thủ/nhà cung cấp), "
            "không được tham gia hoạt động kinh doanh cạnh tranh, "
            "kênh báo cáo vi phạm ẩn danh, và bảo vệ người tố cáo. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "suc_khoe_an_toan_lao_dong.txt",
        "topic": "An toàn lao động và sức khỏe nghề nghiệp",
        "prompt": (
            "Viết tài liệu nội bộ về an toàn lao động và sức khỏe nghề nghiệp (ATVSLĐ) "
            "theo Luật An Toàn Vệ Sinh Lao Động Việt Nam 2015. "
            "Bao gồm: quy định trang bị bảo hộ lao động, quy trình báo cáo tai nạn lao động, "
            "diễn tập phòng cháy chữa cháy (2 lần/năm), sơ cấp cứu tại chỗ, "
            "kiểm tra sức khỏe định kỳ cho lao động làm việc trong môi trường độc hại, "
            "quyền từ chối làm việc trong điều kiện nguy hiểm, và trách nhiệm của người sử dụng lao động. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
    {
        "filename": "phu_cap_va_tro_cap.txt",
        "topic": "Các loại phụ cấp và trợ cấp",
        "prompt": (
            "Viết tài liệu nội bộ liệt kê và giải thích tất cả các loại phụ cấp, trợ cấp mà "
            "nhân viên công ty có thể được hưởng. Bao gồm: phụ cấp chức vụ theo cấp bậc, "
            "phụ cấp trách nhiệm, phụ cấp độc hại/nặng nhọc, phụ cấp khu vực (tỉnh xa), "
            "trợ cấp thôi việc (0.5 tháng lương/năm làm việc), "
            "trợ cấp mất việc do cắt giảm (1 tháng lương/năm), "
            "phụ cấp nhà ở cho nhân viên được điều chuyển, và phụ cấp thâm niên. "
            "Viết bằng tiếng Việt, khoảng 500 từ, chuyên nghiệp."
        ),
    },
]

# ---------------------------------------------------------------------------
# Q&A generation prompt (applied per document)
# ---------------------------------------------------------------------------
QA_SYSTEM = (
    "Bạn là chuyên gia nhân sự. Từ tài liệu HR được cung cấp, hãy tạo ra các cặp câu hỏi-trả lời "
    "thực tế mà nhân viên Việt Nam thường hỏi bộ phận HR. "
    "Trả lời CHÍNH XÁC theo định dạng JSON array, không giải thích thêm."
)

QA_PROMPT_TEMPLATE = """\
Tài liệu:
{document_text}

Tạo 10 cặp câu hỏi-trả lời từ tài liệu trên. Câu hỏi phải tự nhiên như nhân viên hỏi HR.
Trả lời dựa chặt chẽ vào nội dung tài liệu.

Định dạng JSON:
[
  {{"question": "...", "answer": "..."}},
  ...
]"""


def load_api_key() -> str:
    """Load Minimax API key from .env/Minimax.env or environment."""
    key = os.environ.get("MINIMAX_API_KEY", "")
    if key:
        logger.info("API key loaded from environment variable.")
        return key

    env_path = REPO_ROOT / ".env" / "Minimax.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("MINIMAX_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    logger.info(f"API key loaded from {env_path.name}: {key[:8]}...{key[-4:]}")
                    return key

    logger.error("MINIMAX_API_KEY not found. Set it in .env/Minimax.env or as env var.")
    sys.exit(1)


def probe_endpoint(api_key: str):
    """Try each base URL with a cheap test call. Returns working (client, url)."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    for url in MINIMAX_URLS:
        logger.info(f"Probing: {url} ...")
        client = OpenAI(api_key=api_key, base_url=url)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            _ = resp.choices[0].message.content  # confirm it works
            logger.info(f"✅ Working endpoint: {url}")
            return client, url
        except Exception as e:
            err = str(e)
            logger.warning(f"  {url} → {err[:120]}")

    logger.error(
        "Both Minimax endpoints returned 401. Possible fixes:\n"
        "  1. Confirm the API key in .env/Minimax.env is correct\n"
        "  2. Check if your plan is active at minimax.chat or minimaxi.chat\n"
        "  3. The key might belong to a different service (not MiniMax chat API)\n"
        "  4. Try setting MINIMAX_BASE_URL env var to your actual endpoint"
    )
    sys.exit(1)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks that M2.x models emit."""
    import re
    # Remove everything inside <think>...</think> (including the tags)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def generate_text(client, system: str, user: str, max_tokens: int = 1200) -> str:
    """Single chat completion call with retry on rate limit."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            raw = response.choices[0].message.content
            return strip_thinking(raw)
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited — waiting {wait}s ...")
                time.sleep(wait)
            else:
                logger.error(f"API error: {e}")
                raise
    raise RuntimeError("Failed after 3 retries")


def generate_documents(client) -> int:
    """Generate all new HR documents. Returns count of files written."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    system = (
        "Bạn là chuyên gia nhân sự với hơn 15 năm kinh nghiệm tại các công ty lớn ở Việt Nam. "
        "Soạn thảo tài liệu nhân sự chuyên nghiệp, chính xác theo pháp luật Việt Nam hiện hành, "
        "văn phong rõ ràng, có số liệu cụ thể, dễ hiểu với nhân viên."
    )

    for doc in NEW_DOCUMENTS:
        out_path = OUTPUT_DIR / doc["filename"]
        if out_path.exists():
            logger.info(f"  Skip (exists): {doc['filename']}")
            written += 1
            continue

        logger.info(f"  Generating: {doc['topic']} ...")
        try:
            text = generate_text(client, system, doc["prompt"], max_tokens=2500)
            out_path.write_text(text, encoding="utf-8")
            logger.info(f"  Saved: {doc['filename']} ({len(text.split())} words)")
            written += 1
            time.sleep(1)  # polite pause between requests
        except Exception as e:
            logger.error(f"  FAILED {doc['filename']}: {e}")

    return written


def generate_qa_pairs(client) -> int:
    """Generate Q&A pairs from ALL txt files in OUTPUT_DIR. Returns pair count."""
    txt_files = sorted(OUTPUT_DIR.glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in data/viet_labor_docs/ — run doc generation first.")
        return 0

    all_pairs = []
    system = QA_SYSTEM

    for txt_path in txt_files:
        logger.info(f"  Q&A from: {txt_path.name}")
        doc_text = txt_path.read_text(encoding="utf-8")[:3000]  # first 3000 chars is enough
        prompt = QA_PROMPT_TEMPLATE.format(document_text=doc_text)

        try:
            raw = generate_text(client, system, prompt, max_tokens=2000)

            # Extract JSON array from response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                logger.warning(f"  No JSON array in response for {txt_path.name}")
                continue

            pairs = json.loads(raw[start:end])
            # Add source metadata
            for pair in pairs:
                pair["source"] = txt_path.name
            all_pairs.extend(pairs)
            logger.info(f"  Got {len(pairs)} pairs from {txt_path.name}")
            time.sleep(1)
        except json.JSONDecodeError as e:
            logger.warning(f"  JSON parse error for {txt_path.name}: {e}")
        except Exception as e:
            logger.error(f"  FAILED {txt_path.name}: {e}")

    if all_pairs:
        QA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with QA_OUTPUT.open("w", encoding="utf-8") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info(f"\nSaved {len(all_pairs)} Q&A pairs to {QA_OUTPUT}")

    return len(all_pairs)


def main():
    parser = argparse.ArgumentParser(description="Generate Vietnamese HR data via Minimax API")
    parser.add_argument("--qa", action="store_true", help="Also generate Q&A pairs after docs")
    parser.add_argument("--qa-only", action="store_true", help="Only generate Q&A pairs")
    args = parser.parse_args()

    api_key = load_api_key()
    client, active_url = probe_endpoint(api_key)
    logger.info(f"Using model: {MODEL}  |  base_url: {active_url}")

    if not args.qa_only:
        logger.info(f"\n=== Generating {len(NEW_DOCUMENTS)} HR documents ===")
        doc_count = generate_documents(client)
        logger.info(f"Documents done: {doc_count}/{len(NEW_DOCUMENTS)}")

    if args.qa or args.qa_only:
        logger.info("\n=== Generating Q&A pairs from all documents ===")
        qa_count = generate_qa_pairs(client)
        logger.info(f"Q&A pairs done: {qa_count}")

    logger.info("\n=== Next steps ===")
    logger.info("1. python scripts/reindex_chromadb.py   ← rebuild ChromaDB with all docs")
    logger.info("2. python cli_demo.py models/phi-3-mini.gguf   ← test chatbot")
    if args.qa or args.qa_only:
        logger.info(f"3. Q&A data at: {QA_OUTPUT}  ← use for fine-tuning if needed")


if __name__ == "__main__":
    main()
