# -*- coding: utf-8 -*-
"""
Generate synthetic Vietnamese HR policy documents using the local phi-3-mini.gguf model.

Each document covers a real Vietnamese HR topic and is written in natural Vietnamese
so that the ChromaDB knowledge base can answer questions about leave, salary, contracts,
etc. in Vietnamese.

Usage:
    python scripts/download_viet_labor_docs.py

Output: data/viet_labor_docs/*.txt  (one file per topic)

Requirements:
    - models/phi-3-mini.gguf must exist on disk (already downloaded)
    - llama_cpp must be installed (pip install llama-cpp-python)
    - ~2-3 GB RAM for model load
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "phi-3-mini.gguf"
OUTPUT_DIR = REPO_ROOT / "data" / "viet_labor_docs"

# ---------------------------------------------------------------------------
# Document prompts  (topic name → Vietnamese system-prompt)
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "filename": "chinh_sach_nghi_phep.txt",
        "topic": "Chính sách nghỉ phép năm",
        "prompt": (
            "Hãy viết một tài liệu nội bộ chi tiết về chính sách nghỉ phép năm của công ty "
            "theo quy định của Bộ Luật Lao Động Việt Nam. "
            "Tài liệu phải bao gồm: số ngày nghỉ phép theo thâm niên, quy trình xin nghỉ phép, "
            "cách tính lương khi nghỉ phép, nghỉ không lương, và nghỉ do bệnh. "
            "Viết bằng tiếng Việt, khoảng 400-500 từ, văn phong chuyên nghiệp."
        ),
    },
    {
        "filename": "hop_dong_lao_dong.txt",
        "topic": "Hợp đồng lao động và thử việc",
        "prompt": (
            "Hãy viết một tài liệu nội bộ chi tiết về quy định hợp đồng lao động và thời gian thử việc "
            "của công ty theo Bộ Luật Lao Động Việt Nam 2019. "
            "Tài liệu phải bao gồm: các loại hợp đồng (xác định thời hạn, không xác định thời hạn), "
            "thời gian thử việc, lương thử việc, quyền chấm dứt hợp đồng của hai bên, "
            "và các điều khoản bảo mật thông tin. "
            "Viết bằng tiếng Việt, khoảng 400-500 từ, văn phong chuyên nghiệp."
        ),
    },
    {
        "filename": "chinh_sach_luong.txt",
        "topic": "Chính sách lương và thưởng",
        "prompt": (
            "Hãy viết một tài liệu nội bộ chi tiết về chính sách lương, thưởng và phụ cấp của công ty. "
            "Tài liệu phải bao gồm: chu kỳ trả lương, cách tính lương theo hiệu suất (KPI), "
            "thưởng tháng 13, thưởng Tết Nguyên Đán, phụ cấp ăn trưa và đi lại, "
            "điều chỉnh lương hàng năm. "
            "Viết bằng tiếng Việt, khoảng 400-500 từ, văn phong chuyên nghiệp."
        ),
    },
    {
        "filename": "noi_quy_cong_ty.txt",
        "topic": "Nội quy công ty",
        "prompt": (
            "Hãy viết một bản nội quy công ty đầy đủ theo tiêu chuẩn Việt Nam. "
            "Nội quy phải bao gồm: giờ làm việc, quy định về trang phục, sử dụng thiết bị công ty, "
            "cấm hút thuốc và uống rượu bia tại văn phòng, quy định về bảo mật dữ liệu, "
            "nghĩa vụ báo cáo và chấp hành mệnh lệnh cấp trên, và các điều cấm khác. "
            "Viết bằng tiếng Việt, khoảng 400-500 từ, văn phong chuyên nghiệp."
        ),
    },
    {
        "filename": "ky_luat_lao_dong.txt",
        "topic": "Kỷ luật lao động",
        "prompt": (
            "Hãy viết một tài liệu nội bộ về quy trình xử lý kỷ luật lao động của công ty "
            "theo Bộ Luật Lao Động Việt Nam. "
            "Tài liệu phải bao gồm: các mức kỷ luật (khiển trách, cảnh cáo, sa thải), "
            "các hành vi vi phạm dẫn đến kỷ luật, quy trình họp xử lý kỷ luật, "
            "quyền của người lao động khi bị xử lý kỷ luật, và hình thức kháng cáo. "
            "Viết bằng tiếng Việt, khoảng 400-500 từ, văn phong chuyên nghiệp."
        ),
    },
    {
        "filename": "phuc_loi_nhan_vien.txt",
        "topic": "Phúc lợi nhân viên",
        "prompt": (
            "Hãy viết một tài liệu nội bộ chi tiết về các phúc lợi dành cho nhân viên của công ty. "
            "Tài liệu phải bao gồm: bảo hiểm sức khỏe bổ sung, khám sức khỏe định kỳ, "
            "hỗ trợ học phí và phát triển nghề nghiệp, các hoạt động team building, "
            "chính sách ưu đãi mua sản phẩm công ty, quà tặng sinh nhật và ngày lễ, "
            "và các phúc lợi đặc biệt cho nhân viên xuất sắc. "
            "Viết bằng tiếng Việt, khoảng 400-500 từ, văn phong chuyên nghiệp."
        ),
    },
    {
        "filename": "tuyen_dung_dao_tao.txt",
        "topic": "Tuyển dụng và đào tạo",
        "prompt": (
            "Hãy viết một tài liệu nội bộ về quy trình tuyển dụng và chương trình đào tạo của công ty. "
            "Tài liệu phải bao gồm: các bước tuyển dụng (đăng tin, sàng lọc hồ sơ, phỏng vấn, "
            "test năng lực, offer), chương trình onboarding nhân viên mới, kế hoạch đào tạo "
            "hàng năm, ngân sách đào tạo, và chính sách thăng tiến nội bộ. "
            "Viết bằng tiếng Việt, khoảng 400-500 từ, văn phong chuyên nghiệp."
        ),
    },
    {
        "filename": "bao_hiem_xa_hoi.txt",
        "topic": "Bảo hiểm xã hội và y tế",
        "prompt": (
            "Hãy viết một tài liệu nội bộ giải thích chế độ bảo hiểm xã hội, bảo hiểm y tế "
            "và bảo hiểm thất nghiệp theo quy định của Luật Bảo hiểm Xã hội Việt Nam. "
            "Tài liệu phải bao gồm: tỷ lệ đóng BHXH/BHYT/BHTN của công ty và người lao động, "
            "các quyền lợi được hưởng (ốm đau, thai sản, tai nạn lao động), "
            "cách tra cứu sổ BHXH, và thủ tục giải quyết khi nghỉ việc. "
            "Viết bằng tiếng Việt, khoảng 400-500 từ, văn phong chuyên nghiệp."
        ),
    },
]


def load_llm(model_path: Path) -> object:
    """Load phi-3-mini.gguf using llama_cpp with GPU layers (-1) falling back to CPU (0)."""
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama_cpp not installed. Run: pip install llama-cpp-python")
        sys.exit(1)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    # Try GPU first
    logger.info(f"Loading model: {model_path.name} (GPU preferred) ...")
    try:
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,   # offload all layers to GPU
            n_ctx=2048,
            n_batch=512,
            verbose=False,
        )
        logger.info("Model loaded with GPU layers.")
        return llm
    except Exception as gpu_err:
        logger.warning(f"GPU load failed ({gpu_err}), falling back to CPU ...")

    # CPU fallback
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=0,
        n_ctx=2048,
        n_batch=512,
        verbose=False,
    )
    logger.info("Model loaded on CPU.")
    return llm


def generate_document(llm: object, topic: str, prompt: str) -> str:
    """Send a single prompt to the LLM and return the generated text."""
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là chuyên gia nhân sự với nhiều năm kinh nghiệm tại các công ty Việt Nam. "
                "Hãy soạn thảo tài liệu nhân sự chuyên nghiệp, chính xác theo quy định pháp luật Việt Nam."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    logger.info(f"  Generating: {topic} ...")

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=700,
            temperature=0.4,
            top_p=0.9,
            stop=["<|end|>", "<|user|>"],
        )
        text = response["choices"][0]["message"]["content"].strip()
        return text
    except Exception as e:
        logger.error(f"  Generation failed for '{topic}': {e}")
        return f"[ERROR] Could not generate content for {topic}: {e}"


def main() -> None:
    """Main entry point — generate all Vietnamese HR documents."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Load model once, reuse for all documents
    llm = load_llm(MODEL_PATH)

    success_count = 0
    for doc in DOCUMENTS:
        out_path = OUTPUT_DIR / doc["filename"]

        # Skip if already generated (re-run safety)
        if out_path.exists():
            logger.info(f"  Already exists, skipping: {doc['filename']}")
            success_count += 1
            continue

        text = generate_document(llm, doc["topic"], doc["prompt"])

        # Write UTF-8 with BOM-free encoding
        out_path.write_text(text, encoding="utf-8")
        word_count = len(text.split())
        logger.info(f"  Saved {doc['filename']} ({word_count} words)")
        success_count += 1

    logger.info(
        f"\nDone. {success_count}/{len(DOCUMENTS)} documents written to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
