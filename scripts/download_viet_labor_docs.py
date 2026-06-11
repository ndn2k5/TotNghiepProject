# -*- coding: utf-8 -*-
"""
Write synthetic Vietnamese HR policy documents to disk.

No LLM required — content is pre-written and saved directly.
Removes dependency on llama_cpp / llama.dll for knowledge-base population.

Usage:
    python scripts/download_viet_labor_docs.py

Output: data/viet_labor_docs/*.txt  (one file per topic)
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "viet_labor_docs"

DOCUMENTS = {
    "chinh_sach_nghi_phep.txt": """\
CHÍNH SÁCH NGHỈ PHÉP NĂM
Công ty TNHH ABC Việt Nam — Phòng Nhân sự

1. SỐ NGÀY NGHỈ PHÉP NĂM
Theo Bộ Luật Lao Động Việt Nam 2019 (Điều 113), người lao động làm đủ 12 tháng được nghỉ phép năm có lương như sau:
- Dưới 5 năm thâm niên: 12 ngày làm việc/năm
- Từ 5 đến dưới 10 năm thâm niên: 13 ngày làm việc/năm
- Từ 10 đến dưới 15 năm thâm niên: 14 ngày làm việc/năm
- Từ 15 năm thâm niên trở lên: 16 ngày làm việc/năm
Nhân viên làm việc chưa đủ 12 tháng được tính phép theo tỷ lệ số tháng đã làm việc.

2. QUY TRÌNH XIN NGHỈ PHÉP
- Nhân viên nộp đơn xin nghỉ phép qua hệ thống HRM ít nhất 3 ngày làm việc trước ngày nghỉ (trừ trường hợp khẩn cấp).
- Quản lý trực tiếp phê duyệt trong vòng 1 ngày làm việc.
- Nghỉ phép từ 3 ngày liên tiếp trở lên cần được Trưởng phòng và Phòng Nhân sự phê duyệt.
- Đơn xin nghỉ đã được phê duyệt có thể hủy tối đa 1 ngày trước ngày nghỉ.

3. TÍNH LƯƠNG KHI NGHỈ PHÉP
Trong thời gian nghỉ phép năm, nhân viên được hưởng nguyên lương theo hợp đồng lao động. Tiền lương nghỉ phép = Lương cơ bản / Số ngày làm việc chuẩn trong tháng × Số ngày nghỉ phép.

4. CHUYỂN PHÉP VÀ THANH TOÁN PHÉP
- Phép năm chưa sử dụng có thể chuyển sang năm sau tối đa 50% số ngày phép còn lại.
- Phép không được chuyển sẽ bị mất vào ngày 31 tháng 3 năm kế tiếp.
- Khi chấm dứt hợp đồng lao động, nhân viên được thanh toán tiền lương cho số ngày phép chưa sử dụng.

5. NGHỈ ỐM (NGHỈ BỆNH)
- Nhân viên nghỉ ốm từ 1 ngày phải thông báo cho quản lý trực tiếp trước 8:00 sáng.
- Nghỉ ốm từ 2 ngày trở lên cần có giấy xác nhận của cơ sở y tế có thẩm quyền.
- Chế độ ốm đau được hưởng theo quy định BHXH: 75% mức lương đóng BHXH, tối đa 30 ngày/năm nếu đóng BHXH dưới 15 năm.

6. NGHỈ KHÔNG LƯƠNG
Nhân viên có thể xin nghỉ không lương tối đa 30 ngày/năm vì lý do cá nhân. Đơn xin nghỉ không lương cần được Giám đốc Nhân sự phê duyệt. Trong thời gian nghỉ không lương, nhân viên vẫn được duy trì BHYT nhưng phải tự đóng toàn bộ BHXH/BHYT/BHTN.
""",

    "hop_dong_lao_dong.txt": """\
HỢP ĐỒNG LAO ĐỘNG VÀ THỬ VIỆC
Công ty TNHH ABC Việt Nam — Phòng Nhân sự

1. CÁC LOẠI HỢP ĐỒNG LAO ĐỘNG
Theo Bộ Luật Lao Động Việt Nam 2019 (Điều 20), công ty ký kết hợp đồng lao động theo các hình thức:
a) Hợp đồng xác định thời hạn: Thời hạn từ 12 tháng đến 36 tháng. Hết thời hạn, nếu không ký tiếp hoặc không chuyển sang hợp đồng không thời hạn, hợp đồng đương nhiên chấm dứt.
b) Hợp đồng không xác định thời hạn: Áp dụng sau khi đã ký tối đa 2 hợp đồng xác định thời hạn liên tiếp, hoặc ngay từ đầu cho các vị trí quản lý cấp cao.
c) Hợp đồng thời vụ: Thời hạn dưới 12 tháng, áp dụng cho công việc tạm thời hoặc theo mùa vụ.

2. THỜI GIAN THỬ VIỆC
Thời gian thử việc được quy định theo Điều 25 BLLĐ 2019:
- Vị trí quản lý cấp cao (Giám đốc, Trưởng phòng): tối đa 180 ngày
- Vị trí chuyên viên, kỹ sư có bằng đại học/cao đẳng: tối đa 60 ngày
- Vị trí lao động phổ thông, nhân viên hành chính: tối đa 30 ngày
Mỗi công việc chỉ được thử việc một lần duy nhất.

3. LƯƠNG TRONG THỜI GIAN THỬ VIỆC
Lương thử việc tối thiểu bằng 85% mức lương chính thức của vị trí đó theo quy định tại Điều 26 BLLĐ 2019. Công ty cam kết trả lương thử việc đúng hạn vào ngày 15 và ngày cuối tháng.

4. KẾT QUẢ THỬ VIỆC
- Trước khi kết thúc thử việc 5 ngày, quản lý trực tiếp thực hiện đánh giá hiệu suất thử việc.
- Nếu đạt yêu cầu: Ký hợp đồng chính thức và nhân viên được hưởng 100% lương từ ngày ký.
- Nếu không đạt: Thông báo bằng văn bản ít nhất 3 ngày trước khi kết thúc thử việc, nêu rõ lý do.

5. CHẤM DỨT HỢP ĐỒNG
a) Nhân viên muốn nghỉ việc phải thông báo trước:
   - Hợp đồng xác định thời hạn: 30 ngày
   - Hợp đồng không xác định thời hạn: 45 ngày
b) Công ty chấm dứt hợp đồng vì lý do kinh tế phải thông báo trước ít nhất 45 ngày và trả trợ cấp thôi việc.
c) Nhân viên vi phạm nghiêm trọng kỷ luật có thể bị chấm dứt hợp đồng mà không cần thông báo trước.

6. BẢO MẬT THÔNG TIN
Nhân viên cam kết không tiết lộ thông tin mật của công ty trong thời gian làm việc và 2 năm sau khi nghỉ việc. Vi phạm sẽ bị xử lý theo quy định pháp luật.
""",

    "chinh_sach_luong.txt": """\
CHÍNH SÁCH LƯƠNG, THƯỞNG VÀ PHỤ CẤP
Công ty TNHH ABC Việt Nam — Phòng Nhân sự

1. CƠ CẤU THU NHẬP
Thu nhập hàng tháng của nhân viên bao gồm:
- Lương cơ bản: theo thỏa thuận trong hợp đồng lao động
- Phụ cấp ăn trưa: 730.000 VNĐ/tháng (tương đương 35.000 VNĐ/ngày làm việc)
- Phụ cấp đi lại: 500.000 VNĐ/tháng
- Phụ cấp điện thoại (áp dụng cho vị trí kinh doanh, quản lý): 300.000 VNĐ/tháng
- Thưởng hiệu suất (KPI): chi trả hàng quý dựa trên kết quả đánh giá

2. CHU KỲ TRẢ LƯƠNG
- Kỳ 1: Ngày 15 hàng tháng — tạm ứng 40% lương cơ bản
- Kỳ 2: Ngày cuối tháng — chi trả phần còn lại sau khi khấu trừ thuế TNCN, BHXH/BHYT/BHTN
Nếu ngày trả lương trùng vào ngày nghỉ lễ hoặc cuối tuần, lương được chi trả vào ngày làm việc liền trước.

3. THƯỞNG THEO HIỆU SUẤT (KPI)
- Đánh giá KPI theo quý (Q1, Q2, Q3, Q4)
- Mức thưởng: Xuất sắc (≥110% KPI): 2 tháng lương/năm; Tốt (90–109%): 1 tháng lương/năm; Đạt (70–89%): 0,5 tháng lương/năm; Không đạt (<70%): không thưởng
- KPI được thống nhất giữa nhân viên và quản lý vào đầu mỗi năm tài chính.

4. THƯỞNG THÁNG 13 VÀ THƯỞNG TẾT
- Thưởng tháng 13: tương đương 1 tháng lương cơ bản, chi trả trước ngày 20 tháng 12 Dương lịch.
- Thưởng Tết Nguyên Đán: từ 0,5 đến 2 tháng lương tùy theo kết quả kinh doanh của công ty và đánh giá cá nhân. Chi trả trước nghỉ Tết 5 ngày làm việc.
- Điều kiện: nhân viên phải làm việc đủ 6 tháng trở lên trong năm mới được hưởng thưởng tháng 13 và thưởng Tết.

5. ĐIỀU CHỈNH LƯƠNG HÀNG NĂM
- Xét tăng lương định kỳ vào tháng 4 hàng năm (sau khi có kết quả kinh doanh quý I).
- Mức tăng bình quân: 7–12% tùy theo hiệu suất cá nhân và tình hình công ty.
- Nhân viên có thể đề xuất điều chỉnh lương bất thường khi được thăng chức hoặc khi thị trường thay đổi đáng kể.

6. THUẾ THU NHẬP CÁ NHÂN
Công ty thực hiện khấu trừ thuế TNCN tại nguồn theo biểu thuế lũy tiến. Nhân viên cần đăng ký người phụ thuộc tại Phòng Kế toán để được giảm trừ gia cảnh theo quy định.
""",

    "noi_quy_cong_ty.txt": """\
NỘI QUY CÔNG TY
Công ty TNHH ABC Việt Nam

1. GIỜ LÀM VIỆC
- Giờ làm việc chính thức: Thứ Hai đến Thứ Sáu, 8:00 – 17:30 (nghỉ trưa 12:00 – 13:30)
- Thứ Bảy: làm việc theo lịch luân phiên hoặc theo yêu cầu dự án, được thông báo trước ít nhất 3 ngày.
- Nhân viên đến muộn hoặc ra sớm quá 15 phút mà không có lý do chính đáng sẽ bị trừ nửa ngày công.
- Chấm công bằng thẻ từ hoặc nhận diện khuôn mặt. Nhờ người chấm công hộ là vi phạm kỷ luật nghiêm trọng.

2. TRANG PHỤC
- Trang phục chuyên nghiệp, lịch sự khi tiếp khách hoặc họp bên ngoài.
- Thứ Sáu hàng tuần được mặc trang phục thoải mái (casual Friday) nhưng không mặc quần short, áo ba lỗ, dép lê.
- Nhân viên tiếp xúc trực tiếp với khách hàng phải mặc đồng phục công ty vào các ngày Thứ Hai và Thứ Tư.

3. SỬ DỤNG THIẾT BỊ CÔNG TY
- Máy tính, điện thoại, xe công ty chỉ được sử dụng cho mục đích công việc.
- Không cài đặt phần mềm bản quyền trái phép lên thiết bị công ty.
- Khi mang thiết bị ra ngoài văn phòng phải đăng ký với bộ phận IT.
- Thiết bị hỏng do sử dụng sai mục đích hoặc bất cẩn, nhân viên chịu một phần chi phí sửa chữa.

4. CẤM HÚT THUỐC, RƯỢU BIA
- Tuyệt đối không hút thuốc trong tòa nhà, bao gồm nhà vệ sinh và khu vực kỹ thuật.
- Khu vực hút thuốc được bố trí tại sân ngoài tòa nhà, có biển chỉ dẫn.
- Không uống rượu bia trong giờ làm việc. Đến làm việc trong trạng thái say rượu có thể bị xử lý kỷ luật sa thải.

5. BẢO MẬT DỮ LIỆU
- Không chia sẻ tài liệu nội bộ, dữ liệu khách hàng ra ngoài khi chưa được phép.
- Mật khẩu máy tính phải đặt theo tiêu chuẩn IT: tối thiểu 8 ký tự, có chữ hoa, số và ký tự đặc biệt. Đổi mật khẩu mỗi 90 ngày.
- Màn hình máy tính phải khóa khi rời khỏi chỗ ngồi (Windows + L).
- Tài liệu bảo mật phải được hủy bằng máy hủy tài liệu, không được vứt vào thùng rác thông thường.

6. ỨNG XỬ VĂN PHÒNG
- Giữ gìn vệ sinh chung, không để thức ăn qua đêm tại bàn làm việc.
- Tắt điều hòa, đèn khi ra về vào cuối ngày (người ra sau cùng chịu trách nhiệm).
- Tôn trọng đồng nghiệp, không quấy rối, phân biệt đối xử dưới bất kỳ hình thức nào.
- Giải quyết xung đột qua đường nội bộ trước khi khiếu nại lên cấp trên hoặc cơ quan lao động.
""",

    "ky_luat_lao_dong.txt": """\
KỶ LUẬT LAO ĐỘNG
Công ty TNHH ABC Việt Nam — Phòng Nhân sự

1. CÁC MỨC KỶ LUẬT
Theo Điều 124 Bộ Luật Lao Động Việt Nam 2019, công ty áp dụng các hình thức kỷ luật:
a) Khiển trách bằng văn bản: Áp dụng cho vi phạm lần đầu, mức độ nhẹ.
b) Kéo dài thời hạn nâng lương không quá 6 tháng hoặc cách chức: Áp dụng cho vi phạm tái diễn hoặc mức độ trung bình.
c) Sa thải (chấm dứt hợp đồng): Áp dụng cho vi phạm nghiêm trọng theo Điều 125 BLLĐ 2019.

2. HÀNH VI DẪN ĐẾN KỶ LUẬT
Vi phạm nhẹ (khiển trách):
- Đi muộn, về sớm quá 3 lần trong 1 tháng
- Không hoàn thành nhiệm vụ được giao mà không có lý do chính đáng
- Sử dụng điện thoại cá nhân quá mức trong giờ làm việc

Vi phạm trung bình (kéo dài nâng lương hoặc cách chức):
- Vắng mặt không phép từ 3–5 ngày/năm
- Vi phạm quy định bảo mật thông tin mức độ nhẹ
- Gây mâu thuẫn, xung đột với đồng nghiệp ảnh hưởng đến môi trường làm việc

Vi phạm nghiêm trọng (sa thải):
- Trộm cắp tài sản công ty hoặc đồng nghiệp
- Tiết lộ bí mật kinh doanh gây thiệt hại cho công ty
- Quấy rối tình dục tại nơi làm việc
- Tự ý bỏ việc từ 5 ngày liên tiếp trở lên mà không có lý do
- Làm giả hồ sơ, chứng từ, gian lận trong công việc

3. QUY TRÌNH XỬ LÝ KỶ LUẬT
Bước 1: Phòng Nhân sự lập biên bản vi phạm có xác nhận của chứng kiến.
Bước 2: Thông báo bằng văn bản cho nhân viên vi phạm, nêu rõ hành vi và điều khoản vi phạm.
Bước 3: Tổ chức họp xử lý kỷ luật. Nhân viên có quyền trình bày ý kiến và có người bảo vệ quyền lợi (đại diện công đoàn hoặc người thân).
Bước 4: Ra quyết định kỷ luật bằng văn bản, có chữ ký của Ban Giám đốc.
Bước 5: Lưu hồ sơ kỷ luật tại Phòng Nhân sự tối thiểu 3 năm.

4. QUYỀN CỦA NGƯỜI LAO ĐỘNG KHI BỊ KỶ LUẬT
- Được thông báo trước ít nhất 5 ngày làm việc về cuộc họp kỷ luật.
- Được trình bày quan điểm và cung cấp bằng chứng tại cuộc họp.
- Được có đại diện công đoàn tham dự cuộc họp (nếu có).
- Được khiếu nại quyết định kỷ luật lên Hội đồng Hòa giải nội bộ trong vòng 5 ngày làm việc sau khi nhận quyết định.

5. KHÁNG CÁO
Nhân viên cho rằng quyết định kỷ luật không công bằng có thể:
- Nộp đơn kháng cáo lên Giám đốc Nhân sự trong 5 ngày làm việc.
- Yêu cầu hòa giải qua Trung tâm Hòa giải lao động quận/huyện.
- Khởi kiện tại Tòa án Nhân dân có thẩm quyền theo quy định BLLĐ 2019.
""",

    "phuc_loi_nhan_vien.txt": """\
PHÚC LỢI NHÂN VIÊN
Công ty TNHH ABC Việt Nam — Phòng Nhân sự

1. BẢO HIỂM SỨC KHỎE BỔ SUNG
Ngoài BHYT bắt buộc theo BHXH, công ty mua thêm gói bảo hiểm sức khỏe toàn diện cho toàn bộ nhân viên chính thức (sau 6 tháng làm việc):
- Khám, chữa bệnh ngoại trú: tối đa 10 triệu VNĐ/năm
- Nằm viện nội trú: tối đa 60 triệu VNĐ/năm
- Tai nạn 24/7: tối đa 100 triệu VNĐ
- Thai sản (cho nhân viên nữ): hỗ trợ thêm 5 triệu VNĐ/lần sinh
Nhân viên có thể mua thêm gói bổ sung cho người thân với phí ưu đãi qua công ty.

2. KHÁM SỨC KHỎE ĐỊNH KỲ
Toàn bộ nhân viên được khám sức khỏe định kỳ hàng năm tại cơ sở y tế đối tác. Gói khám bao gồm: xét nghiệm máu toàn bộ, siêu âm ổ bụng, chụp X-quang ngực, kiểm tra thị lực và răng miệng.

3. HỖ TRỢ HỌC PHÍ VÀ PHÁT TRIỂN NGHỀ NGHIỆP
- Ngân sách đào tạo cá nhân: 5 triệu VNĐ/năm/nhân viên cho các khóa học liên quan đến công việc.
- Hỗ trợ 50% học phí các chương trình đào tạo dài hạn (MBA, chứng chỉ nghề nghiệp quốc tế) nếu cam kết làm việc tối thiểu 2 năm sau khi hoàn thành.
- Thư viện nội bộ với hàng trăm đầu sách chuyên ngành và tài khoản học trực tuyến (Coursera, LinkedIn Learning).

4. TEAM BUILDING VÀ CÁC HOẠT ĐỘNG TẬP THỂ
- Du lịch team building hàng năm: 2–3 ngày, toàn bộ chi phí do công ty chi trả.
- Tiệc cuối năm (Year-end Party) với phần thưởng bốc thăm may mắn.
- Các câu lạc bộ thể thao (bóng đá, cầu lông, yoga) được công ty tài trợ một phần.
- Ngày hội thể thao nội bộ tổ chức mỗi quý với giải thưởng hấp dẫn.

5. QUÀ TẶNG VÀ HỖ TRỢ ĐẶC BIỆT
- Sinh nhật: phiếu quà tặng 500.000 VNĐ
- Kết hôn: hỗ trợ 2 triệu VNĐ và 3 ngày phép có lương
- Sinh con: hỗ trợ 3 triệu VNĐ (ngoài thai sản BHXH)
- Hiếu: hỗ trợ 2 triệu VNĐ và 3 ngày phép có lương khi cha/mẹ/vợ/chồng/con mất
- Tết Nguyên Đán: quà tặng cho toàn bộ nhân viên và con em dưới 15 tuổi (200.000 VNĐ/trẻ)

6. CHÍNH SÁCH ĂN TRƯA
Căng tin nội bộ phục vụ bữa trưa với giá ưu đãi 25.000 VNĐ/suất (công ty hỗ trợ thêm 10.000 VNĐ/suất). Nhân viên không dùng căng tin được nhận phụ cấp ăn trưa 35.000 VNĐ/ngày làm việc tính vào lương hàng tháng.
""",

    "tuyen_dung_dao_tao.txt": """\
TUYỂN DỤNG VÀ ĐÀO TẠO
Công ty TNHH ABC Việt Nam — Phòng Nhân sự

1. QUY TRÌNH TUYỂN DỤNG
Bước 1 — Đề xuất tuyển dụng: Trưởng bộ phận nộp "Phiếu yêu cầu tuyển dụng" lên Phòng Nhân sự, nêu rõ vị trí, số lượng, mô tả công việc và yêu cầu ứng viên.
Bước 2 — Đăng tuyển: Phòng Nhân sự đăng tin tuyển dụng trên LinkedIn, VietnamWorks, TopCV và website nội bộ trong vòng 3 ngày làm việc sau khi nhận phiếu.
Bước 3 — Sàng lọc hồ sơ: Phòng Nhân sự sàng lọc trong vòng 5 ngày làm việc, chuyển hồ sơ đạt tiêu chuẩn cho bộ phận liên quan.
Bước 4 — Phỏng vấn vòng 1: Phỏng vấn năng lực chuyên môn với Trưởng/Phó bộ phận (30–60 phút).
Bước 5 — Bài kiểm tra thực tế: Ứng viên vào vòng 2 thực hiện bài test kỹ năng liên quan đến vị trí tuyển dụng.
Bước 6 — Phỏng vấn vòng 2: Phỏng vấn với Giám đốc Nhân sự hoặc Ban Giám đốc (45 phút).
Bước 7 — Offer: Phòng Nhân sự gửi thư mời nhập việc (offer letter) trong vòng 2 ngày làm việc sau khi có quyết định tuyển dụng.

2. ƯU TIÊN TUYỂN DỤNG NỘI BỘ
Tất cả vị trí tuyển dụng được thông báo nội bộ trước ít nhất 5 ngày làm việc. Nhân viên hiện tại đủ điều kiện được khuyến khích ứng tuyển và được ưu tiên xem xét so với ứng viên bên ngoài.

3. CHƯƠNG TRÌNH ONBOARDING
Tuần 1: Định hướng công ty (lịch sử, văn hóa, sản phẩm), thủ tục hành chính (hợp đồng, BHXH, tài khoản ngân hàng), cấp phát thiết bị làm việc.
Tháng 1: Gặp gỡ các bộ phận liên quan, được phân công "người hướng dẫn" (buddy) giúp hòa nhập.
Tháng 2–3: Nhận nhiệm vụ thực tế đầu tiên, đánh giá 60 ngày để đảm bảo việc hòa nhập tốt.

4. ĐÀO TẠO BẮT BUỘC
Tất cả nhân viên phải hoàn thành các khóa đào tạo bắt buộc hàng năm:
- An toàn thông tin và bảo mật dữ liệu (2 giờ, online)
- Phòng chống quấy rối tại nơi làm việc (1 giờ, online)
- Tuân thủ pháp lý và đạo đức kinh doanh (2 giờ, online)
- Sơ cấp cứu (4 giờ, offline, 2 năm/lần)

5. KẾ HOẠCH PHÁT TRIỂN CÁ NHÂN (IDP)
Mỗi nhân viên xây dựng Kế hoạch Phát triển Cá nhân hàng năm cùng với quản lý trực tiếp. IDP bao gồm: mục tiêu nghề nghiệp 1–3 năm, kỹ năng cần phát triển, các khóa đào tạo dự kiến, và KPI hỗ trợ thăng tiến.

6. THĂNG TIẾN NỘI BỘ
Xét thăng chức 2 lần/năm (tháng 4 và tháng 10). Tiêu chí: hoàn thành ≥90% KPI trong 2 kỳ liên tiếp, có năng lực lãnh đạo (đối với vị trí quản lý), và được quản lý trực tiếp đề xuất.
""",

    "bao_hiem_xa_hoi.txt": """\
BẢO HIỂM XÃ HỘI, BẢO HIỂM Y TẾ VÀ BẢO HIỂM THẤT NGHIỆP
Công ty TNHH ABC Việt Nam — Phòng Nhân sự

1. TỶ LỆ ĐÓNG BẢO HIỂM (2024)
Theo Luật BHXH và các nghị định hiện hành, tỷ lệ đóng như sau:

Bảo hiểm xã hội (BHXH):
- Công ty đóng: 17,5% lương đóng BHXH (trong đó: hưu trí/tử tuất 14%, ốm đau/thai sản 3%, tai nạn lao động 0,5%)
- Người lao động đóng: 8% lương đóng BHXH (hưu trí/tử tuất)

Bảo hiểm y tế (BHYT):
- Công ty đóng: 3% lương đóng BHXH
- Người lao động đóng: 1,5% lương đóng BHXH

Bảo hiểm thất nghiệp (BHTN):
- Công ty đóng: 1% lương đóng BHXH
- Người lao động đóng: 1% lương đóng BHXH

Tổng khấu trừ từ lương nhân viên: 10,5% lương đóng BHXH.

2. MỨC LƯƠNG ĐÓNG BHXH
Lương đóng BHXH tối thiểu bằng mức lương tối thiểu vùng. Lương đóng BHXH tối đa bằng 20 lần lương cơ sở (hiện tại = 36 triệu VNĐ/tháng). Phụ cấp ăn trưa và đi lại không tính vào lương đóng BHXH.

3. QUYỀN LỢI BHXH
a) Ốm đau: Hưởng 75% mức lương đóng BHXH, tối đa:
   - 30 ngày/năm nếu đóng BHXH dưới 15 năm
   - 40 ngày/năm nếu đóng từ 15 đến dưới 30 năm
   - 60 ngày/năm nếu đóng từ 30 năm trở lên

b) Thai sản:
   - Nghỉ khám thai: 5 lần × tối đa 2 ngày/lần
   - Nghỉ sinh: 6 tháng (sinh thường), có thể gia hạn thêm nếu sinh đôi trở lên
   - Hưởng 100% mức lương đóng BHXH bình quân 6 tháng trước nghỉ

c) Hưu trí: Đủ điều kiện hưởng lương hưu khi đủ tuổi (60 tuổi nam, 55 tuổi nữ năm 2024) và đóng BHXH đủ 20 năm.

4. BẢO HIỂM THẤT NGHIỆP
Hưởng trợ cấp thất nghiệp khi: chấm dứt hợp đồng lao động không do lỗi của người lao động, đang đóng BHTN, và đăng ký thất nghiệp tại Trung tâm Dịch vụ Việc làm.
- Mức hưởng: 60% mức lương bình quân 6 tháng trước khi nghỉ
- Thời gian hưởng: 3 tháng (đóng đủ 12–36 tháng), cộng thêm 1 tháng cho mỗi 12 tháng đóng thêm

5. THỦ TỤC KHI NGHỈ VIỆC
Khi chấm dứt hợp đồng, Phòng Nhân sự hỗ trợ:
- Chốt sổ BHXH trong vòng 14 ngày làm việc
- Trả sổ BHXH bản gốc cho nhân viên
- Cấp xác nhận thời gian đóng BHTN để đăng ký thất nghiệp

6. TRA CỨU THÔNG TIN BHXH
Nhân viên có thể tra cứu quá trình đóng BHXH, BHYT qua:
- Ứng dụng VssID (Bảo hiểm Xã hội Số)
- Website: baohiemxahoi.gov.vn
- Hoặc liên hệ trực tiếp Phòng Nhân sự để được hỗ trợ
""",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    written = 0
    for filename, content in DOCUMENTS.items():
        out_path = OUTPUT_DIR / filename
        if out_path.exists():
            logger.info(f"  Already exists, skipping: {filename}")
            written += 1
            continue
        out_path.write_text(content, encoding="utf-8")
        word_count = len(content.split())
        logger.info(f"  Saved {filename} ({word_count} words)")
        written += 1

    logger.info(f"\nDone. {written}/{len(DOCUMENTS)} documents written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
