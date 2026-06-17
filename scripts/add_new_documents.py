#!/usr/bin/env python3
"""
Add new HR documents to ChromaDB from fetched web content
"""
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import VectorStoreManager, LocalEmbedder
from src.hybrid_retriever import HybridRetriever

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into chunks with overlap"""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Keep last words for overlap
            overlap_words = int(len(current_chunk) * overlap / chunk_size)
            current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
            current_length = sum(len(w) + 1 for w in current_chunk)
        
        current_chunk.append(word)
        current_length += word_length
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_documents_from_content():
    """Create documents from FPD Education and FPT Software handbook content"""
    
    documents = []
    doc_id = 257  # Start from 257 (256 existing docs)
    
    # ===== FPD Education Employee Handbook 2024 =====
    fpd_content = """
    FPD Education Employee Handbook 2024 - Sổ tay nhân viên
    
    GIỚI THIỆU CHUNG:
    - Tổ chức Giáo dục FPT là một đơn vị thành viên của Tập đoàn FPT
    - Sứ mệnh: Cung cấp năng lực cạnh tranh toàn cầu cho đông đảo người học
    - Tầm nhìn: Trở thành một hệ thống giáo dục mang tính quốc tế
    
    2. HỢP ĐỒNG SỬ DỤNG NHÂN LỰC:
    - Tuyển dụng công khai thông qua các phương tiện thông tin
    - Thưởng golden cho CBGV giới thiệu ứng viên có trình độ Tiến sĩ, Thạc sĩ
    
    3. QUYỀN LỢI CỦA NGƯỜI LAO ĐỘNG:
    
    3.1. XẾP CẤP CÁN BỘ:
    - Cấp cán bộ được xác định khi CBGV ký HDLD chính thức
    - Quyền lợi bảo hiểm sức khỏe và tai nạn FPT Care theo cấp cán bộ
    - Định kỳ hàng năm, lãnh đạo đơn vị rà soát lại cấp cán bộ
    
    3.2. TIÊU CHUẨN CHỨC DANH NGHỀ NGHIỆP GIẢNG VIÊN:
    - Giảng viên tập sự (GVTS)
    - Giảng viên (GV)
    - Giảng viên chính (GVC)
    - Giảng viên cao cấp (GVCC)
    - Xét chúc danh vào đầu mỗi học kỳ
    
    3.3. CẤU TRÚC THU NHẬP CỦA CBGV:
    - Tiền lương: Lương theo chức danh công việc + khoán bổ sung không định hạn
    - Tiền lương tháng 13 theo hiệu quả hoạt động
    - Khoán lương đặc thù áp dụng cho một số vị trí hoặc đơn vị
    - Các khoán phúc lợi theo chính sách của Trường
    
    3.4. TIỀN LƯƠNG THÁNG:
    - Tiền lương tháng kế hoạch được xác định căn cứ vào:
      * Chức danh công việc
      * Trình độ chuyên môn kỹ thuật
      * Kinh nghiệm
      * Mặt bằng thu nhập trung bình của thị trường
    - Lương cơ bản: Căn cứ trên trình độ, kinh nghiệm, kỹ năng làm việc
    - Khoán bổ sung kế hoạch: Phần chênh lệch giữa tiền lương tháng kế hoạch và lương cơ bản
    - Khoán bổ sung thực tế: Tính trên số ngày làm việc thực tế trong tháng và hiệu quả thực hiện công việc
    
    3.5. TIỀN THƯỞNG LƯƠNG THÁNG THỨ 13:
    - Cán bộ tuyên sinh: Tính theo khối lượng và kết quả công việc thực hiện trong năm
    - Giảng viên: Tương đương 88 giờ giảng dạy mức F hệ số 1 hoặc 1 tháng tiền lương kế hoạch
    
    3.6. NGÀY PHÉP VÀ NGHỈ NGƠI:
    
    3.6.1. NGÀY PHÉP HÀNG NĂM:
    - CBGV ký HDLD có thời gian làm việc từ đủ 12 tháng trở lên có thể tạm ứng ngay nghỉ trong năm
    - Không được vượt quá tổng số ngày phép của năm đó nếu được Trường đồng ý
    - CBGV không sử dụng ngày phép hàng năm sẽ không được thanh toán lương
    - Khi chấm dứt HDLD mà chưa sử dụng hết ngày phép hàng năm được huởng theo quy định
    - Nếu CBGV sử dụng quá số ngày phép được huởng thì phải nộp lại tiền lương
    
    3.6.2. NGÀY PHÉP THEO QUY ĐỊNH:
    - Căn cứ vào chiều dài thời gian làm việc tại Trường
    - Trường hợp CBGV không bo trí được nghi do yêu cầu công việc: chuyên số ngày nghỉ sang hết quý 1 năm tiếp theo
    
    3.7. NGÀY NGHỈ KỊP HÔN, NGHỈ TANG:
    - Kết hôn: 03 ngày hưởng nguyên lương
    - Con dâu/con trai của CBGV kết hôn: 01 ngày hưởng nguyên lương
    - Cha dạo/mẹ dạo/cha nuôi/mẹ nuôi qua đời: 03 ngày hưởng nguyên lương
    - Ông nội/bà nội/ông ngoại/bà ngoại/anh/chị/em ruột cha/mẹ kết hôn: 01 ngày không hưởng lương
    
    3.8. NGÀY NGHỈ KHÔNG HƯỞNG LƯƠNG:
    - Dành cho CBGV ký HDLD có thời hạn từ 12-36 tháng hoặc HDLD không xác định thời hạn
    - Hỗ trợ giải quyết các vấn đề gia đình hoặc bản thân
    - Tối đa 01 tháng cho một lần xin nghỉ
    - Nếu xin lúc HDLD sắp hết hạn, chỉ phê duyệt từ lúc xin đến ngày hết hạn HDLD
    - Nếu quá 14 ngày làm việc trong tháng: không được hưởng các khoản hỗ trợ phục vụ công việc
    
    3.9. NGÀY NGHỈ HƯỞNG CHẾ BHXH:
    - Ngày nghỉ khám thai
    - Ngày nghỉ sinh con
    - Ngày nghỉ sảy thai
    - Ngày nghỉ chăm sóc con ốm
    - Ngày nghỉ bản thân ốm
    - Chi tiết tại Quy định về chế độ nghỉ trang 12
    
    3.10. NGÀY NGHỈ HÈ, NGÀY NGHỈ MÁT:
    - Hàng năm Trường quy định số ngày nghỉ hè cho CBGV
    - Số ngày nghỉ này không bị trừ vào ngày nghỉ hàng năm
    - Thời gian nghỉ: 01 tuần (thời gian cụ thể do Trường quy định theo lịch hàng năm)
    - Điều kiện và định mức tiêu chuẩn nghỉ mát do Trường quy định hàng năm
    - CBGV được hưởng nguyên tiền lương trong thời gian này
    
    3.11. NGÀY NGHỈ NHÂN NGÀY THÀNH LẬP TẬP FPT:
    - Tất cả CBGV được nghi để tham gia các hoạt động thể thao, văn hóa, văn nghệ do Tập đoàn tổ chức
    - Được hưởng nguyên lương (cụ thể theo lịch tổ chức lễ hội hàng năm)
    
    3.12. CHẾ ĐỘ ĐI CÔNG TÁC:
    - CBGV khi đi công tác phải tuân theo quy định của Trường và Bộ phân tài chính
    - Được hưởng nguyên lương trong thời gian đi công tác
    - Chi tiết Quy định chế độ công tác phí trong nước
    
    3.13. ĐÁNH GIÁ CÔNG VIỆC:
    - Kết quả đánh giá là yếu tố quan trọng để xem xét việc tăng lương, thưởng, khả năng thăng tiến
    - Hay hạ cấp hoặc thôi việc của CBGV
    
    3.14. THĂNG TIẾN ĐỀ BẠT:
    - Tất cả CBGV trong Trường có đầy đủ khả năng và các tố chất/tiêu chuẩn phù hợp
    - Có thể được cán bộ quản lý, dạo tạo, bồi dưỡng và giới thiệu vào các vị trí đó
    - Các trường hợp đề bạt được quyết định trên cơ sở thảo dò ý kiến đồng sự
    - CBGV được xét đề bạt phải có ít nhất 01 năm làm việc tại Trường (trừ các trường hợp đặc biệt)
    - Được đa số CBGV được mời tham gia đánh giá, bỏ phiếu tán thành
    
    3.15. ĐÀO TẠO:
    - Trường khuyến khích và tạo điều kiện cho CBGV học hỏi, trau dồi kỹ năng, kiến thức
    - Mỗi CBGV được đào tạo ít nhất một lần trong năm theo định hướng đào tạo phục vụ công việc hiện tại
    - Trường tạo điều kiện tham gia đào tạo các kiến thức và kỹ năng phù hợp
    - CBGV có quyền đề nghị Trường hỗ trợ về chi phí
    - Hàng năm Trường ban hành Quy định đào tạo nội bộ gồm đối tượng áp dụng, yêu cầu học tập
    - Mỗi CBGV (trừ bảo vệ, tập vụ, lái xe) có tổng thời gian làm việc từ 90 ngày trở lên phải hoàn thành 30 giờ học tập
    - Có ít nhất 01 khóa học MOOC thuộc danh mục ban hành
    
    3.16. CHẾ ĐỘ ĐI ĐÀO TẠO TIẾN SĨ:
    - Đối tượng áp dụng: CBGV là công dân Việt Nam ký HDLD làm việc toàn thời gian
    - Hoặc cán bộ nhân viên ký HDLD làm việc toàn thời gian đi đào tạo tiến sĩ tập trung toàn thời gian
    - Tổng tiền hỗ trợ tối đa 200 triệu đồng
    - Hưởng tiền lương ghi trên HDLD, GV vẫn hưởng thù lao từ các công việc thực hiện trong thời gian đi đào tạo
    - Được tham gia đóng bảo hiểm bắt buộc
    - Thời gian hưởng chế độ: Căn cứ theo thời gian đào tạo thực tế nhưng không quá 03 năm
    - Thời gian yêu cầu làm việc sau đào tạo: Bằng thời gian hưởng chế độ hỗ trợ
    
    3.17. CHẾ ĐỘ BỒI THƯỜNG CHI PHÍ ĐÀO TẠO:
    - CBGV được hưởng chế độ hỗ trợ đi đào tạo tự ý bỏ việc: Bồi thường 1.5 lần chi phí đào tạo
    - Hoặc không được số đào tạo cấp văn bằng tốt nghiệp: Bồi thường 1.5 lần
    - Hoặc sau đào tạo chưa về làm việc mà tự ý bỏ việc: Bồi thường 1.5 lần
    - Hoặc sau đào tạo về làm việc nhưng chưa làm hết thời gian: Bồi thường 1.5 lần theo thời gian chưa làm đủ
    
    3.18. KHEN THƯỞNG THÀNH TÍCH NGHIÊN CỨU KHOA HỌC:
    - Khen thưởng bài báo khoa học
    - Mức thưởng căn cứ vào chất lượng tạp chí, số lượng tác giả, số lượng đơn vị chủ quản
    - Khen thưởng bằng độc quyền sáng chế và giải pháp hữu ích
    - Khen thưởng theo số lượng trích dẫn trên Scopus và Google Scholar
    
    3.19. BẢO HIỂM CHĂM SÓC SỨC KHỎE FPTCARE:
    - CBGV ký HDLD với Trường được mua bảo hiểm FPTCare
    - Thời gian mua được tính từ ngày hiệu lực HDLD đầu tiên
    - Mức mua được tính theo cấp cán bộ được xếp
    - Trường chi phí mua loại hình bảo hiểm này cho CBGV
    - Nếu cấp cán bộ tăng hoặc giảm thì mức mua và quyền lợi cũng tăng hoặc giảm
    - CBGV được cấp thẻ bảo hiểm điện tử và nhận thẻ qua email tự động từ công ty bảo hiểm
    - Được thanh toán chi phí thực tế khi bị tai nạn hoặc gặp vấn đề sức khỏe
    - Được trả một khoản phụ cấp cho những ngày nằm viện
    - Được trả một khoản tiền bồi thường nếu không may bị tử vong do tai nạn
    
    3.20. CÁC KHOẢN HỖ TRỢ:
    - Chi phí hỗ trợ công nghệ: Hỗ trợ sử dụng máy tính xách tay của cá nhân thay cho máy tính của Trường
    - Chi phí hỗ trợ di lại: Cung cấp phương tiện xe ô tô để đưa đón CBGV
    - Chi phí hỗ trợ xăng xe: Quyết định chi trả một phần hoặc toàn bộ chi phí
    - Chi phí hỗ trợ cước phí điện thoại: Quyết định chi trả một phần hoặc toàn bộ
    - Các khoản chi phí khác: Tùy theo yêu cầu của từng vị trí, Trường quyết định chi trả
    
    4. TUÂN THỦ CÁC QUY TẮC VĂN HÓA ỨNG XỬ:
    
    4.1. QUY TẮC ỨNG XỬ TRONG TRƯỜNG:
    - Lịch làm việc được Trường quy định rõ
    - Trang phục và thẻ nhân viên: Theo quy định của từng phòng ban
    - Tiếp khách nơi làm việc: Thân thiện, lịch sự, chuyên nghiệp
    - Giao tiếp nội bộ: Tôn trọng, đúng mực, minh bạch
    
    4.2. QUY TẮC ỨNG XỬ VỚI BÊN NGOÀI:
    - Đại diện tốt cho Trường
    - Bảo vệ danh tiếng và lợi ích của Trường
    - Tôn trọng khách hàng, đối tác
    
    5. QUY ĐỊNH NỘI QUY LÀM VIỆC:
    
    5.1. THỜI GIAN LÀM VIỆC:
    - Được Trường quy định rõ trong hợp đồng lao động
    - CBGV phải tuân thủ giờ làm việc
    
    5.2. TRANG PHỤC VÀ THẺ NHÂN VIÊN:
    - Theo quy định của từng phòng ban
    - Phải mang thẻ nhân viên hàng ngày
    
    5.3. TIẾP KHÁCH NƠI LÀM VIỆC:
    - Thân thiện, lịch sự, chuyên nghiệp
    
    5.4. GIAO TIẾP NỘI BỘ:
    - Tôn trọng, đúng mực, minh bạch
    
    5.5. SỬ DỤNG SỐ VẬT CHẤT:
    - Sử dụng có trách nhiệm
    - Báo cáo khi bị hư hỏng
    
    5.6. CHẤP HÀNH KỶ LUẬT LAO ĐỘNG:
    - CBGV phải chấp hành các quy định về kỷ luật
    - Vi phạm có thể bị xử lý kỷ luật
    """
    
    # Chunk FPD content
    fpd_chunks = chunk_text(fpd_content)
    for i, chunk in enumerate(fpd_chunks):
        if chunk.strip():
            documents.append({
                "id": f"fpd-{doc_id}",
                "content": chunk,
                "metadata": {
                    "source": "FPD Education Employee Handbook 2024",
                    "page": i + 1,
                    "doc_type": "handbook"
                }
            })
            doc_id += 1
    
    # ===== FPT Software Intern Handbook =====
    fpt_intern_content = """
    FPT Software Intern Handbook - Sổ tay Thực tập sinh FPT Software
    
    LỜI CHÀO MỪNG:
    Chào mừng các bạn thực tập sinh đến với FPT Software!
    
    Sổ tay này được biên soạn nhằm cung cấp cho các bạn những thông tin cần thiết và hữu ích nhất
    trong suốt quá trình thực tập tại công ty. Chúng tôi hy vọng rằng, với những kiến thức và kinh nghiệm
    thu được, các bạn sẽ có một kỳ thực tập thành công và đáng nhớ.
    
    GIỚI THIỆU CHUNG VỀ FPT SOFTWARE:
    
    Tên đầy đủ: Công ty TNHH Phần mềm FPT
    Tên tiếng Anh: FPT Software Company Limited
    Ngành nghề: Phát triển phần mềm, gia công phần mềm, dịch vụ CNTT
    Địa chỉ: Đội xác định theo chi nhánh
    
    PHÒNG BAN/DỰ ÁN THỰC TẬP PHỔ BIẾN:
    
    1. PHÁT TRIỂN PHẦN MỀM:
    - Tham gia vào các dự án phát triển ứng dụng web, mobile, desktop
    - Làm việc với các công nghệ: Java, C#, Python, JavaScript, TypeScript, React, Angular
    - Xây dựng tính năng mới, sửa bug, tối ưu hóa code
    
    2. KIỂM THỬ PHẦN MỀM (QA):
    - Thực hiện các hoạt động kiểm thử, đảm bảo chất lượng sản phẩm
    - Viết test case, thực hiện test thủ công, tự động hóa test
    - Báo cáo bug, xác minh fix
    
    3. PHÂN TÍCH NGHIỆP VỤ (BA):
    - Tìm hiểu yêu cầu của khách hàng, phân tích và đưa ra giải pháp
    - Viết tài liệu yêu cầu, kỹ thuật
    - Liên kết giữa khách hàng và team phát triển
    
    4. DATA SCIENCE:
    - Tham gia vào các dự án phân tích dữ liệu, khai phá thông tin
    - Sử dụng Python, R, SQL
    - Xây dựng mô hình dự đoán
    
    5. AI/ML:
    - Nghiên cứu và phát triển các ứng dụng trí tuệ nhân tạo, học máy
    - Sử dụng TensorFlow, PyTorch, scikit-learn
    - Xây dựng mô hình deep learning
    
    HỖ TRỢ VÀ GIẢI ĐÁP THẮC MẮC:
    
    - Mentor: Mỗi thực tập sinh được gán một mentor để hướng dẫn
    - Phòng Nhân sự: Xử lý các vấn đề hành chính, phúc lợi
    - Team Lead: Giám sát công việc hàng ngày
    - Office Manager: Hỗ trợ về cơ sở vật chất, môi trường làm việc
    
    QUY ĐỊNH CHUNG:
    
    1. THỜI GIAN THỰC TẬP:
    - Kỳ thực tập thường từ 3-6 tháng
    - Có thể gia hạn dựa trên kết quả làm việc
    - Tuân thủ lịch làm việc của công ty
    
    2. CHÍNH SÁCH LỰC VÀ LƯƠNG:
    - Thực tập sinh được hỗ trợ sinh hoạt phí hàng tháng
    - Mức hỗ trợ tùy thuộc vào kỳ thực tập và vị trí
    - Được thanh toán đúng hạn hàng tháng
    
    3. NGÀY NGHỈ VÀ PHÉP:
    - Được nghỉ các ngày lễ, Tết theo quy định của Nhà nước
    - Được nghỉ phép hàng năm: 1 ngày/tháng (tối đa 6 ngày/năm)
    - Được nghỉ bảo hiểm xã hội (thai sản, ốm đau) khi đủ điều kiện
    
    4. QUYỀN LỢI VÀ PHÚC LỢI:
    - Được hưởng bảo hiểm thất nghiệp
    - Được hỗ trợ tiền ăn trưa
    - Được sử dụng các tiện ích công ty (phòng tập, thư viện, cafe)
    - Được tham gia các hoạt động team building, sự kiện của công ty
    
    5. QUYỀN ĐƯỢC ĐÀO TẠO:
    - Được đào tạo kỹ năng mềm (communication, teamwork, leadership)
    - Được tham gia các khóa học kỹ thuật
    - Được tham gia các sự kiện chia sẻ kiến thức
    - Được hỗ trợ cấp chứng chỉ chuyên môn
    
    6. KỸ LUẬT VÀ QUY TẮC HÀNH VI:
    - Tuân thủ giờ làm việc
    - Mặc áo công ty hoặc trang phục lịch sự
    - Không được sử dụng điện thoại trong giờ làm việc (trừ khi cần thiết)
    - Không được nói những điều tiêu cực về công ty, khách hàng, đồng nghiệp
    - Vi phạm kỷ luật có thể dẫn đến chấm dứt hợp đồng
    
    7. CHÍNH SÁCH AN TOÀN VỀ SINH:
    - Công ty cung cấp môi trường làm việc an toàn
    - Bảo hiểm tai nạn lao động được Công ty mua
    - Khám sức khỏe định kỳ hàng năm
    
    8. BẢO MẬT VÀ BẢO VỆ DỮ LIỆU:
    - Các thực tập sinh phải tuân thủ quy định bảo mật của công ty
    - Không được tiết lộ thông tin bí mật của công ty hoặc khách hàng
    - Không được sao chép, chia sẻ dữ liệu công việc
    - Vi phạm bảo mật sẽ bị xử lý theo luật pháp
    
    HƯỚNG NGHIỆP VÀ PHÁT TRIỂN:
    
    - Các thực tập sinh có kết quả tốt được ưu tiên tuyển dụng làm nhân viên
    - Được hỗ trợ lập kế hoạch phát triển sự nghiệp
    - Có cơ hội thăng tiến nhanh dựa trên kết quả công việc
    """
    
    # Chunk FPT content
    fpt_chunks = chunk_text(fpt_intern_content)
    for i, chunk in enumerate(fpt_chunks):
        if chunk.strip():
            documents.append({
                "id": f"fpt-intern-{doc_id}",
                "content": chunk,
                "metadata": {
                    "source": "FPT Software Intern Handbook",
                    "page": i + 1,
                    "doc_type": "handbook"
                }
            })
            doc_id += 1
    
    return documents

def add_documents_to_chromadb(documents):
    """Add documents to ChromaDB"""
    print(f"\n[*] Khởi tạo ChromaDB...")
    
    # Initialize embedder and vector store
    embedder = LocalEmbedder()
    vector_store = VectorStoreManager(persist_dir="./chroma_db")
    vector_store.create_collection()
    
    print(f"[*] Thêm {len(documents)} đoạn text mới vào ChromaDB...")
    
    # Add documents
    texts = [doc['content'] for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]
    ids = [doc['id'] for doc in documents]
    
    # Embed texts
    embeddings = embedder.embed(texts, show_progress=True)
    
    vector_store.collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Thêm thành công {len(documents)} documents mới!")
    
    # Check total documents
    count = vector_store.collection.count()
    print(f"📊 Tổng số documents trong ChromaDB: {count}")
    
    return vector_store

if __name__ == "__main__":
    # Create documents from content
    print("=" * 80)
    print("📥 THÊM DOCUMENTS MỚI VÀO CHROMADB")
    print("=" * 80)
    
    documents = create_documents_from_content()
    print(f"\n✅ Tạo {len(documents)} chunks từ 2 documents")
    
    # Add to ChromaDB
    vector_store = add_documents_to_chromadb(documents)
    
    print("\n✅ Hoàn tất! Knowledge base đã được cập nhật")
    print("Bước tiếp theo: Chạy test_qa_optimized.py để kiểm tra cải thiện")
