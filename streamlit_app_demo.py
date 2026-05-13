"""
Demo mode for Streamlit app (without real Phi-3-Mini model)
Shows chatbot working with mock responses
"""

import streamlit as st
import logging
from pathlib import Path
from time import sleep

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="HR Policy Chatbot (Demo)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 HR Policy Chatbot (DEMO MODE)")
st.markdown(
    "⚠️ **Demo Mode** - Shows UI with mock responses (no real model)\n\n"
    "To use the real chatbot with Phi-3-Mini:\n"
    "1. Download model to ./models/phi-3-mini.gguf\n"
    "2. Run: `streamlit run streamlit_app.py`"
)

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Demo Settings")
    demo_speed = st.slider("Response time (seconds)", 1, 10, 3)
    st.info("This is a demo showing the UI layout. Real model responses will be generated with Phi-3-Mini GGUF.")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📝 Your Question")
    question = st.text_area(
        "Ask a question about HR policies:",
        height=150,
        placeholder="E.g., 'Bao nhiêu ngày nghỉ phép mỗi năm?'",
        label_visibility="collapsed",
    )
    
    submit = st.button("🔍 Search & Respond", use_container_width=True, type="primary")

with col2:
    if submit and question:
        # Demo responses
        demo_responses = {
            "phép": {
                "answer": "Theo chính sách công ty, mỗi nhân viên được 15 ngày nghỉ phép có lương mỗi năm. Ngày nghỉ phép được tính theo năm lịch từ tháng 1 đến tháng 12.",
                "confidence": 0.92,
                "pages": [1, 2],
            },
            "lương": {
                "answer": "Lương được trả vào ngày 25 hàng tháng. Nếu ngày 25 rơi vào thứ 7 hoặc chủ nhật, lương sẽ được trả vào ngày làm việc tiếp theo.",
                "confidence": 0.88,
                "pages": [3],
            },
            "hợp đồng": {
                "answer": "Hợp đồng lao động có thời hạn 12 tháng. Thời gian thử việc là 3 tháng. Sau thời gian thử việc, nếu thỏa điều kiện, sẽ ký hợp đồng chính thức.",
                "confidence": 0.85,
                "pages": [1, 5],
            },
            "": {
                "answer": "Vui lòng nhập câu hỏi về chính sách HR.",
                "confidence": 0.5,
                "pages": [],
            }
        }
        
        # Find matching demo response
        response_key = next((key for key in demo_responses if key in question.lower()), "")
        demo_resp = demo_responses.get(response_key, demo_responses[""])
        
        with st.spinner(f"Generating response (demo: {demo_speed}s)..."):
            sleep(demo_speed)
        
        st.markdown("### 💬 Answer (Demo)")
        st.markdown(demo_resp["answer"])
        
        st.markdown("### 📖 Sources")
        for page in demo_resp["pages"]:
            st.info(f"📄 Page {page}: Sample handbook content")
        
        if not demo_resp["pages"]:
            st.info("No specific sources identified")
        
        st.markdown("### 📊 Performance")
        cols = st.columns(4)
        cols[0].metric("Retrieval", "12ms")
        cols[1].metric("Response", f"{demo_speed*1000:.0f}ms")
        cols[2].metric("Total", f"{(demo_speed+0.012)*1000:.0f}ms")
        cols[3].metric("Confidence", f"{demo_resp['confidence']*100:.0f}%")
        
    elif submit:
        st.warning("⚠️ Please enter a question")

st.divider()

with st.expander("ℹ️ About Demo Mode"):
    st.markdown("""
    **Current Status:** Demo UI (mock responses)
    
    **To Enable Real Responses:**
    1. Download Phi-3-Mini GGUF model (~2.3GB)
    2. Save to: `./models/phi-3-mini.gguf`
    3. Run: `streamlit run streamlit_app.py`
    
    **Model Download Instructions:**
    - Visit: https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF
    - Download: Phi-3-mini-4k-instruct-Q4_K_M.gguf
    - Or use: `pip install huggingface-hub` → `python download_final.py`
    """)

st.markdown("---")
st.caption("Demo Mode - Click 'About Demo Mode' above for setup instructions")
