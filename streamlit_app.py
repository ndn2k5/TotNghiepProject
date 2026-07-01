"""
Streamlit Web UI for Vietnamese HR Policy Chatbot

Main entry point: streamlit run streamlit_app.py

Features:
- Question input (Vietnamese support)
- Answer generation with sources
- Performance metrics
- Session state management
"""

import streamlit as st
import logging
import time
import os
from pathlib import Path

# Force CPU for embeddings — prevents CUDA context conflict between
# sentence-transformers and llama_cpp on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.embeddings import LocalEmbedder, VectorStoreManager   # torch/onnxruntime first
from src.retriever import Retriever
from src.question_normalizer import QuestionNormalizer
from src.responder import ResponseGenerator, format_response_for_display  # llama_cpp last

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize session state ────────────────────────────────────────

@st.cache_resource
def initialize_components():
    """Initialize and cache all NLP components."""
    try:
        with st.spinner("Initializing components..."):
            # Check if Phi-3-Mini model exists
            model_path = Path("./models/phi-3-mini.gguf")
            if not model_path.exists():
                st.error(
                    f"❌ Model not found: {model_path}\n\n"
                    "Please download Phi-3-Mini GGUF model and save to ./models/phi-3-mini.gguf\n\n"
                    "Download: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
                )
                st.stop()

            # Initialize components
            normalizer = QuestionNormalizer(use_llm=False)
            embedder = LocalEmbedder()
            vector_store = VectorStoreManager()
            vector_store.create_collection()
            retriever = Retriever(
                vector_store=vector_store,
                embedder=embedder,
                use_reranking=False,
            )
            # Force CPU-only mode (gpu_layers = 0) to prevent CUDA context crashes
            # in Streamlit's multi-threaded environment on Windows
            gpu_layers = 0
            if gpu_layers == 0:
                st.warning(f"⚠️ Running on CPU — expect ~20-30s/answer. Download Q4 GGUF for GPU speed.")
            responder = ResponseGenerator(
                model_path=str(model_path),
                language="vi",
                max_tokens=256,
                temperature=0.3,
                n_gpu_layers=gpu_layers,
            )

            return {
                "normalizer": normalizer,
                "embedder": embedder,
                "vector_store": vector_store,
                "retriever": retriever,
                "responder": responder,
            }

    except Exception as e:
        st.error(f"❌ Failed to initialize components: {e}")
        st.stop()


def main():
    """Main Streamlit app."""

    # Header
    st.title("📚 HR Policy Chatbot")
    st.markdown(
        "Ask questions about company HR policies in Vietnamese. "
        "Powered by local AI (no cloud)."
    )

    # Initialize components
    components = initialize_components()

    # ── Sidebar: Settings & Info ────────────────────────────────────

    with st.sidebar:
        st.header("⚙️ Settings")

        st.markdown("### Model Configuration")
        top_k = st.slider(
            "Number of document chunks to retrieve",
            min_value=1,
            max_value=5,
            value=3,
            help="More chunks = more context but longer retrieval time",
        )

        temperature = st.slider(
            "Response temperature (creativity)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative",
        )

        st.markdown("### Information")
        st.info(
            "**Capabilities:**\n"
            "- Retrieve HR policies from handbook\n"
            "- Generate Vietnamese answers\n"
            "- Show document sources\n"
            "- Display performance metrics\n\n"
            "**Limitations:**\n"
            "- Only answers based on provided handbook\n"
            "- ~20-30s per response (CPU-only)\n"
            "- Limited to HR domain"
        )

        st.markdown("### About")
        st.caption("Version: 1.0 (Phase 3)\nCreated: May 2026")

    # ── Main Content ────────────────────────────────────────────────

    # Create columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📝 Your Question")
        question = st.text_area(
            "Ask a question about HR policies (in Vietnamese):",
            height=150,
            placeholder="E.g., 'Bao nhiêu ngày nghỉ phép mỗi năm?'",
            label_visibility="collapsed",
        )

        submit_button = st.button(
            "🔍 Search & Respond",
            use_container_width=True,
            type="primary",
        )

    with col2:
        if submit_button and question:
            with st.spinner("Processing your question..."):
                try:
                    # Normalize question
                    start_time = time.time()
                    normalized_q = components["normalizer"].normalize(question)
                    norm_time = (time.time() - start_time) * 1000

                    # Retrieve chunks
                    start_time = time.time()
                    retrieved, retrieval_time = components["retriever"].retrieve(
                        normalized_q,
                        top_k=top_k,
                        benchmark=True,
                    )
                    retrieval_ms = retrieval_time * 1000

                    if not retrieved:
                        st.warning(
                            "⚠️ No relevant chunks found in the handbook. "
                            "Try a different question."
                        )
                    else:
                        # Generate response
                        start_time = time.time()
                        response = components["responder"].generate(
                            normalized_q,
                            retrieved,
                            benchmark=True,
                        )

                        # Display answer
                        st.markdown("### 💬 Answer")
                        st.markdown(response.answer)

                        # Display sources
                        st.markdown("### 📖 Sources")
                        if response.sources:
                            for i, source in enumerate(response.sources, 1):
                                with st.expander(
                                    f"Source {i} (Page {source.get('page', '?')})"
                                ):
                                    st.text(source.get("text", "No text"))
                        else:
                            st.info("No specific sources identified")

                        # Display metrics
                        st.markdown("### 📊 Performance Metrics")
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Normalization", f"{norm_time:.1f}ms")
                        with metric_cols[1]:
                            st.metric("Retrieval", f"{retrieval_ms:.1f}ms")
                        with metric_cols[2]:
                            st.metric("Response", f"{response.latency_ms:.1f}ms")
                        with metric_cols[3]:
                            st.metric("Confidence", f"{response.confidence*100:.0f}%")

                        # Display detailed results
                        with st.expander("📋 Detailed Results"):
                            st.markdown("**Normalized Question:**")
                            st.code(normalized_q)

                            st.markdown("**Retrieved Chunks:**")
                            for i, chunk in enumerate(retrieved, 1):
                                st.markdown(
                                    f"**Chunk {i}** (Page {chunk.metadata.get('page_num', '?')}, "
                                    f"Distance: {chunk.distance:.3f})"
                                )
                                st.text(chunk.text[:200] + "...")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    logger.exception(e)

        elif submit_button:
            st.warning("⚠️ Please enter a question first")

    # ── Footer ──────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown(
        "**Note:** This chatbot retrieves information only from the provided HR handbook. "
        "It cannot provide personalized legal advice. For specific issues, contact HR."
    )


if __name__ == "__main__":
    main()
