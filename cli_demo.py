"""
Interactive CLI chatbot using the local RAG pipeline.

Usage:
    python cli_demo.py <model_path> [pdf_path]

Example:
    python cli_demo.py models/phi3-q4.gguf data/sample_handbook.pdf
"""

import sys
import time
from pathlib import Path


def main():
    # ── Parse arguments ─────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("=" * 60)
        print("  RAG Chatbot — Local GGUF Edition")
        print("=" * 60)
        print()
        print("Usage: python cli_demo.py <model_path> [pdf_path]")
        print()
        print("Examples:")
        print("  python cli_demo.py models/phi3-q4.gguf")
        print("  python cli_demo.py models/phi3-q4.gguf data/sample_handbook.pdf")
        print()

        # List available models
        from src.gguf_models import list_available_models
        models = list_available_models()
        if models:
            print("Available models:")
            for m in models:
                size_mb = m.stat().st_size / (1024 * 1024)
                print(f"  ✓ {m} ({size_mb:.0f} MB)")
        else:
            print("⚠ No GGUF models found in models/ directory.")
            print("  Download one first — see scripts/download_model.py")
        sys.exit(1)

    model_path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_handbook.pdf"

    # ── Initialize pipeline ─────────────────────────────────────────
    print("=" * 60)
    print("  RAG Chatbot — Loading ...")
    print("=" * 60)
    print()

    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(model_path)

    # Ingest PDF if store is empty
    if pipeline.vector_store.count() == 0:
        print(f"📄 Ingesting: {pdf_path}")
        count = pipeline.ingest_pdf(pdf_path)
        print(f"✅ Ingested {count} chunks")
    else:
        print(f"📦 Vector store already has {pipeline.vector_store.count()} documents")

    stats = pipeline.get_stats()
    print(f"🤖 Model: {stats['model']}")
    print(f"📚 Documents: {stats['documents_in_store']}")
    print()

    # ── Interactive chat loop ───────────────────────────────────────
    print("=" * 60)
    print("  Chat với tài liệu nội bộ")
    print("  Gõ 'exit' để thoát, 'stats' để xem thống kê")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("🧑 Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Tạm biệt!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("\n👋 Tạm biệt!")
            break
        if question.lower() == "stats":
            s = pipeline.get_stats()
            print(f"  Model: {s['model']}")
            print(f"  Documents: {s['documents_in_store']}")
            print(f"  Embedding dim: {s['embedding_dimension']}")
            continue

        # Get answer
        result = pipeline.answer(question)

        # Display answer
        print(f"\n🤖 Bot: {result['answer']}")

        # Display sources
        if result["sources"]:
            sources_str = ", ".join(
                f"trang {s['page']} ({s['source']})" for s in result["sources"]
            )
            print(f"📎 Nguồn: {sources_str}")

        # Display timing
        t = result["timing"]
        print(f"⏱  Retrieval: {t['retrieval_seconds']:.2f}s | "
              f"Generation: {t['generation_seconds']:.2f}s | "
              f"Total: {t['total_seconds']:.2f}s")
        print()


if __name__ == "__main__":
    main()
