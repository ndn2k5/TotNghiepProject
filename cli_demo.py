"""
Interactive CLI chatbot using the local RAG pipeline.

Usage:
    python cli_demo.py <model_path> [pdf_path]

Example:
    python cli_demo.py models/phi3-q4.gguf data/sample_handbook.pdf
"""

import sys
import os
import time
from pathlib import Path

# Force CPU for embeddings — prevents CUDA context conflict between
# sentence-transformers and llama_cpp on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

    pipeline = RAGPipeline(model_path, n_gpu_layers=0)  # CPU — avoids CUDA conflict with embeddings
    # Now that torch is CPU-only, you can safely try to offload LLM layers to GPU 
    # if your llama-cpp-python was compiled with CUDA support.
    pipeline = RAGPipeline(model_path, n_gpu_layers=-1) 

    # Ingest PDF if store is empty
    if pipeline.vector_store.count() == 0:
        print(f"Ingesting: {pdf_path}")
        count = pipeline.ingest_pdf(pdf_path)
        print(f"Successfully ingested {count} chunks")
    else:
        print(f"Vector store already has {pipeline.vector_store.count()} documents")

    stats = pipeline.get_stats()
    print(f"Model: {stats['model']}")
    print(f"Documents: {stats['documents_in_store']}")
    print()

    # ── Interactive chat loop ───────────────────────────────────────
    print("=" * 60)
    print("  Chat with internal documents")
    print("  Type 'exit' to quit, 'stats' to see statistics")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("\nGoodbye!")
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
        print(f"\nBot: {result['answer']}")

        # Display sources
        if result["sources"]:
            sources_str = ", ".join(
                f"page {s['page']} ({s['source']})" for s in result["sources"]
            )
            print(f"Source: {sources_str}")

        # Display timing
        t = result["timing"]
        print(f"Time - Retrieval: {t['retrieval_seconds']:.2f}s | "
              f"Generation: {t['generation_seconds']:.2f}s | "
              f"Total: {t['total_seconds']:.2f}s")
        print()


if __name__ == "__main__":
    main()
