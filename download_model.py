"""
Download Phi-3-Mini GGUF model from HuggingFace

Usage:
    python download_model.py
"""

import os
from pathlib import Path

def download_model():
    """Download Phi-3-Mini GGUF model"""
    
    # Create models directory
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "phi-3-mini.gguf"
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024**3):.2f} GB")
        return
    
    print("=" * 60)
    print("DOWNLOADING PHI-3-MINI GGUF MODEL")
    print("=" * 60)
    print()
    print("Model: Phi-3-Mini-4K-Instruct (GGUF Q4_K_M Quantized)")
    print("Size: ~2.3 GB")
    print("Location: ./models/phi-3-mini.gguf")
    print()
    print("This will take 5-15 minutes depending on your internet speed...")
    print()
    
    try:
        from huggingface_hub import hf_hub_download
        
        print("[1/2] Installing HuggingFace Hub...")
        
        print("[2/2] Downloading model (this may take a while)...")
        print()
        
        model_file = hf_hub_download(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4_k_m.gguf",
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        
        print()
        print("=" * 60)
        print("✅ MODEL DOWNLOADED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Location: {model_file}")
        print(f"Size: {Path(model_file).stat().st_size / (1024**3):.2f} GB")
        print()
        print("Next steps:")
        print("  1. Run: streamlit run streamlit_app.py")
        print("  2. Open: http://localhost:8501")
        print("  3. Ask HR questions in Vietnamese!")
        
    except ImportError:
        print("❌ huggingface-hub not installed")
        print()
        print("Install it with:")
        print("  pip install huggingface-hub")
        print()
        print("Then run this script again.")
        return
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("Alternative: Download manually from:")
        print("  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
        print()
        print("1. Click: Phi-3-mini-4k-instruct-q4_k_m.gguf")
        print("2. Click: Download")
        print("3. Save to: ./models/phi-3-mini.gguf")
        return

if __name__ == "__main__":
    download_model()
