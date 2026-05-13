"""
Download Phi-3-Mini using _download (most reliable)
"""
import os
from pathlib import Path

# Set token
TOKEN = ""
os.environ["HF_TOKEN"] = TOKEN

from huggingface_hub import _download

print("=" * 70)
print("Downloading Phi-3-Mini GGUF Model")
print("=" * 70)
print()

model_path = Path("./models/phi-3-mini.gguf")

if model_path.exists():
    print(f"✅ Model already exists")
    print(f"   Size: {model_path.stat().st_size / (1024**3):.2f} GB")
    print()
    print("Ready to use! Run:")
    print("  streamlit run streamlit_app.py")
    exit(0)

model_path.parent.mkdir(exist_ok=True)

print("Downloading from TheBloke/Phi-3-mini-4k-instruct-GGUF...")
print()

try:
    # Try different possible filenames
    filenames = [
        "phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "Phi-3-mini-4k-instruct-q4_k_m.gguf",
        "phi-3-mini-4k-instruct-q4_k_m.gguf",
    ]
    
    downloaded = False
    for filename in filenames:
        try:
            print(f"Trying: {filename}...")
            downloaded_path = _download(
                repo_id="TheBloke/Phi-3-mini-4k-instruct-GGUF",
                filename=filename,
                local_dir=str(model_path.parent),
                local_dir_use_symlinks=False,
                token=TOKEN,
            )
            
            # Rename to standard name
            import shutil
            shutil.move(downloaded_path, model_path)
            downloaded = True
            break
        except Exception as e:
            print(f"  ❌ Not found")
            continue
    
    if downloaded:
        print()
        print("=" * 70)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 70)
        print(f"Location: {model_path}")
        print(f"Size: {model_path.stat().st_size / (1024**3):.2f} GB")
        print()
        print("Ready to use! Run:")
        print("  streamlit run streamlit_app.py")
    else:
        print()
        print("❌ Could not find model file")
        print()
        print("Available models might be at different location.")
        print("Try manual download from:")
        print("  https://huggingface.co/TheBloke/Phi-3-mini-4k-instruct-GGUF")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Manual download:")
    print("  1. https://huggingface.co/TheBloke/Phi-3-mini-4k-instruct-GGUF")
    print("  2. Download: phi-3-mini-4k-instruct.Q4_K_M.gguf")
    print("  3. Save to: ./models/phi-3-mini.gguf")
