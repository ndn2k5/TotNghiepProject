"""
Simple model downloader using requests
"""
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination, chunk_size=8192):
    """Download file from URL with progress bar"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    if response.status_code != 200:
        print(f"❌ Download failed: {response.status_code}")
        return False
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return True

print("=" * 70)
print("Downloading Phi-3-Mini GGUF Model")
print("=" * 70)
print()

# Try multiple sources
sources = [
    ("https://huggingface.co/TheBloke/Phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.Q4_K_M.gguf", "TheBloke (Recommended)"),
    ("https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/phi-3-mini-4k-instruct-q4_k_m.gguf", "Microsoft (Official)"),
]

model_path = Path("./models/phi-3-mini.gguf")

if model_path.exists():
    print(f"✅ Model already exists!")
    print(f"   Size: {model_path.stat().st_size / (1024**3):.2f} GB")
    exit(0)

success = False
for url, source in sources:
    print(f"Trying source: {source}")
    if download_file(url, model_path):
        success = True
        break
    else:
        model_path.unlink(missing_ok=True)
        print()

if success:
    print()
    print("=" * 70)
    print(f"✅ SUCCESS! Model downloaded")
    print("=" * 70)
    print(f"Location: {model_path}")
    print(f"Size: {model_path.stat().st_size / (1024**3):.2f} GB")
    print()
    print("Next: streamlit run streamlit_app.py")
else:
    print()
    print("=" * 70)
    print("❌ Download failed from all sources")
    print("=" * 70)
    print()
    print("Manual download:")
    print("1. https://huggingface.co/TheBloke/Phi-3-mini-4k-instruct-GGUF")
    print("2. Click: phi-3-mini-4k-instruct.Q4_K_M.gguf")
    print("3. Save to: ./models/phi-3-mini.gguf")
