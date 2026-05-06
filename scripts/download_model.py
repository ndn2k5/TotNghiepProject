"""
Download a GGUF model from Hugging Face for local inference.

Usage:
    python scripts/download_model.py [model_choice]

Model choices:
    phi3   — Phi-3.1-mini-4k-instruct Q4_K_M (~2.3 GB) [RECOMMENDED]
    qwen   — Qwen2.5-1.5B-Instruct Q4_K_M (~1.0 GB)
    tiny   — TinyLlama-1.1B Q4_K_M (~0.6 GB) [for testing]
"""

import sys
import subprocess
from pathlib import Path

MODELS = {
    "phi3": {
        "repo": "bartowski/Phi-3.1-mini-4k-instruct-GGUF",
        "file": "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf",
        "size": "~2.3 GB",
        "description": "Best quality for RAG, recommended for production",
    },
    "qwen": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size": "~1.0 GB",
        "description": "Smaller, faster, good for preprocessing",
    },
    "tiny": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size": "~0.6 GB",
        "description": "Smallest, for quick testing only",
    },
}


def download_model(choice: str = "phi3"):
    """Download a GGUF model using huggingface-cli."""

    if choice not in MODELS:
        print(f"Unknown model: {choice}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model = MODELS[choice]
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    target = models_dir / model["file"]
    if target.exists():
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists: {target} ({size_mb:.0f} MB)")
        return str(target)

    print(f"📥 Downloading: {model['file']}")
    print(f"   Repo: {model['repo']}")
    print(f"   Size: {model['size']}")
    print(f"   Description: {model['description']}")
    print()

    # Method 1: Try huggingface-cli (preferred)
    try:
        subprocess.run(
            [
                sys.executable, "-m", "huggingface_hub", "download",
                model["repo"],
                model["file"],
                "--local-dir", str(models_dir),
                "--local-dir-use-symlinks", "False",
            ],
            check=True,
        )
        if target.exists():
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"\n✅ Downloaded: {target} ({size_mb:.0f} MB)")
            return str(target)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("huggingface-cli failed, trying direct download...")

    # Method 2: Direct download via Python
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=model["repo"],
            filename=model["file"],
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        print(f"\n✅ Downloaded: {path}")
        return path
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"\nManual download:")
        print(f"  1. Go to: https://huggingface.co/{model['repo']}")
        print(f"  2. Download: {model['file']}")
        print(f"  3. Place it in: {models_dir.absolute()}/")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  GGUF Model Downloader")
    print("=" * 60)
    print()

    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        print("Available models:")
        for key, info in MODELS.items():
            tag = " [RECOMMENDED]" if key == "phi3" else ""
            print(f"  {key:6s} — {info['file']}")
            print(f"          {info['size']} | {info['description']}{tag}")
        print()
        choice = input("Choose model (phi3/qwen/tiny) [phi3]: ").strip().lower()
        if not choice:
            choice = "phi3"

    download_model(choice)


if __name__ == "__main__":
    main()
