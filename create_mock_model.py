"""
Create a mock GGUF model for testing UI (not for production)
This is just a placeholder file for UI testing
"""
from pathlib import Path

print("Creating mock GGUF model for UI testing...")
print()

models_dir = Path("./models")
models_dir.mkdir(exist_ok=True)

mock_model = models_dir / "phi-3-mini.gguf"

# Create a dummy file (real model would be 2.3GB)
# For now, just create a small placeholder
with open(mock_model, 'wb') as f:
    # Write minimal GGUF header (not a real model, just for file existence)
    f.write(b"MOCK_GGUF_MODEL_PLACEHOLDER")

print(f"✅ Mock model created: {mock_model}")
print(f"   Size: {mock_model.stat().st_size} bytes (mock only)")
print()
print("⚠️  This is a test placeholder - not a real model!")
print()
print("To get the real model:")
print("  1. Visit: https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF")
print("  2. Download: Phi-3-mini-4k-instruct-Q4_K_M.gguf")
print("  3. Replace ./models/phi-3-mini.gguf with the real model")
print()
print("Then run:")
print("  streamlit run streamlit_app.py")
