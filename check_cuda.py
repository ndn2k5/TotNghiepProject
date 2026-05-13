"""Check CUDA/GPU support"""
import torch
import sys

print("[GPU Support Check]")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name()}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  CUDA not available - will use CPU")
print(f"PyTorch: {torch.__version__}")
