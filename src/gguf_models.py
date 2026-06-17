# -*- coding: utf-8 -*-
"""
Local GGUF model loading and inference using llama-cpp-python.
Runs entirely on CPU — no API calls needed.
"""

import torch  # Required to initialize CUDA environment for llama_cpp on Windows
from llama_cpp import Llama
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalGGUFModel:
    """Wraps llama-cpp-python for local GGUF model inference."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        """
        Load a GGUF model from disk.

        Args:
            model_path: Path to the .gguf model file
            n_ctx: Context window size (tokens). 2048 is safe for most models.
            n_threads: Number of CPU threads. Defaults to (cpu_count // 2).
            n_gpu_layers: Number of layers to offload to GPU. -1 = all layers. 0 = CPU only.
            verbose: If True, print llama.cpp internal logs.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Download a GGUF model and place it in the models/ directory.\n"
                f"Recommended: Phi-3-mini-4k-instruct-q4_k_m.gguf (~2.3 GB)"
            )

        if n_threads is None:
            n_threads = max(1, (os.cpu_count() or 4) // 2)

        device_info = "GPU (CUDA)" if n_gpu_layers > 0 else "CPU"
        logger.info(f"Loading GGUF model: {self.model_path.name} "
                     f"(ctx={n_ctx}, threads={n_threads}, device={device_info}) ...")

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self.n_ctx = n_ctx
        logger.info(f"✅ Model loaded successfully: {self.model_path.name} (GPU acceleration enabled)")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[list] = None,
    ) -> str:
        """
        Generate text completion from a prompt.

        Args:
            prompt: The input prompt (including any system/context text)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling threshold
            stop: List of stop sequences

        Returns:
            Generated text string
        """
        if stop is None:
            stop = ["<|end|>", "<|user|>", "<|system|>", "User:", "Human:"]

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False,
        )

        text = output["choices"][0]["text"].strip()
        logger.debug(f"Generated {len(text)} chars from prompt of {len(prompt)} chars")
        return text

    def generate_with_metadata(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> Dict[str, Any]:
        """
        Generate text with advanced sampling parameters and return with usage metadata.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0=greedy, higher=more creative)
            top_p: Nucleus sampling threshold (0.9 = use top 90% probability mass)
            top_k: Keep only top k tokens by probability

        Returns:
            Dict with keys: text, prompt_tokens, completion_tokens, total_tokens
        """
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=["<|end|>", "<|user|>", "<|system|>", "User:", "Human:"],
            echo=False,
        )

        usage = output.get("usage", {})
        return {
            "text": output["choices"][0]["text"].strip(),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    @property
    def model_name(self) -> str:
        """Return the model filename without extension."""
        return self.model_path.stem


def list_available_models(models_dir: str = "models") -> list:
    """
    List all .gguf files in the models directory.

    Args:
        models_dir: Path to directory containing GGUF files

    Returns:
        List of model file paths
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    return sorted(models_path.glob("*.gguf"))


if __name__ == "__main__":
    # Quick check: list available models
    models = list_available_models()
    if models:
        print(f"Found {len(models)} GGUF model(s):")
        for m in models:
            size_mb = m.stat().st_size / (1024 * 1024)
            print(f"  - {m.name} ({size_mb:.0f} MB)")
    else:
        print("No GGUF models found in models/ directory.")
        print("Download one with:")
        print("  huggingface-cli download bartowski/Phi-3.1-mini-4k-instruct-GGUF "
              "--include 'Phi-3.1-mini-4k-instruct-Q4_K_M.gguf' --local-dir models/")
