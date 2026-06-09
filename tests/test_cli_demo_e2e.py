import pytest
import subprocess
import sys
import os
from pathlib import Path

# Use a non-UTF-8 encoding if Windows terminal uses something else, 
# but usually 'cp1252' or 'utf-8' with 'replace' is safer for tests.

def test_cli_help():
    """Verify CLI shows help when no arguments are provided."""
    result = subprocess.run(
        [sys.executable, "cli_demo.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    assert result.returncode == 1
    assert "Usage: python cli_demo.py" in result.stdout

def test_cli_invalid_model():
    """Verify CLI handles non-existent model path."""
    result = subprocess.run(
        [sys.executable, "cli_demo.py", "non_existent_model.gguf"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    assert result.returncode != 0
    # Search in both stdout and stderr since output might go to either
    combined_output = (result.stdout + result.stderr).lower()
    assert "non_existent_model.gguf" in combined_output or "not found" in combined_output or "filenotfounderror" in combined_output

@pytest.mark.skipif(not Path("models/phi-3-mini.gguf").exists(), reason="Real model required for E2E check")
def test_cli_interactive_smoke():
    """Smoke test: ask a question and exit."""
    # This runs a real model, might be slow
    process = subprocess.Popen(
        [sys.executable, "cli_demo.py", "models/phi-3-mini.gguf"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )
    
    try:
        # Ask a simple question and then exit
        stdout, stderr = process.communicate(input="Chào bạn\nexit\n", timeout=60)
        combined = (stdout + stderr).lower()
        # Look for the new simplified labels
        assert "bot:" in combined or "answer:" in combined
        assert "tạm biệt" in combined or "bye" in combined
    except subprocess.TimeoutExpired:
        process.kill()
        pytest.fail("CLI timed out")
