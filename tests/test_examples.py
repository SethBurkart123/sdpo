"""
Smoke tests for example scripts.

Each test runs the example's main() as a subprocess to ensure import ordering
is correct (Unsloth requires patching before SDPOTrainer import). Tests are
marked with @pytest.mark.gpu and require a CUDA device.

Usage:
    pytest tests/test_examples.py -m gpu -v
"""

import subprocess
import sys

import pytest

PYTHON = sys.executable


@pytest.mark.gpu
def test_basic_sdpo():
    result = subprocess.run(
        [PYTHON, "examples/basic_sdpo.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"basic_sdpo.py failed:\n{result.stderr[-2000:]}"


@pytest.mark.gpu
def test_sdpo_lora_ema():
    result = subprocess.run(
        [PYTHON, "examples/sdpo_lora_ema.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"sdpo_lora_ema.py failed:\n{result.stderr[-2000:]}"


@pytest.mark.gpu
def test_sdpo_rich_feedback():
    result = subprocess.run(
        [PYTHON, "examples/sdpo_rich_feedback.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"sdpo_rich_feedback.py failed:\n{result.stderr[-2000:]}"


@pytest.mark.gpu
def test_sdpo_with_unsloth():
    result = subprocess.run(
        [PYTHON, "examples/sdpo_with_unsloth.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"sdpo_with_unsloth.py failed:\n{result.stderr[-2000:]}"
