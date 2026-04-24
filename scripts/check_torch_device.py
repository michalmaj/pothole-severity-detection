"""Check available PyTorch compute backends.

This script is intentionally small and explicit.
It helps verify whether the local machine can use CUDA, Apple MPS, or CPU.
"""

from __future__ import annotations

import platform

import torch


def select_device() -> str:
    """Select the best available PyTorch device.

    Priority:
    1. CUDA for NVIDIA GPUs.
    2. MPS for Apple Silicon GPUs through Metal.
    3. CPU as a universal fallback.
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main() -> None:
    """Print PyTorch backend information."""
    device = select_device()

    print(f"Python platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if device == "cuda":
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    if device == "mps":
        x = torch.ones((3, 3), device="mps")
        y = x @ x
        print("MPS test tensor:")
        print(y.cpu())


if __name__ == "__main__":
    main()
