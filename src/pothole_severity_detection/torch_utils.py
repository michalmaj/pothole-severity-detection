"""Torch device selection shared by the training and evaluation scripts."""

from __future__ import annotations

import torch


def select_device() -> str:
    """Return the best available torch device: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"
