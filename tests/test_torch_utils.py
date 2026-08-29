"""Tests for torch device selection."""

from __future__ import annotations

from pothole_severity_detection import torch_utils


def test_select_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch_utils.torch.backends.mps, "is_available", lambda: True)
    assert torch_utils.select_device() == "cuda"


def test_select_device_falls_back_to_mps(monkeypatch) -> None:
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch_utils.torch.backends.mps, "is_available", lambda: True)
    assert torch_utils.select_device() == "mps"


def test_select_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch_utils.torch.backends.mps, "is_available", lambda: False)
    assert torch_utils.select_device() == "cpu"
