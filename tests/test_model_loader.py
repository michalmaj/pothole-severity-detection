"""Tests for YOLO model loading utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from pothole_severity_detection.inference.model_loader import load_model


def test_load_model_raises_error_for_missing_weights(tmp_path: Path) -> None:
    missing_model_path = tmp_path / "missing_model.pt"

    with pytest.raises(FileNotFoundError, match="Model weights were not found"):
        load_model(missing_model_path)
