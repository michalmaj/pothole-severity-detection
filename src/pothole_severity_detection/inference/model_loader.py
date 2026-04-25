"""YOLO model loading utilities."""

from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("runs/local_smoke/yolov12_smoke/weights/best.pt")


def load_model(model_path: str | Path | None = None) -> YOLO:
    """Load a YOLO model from a local weights file.

    Args:
        model_path: Optional path to model weights. If not provided, the
            function checks the MODEL_PATH environment variable and then falls
            back to the default local smoke training output.

    Returns:
        Loaded YOLO model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    resolved_model_path = Path(
        model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    ).expanduser()

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            "Model weights were not found. "
            f"Expected path: {resolved_model_path}. "
            "Run local smoke training first or set MODEL_PATH."
        )

    return YOLO(str(resolved_model_path))
