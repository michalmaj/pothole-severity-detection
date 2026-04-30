"""YOLO model loading utilities."""

from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO

DEFAULT_MODEL_PATH = Path(
    "weights/local/yolov12n_cpu_100e_plus_60e_416_b2_gentle_aug_best.pt"
)


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve model path from an argument, environment variable, or default path."""
    return Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()


def load_model(model_path: str | Path | None = None) -> YOLO:
    """Load a YOLO model from a local weights file.

    Args:
        model_path: Optional path to model weights. If not provided, the
            function checks the MODEL_PATH environment variable and then falls
            back to the best documented local baseline.

    Returns:
        Loaded YOLO model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    resolved_model_path = resolve_model_path(model_path)

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            "Model weights were not found. "
            f"Expected path: {resolved_model_path}. "
            "Run training first or set MODEL_PATH."
        )

    return YOLO(str(resolved_model_path))
