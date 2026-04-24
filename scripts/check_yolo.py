"""Check whether YOLOv12 can be initialized in the local environment.

This script does not train the model.
It only verifies that the YOLOv12 package is installed correctly and that
the selected PyTorch device can be used for a lightweight sanity check.
"""

from __future__ import annotations

import torch
from ultralytics import YOLO


def select_device() -> str:
    """Select the best available device for local model checks."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main() -> None:
    """Initialize YOLOv12 and print basic environment information."""
    device = select_device()

    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")

    model = YOLO("yolov12s.yaml")

    print("YOLOv12 model initialized successfully.")
    print(f"Model task: {model.task}")


if __name__ == "__main__":
    main()
