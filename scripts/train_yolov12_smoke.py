"""Run a minimal YOLOv12 training smoke test.

This script is intended for local development.
It verifies that the training pipeline works end to end:

dataset -> YOLOv12 model -> selected device -> training output.

It is not meant to produce a high-quality model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def select_device() -> str:
    """Select the best available training device."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the smoke training script."""
    parser = argparse.ArgumentParser(
        description="Run a minimal YOLOv12 training smoke test."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/pothole_detection_v2/data.yaml"),
        help="Path to the YOLO dataset configuration file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov12n.yaml",
        help="YOLOv12 model configuration or checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device: auto, cpu, mps, cuda, or CUDA device index.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the YOLOv12 smoke training."""
    args = parse_args()

    data_path = args.data.resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}. "
            "Place the dataset under data/pothole_detection_v2/ first."
        )

    device = select_device() if args.device == "auto" else args.device

    print(f"Dataset config: {data_path}")
    print(f"Model: {args.model}")
    print(f"Selected device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        amp=False,
        project="runs/local_smoke",
        name="yolov12_smoke",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
