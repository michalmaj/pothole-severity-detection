"""Evaluate a YOLOv12 model on a selected dataset split.

The script runs Ultralytics validation and stores selected metrics in a JSON
file. It is intended for local and remote experiment tracking.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


def select_device() -> str:
    """Select the best available device for evaluation."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv12 model on a dataset split."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to YOLO model weights.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/pothole_detection_v2/data.yaml"),
        help="Path to YOLO dataset configuration file.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used for evaluation.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolov12_evaluation",
        help="Experiment name used for output directory.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Input image size used during evaluation.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Evaluation device: auto, cpu, mps, cuda, or CUDA device index.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory where evaluation metrics will be saved.",
    )

    return parser.parse_args()


def get_metric(source: Any, name: str) -> float | None:
    """Safely read a numeric metric from an object."""
    value = getattr(source, name, None)

    if value is None:
        return None

    return float(value)


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Save metrics dictionary as a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    """Run YOLOv12 evaluation and save metrics."""
    args = parse_args()

    model_path = args.model.resolve()
    data_path = args.data.resolve()
    device = select_device() if args.device == "auto" else args.device

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    print(f"Model: {model_path}")
    print(f"Dataset config: {data_path}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")

    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        project=str(args.output_dir),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    box_metrics = results.box

    metrics = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "data_path": str(data_path),
        "split": args.split,
        "device": device,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "metrics": {
            "precision": get_metric(box_metrics, "mp"),
            "recall": get_metric(box_metrics, "mr"),
            "map50": get_metric(box_metrics, "map50"),
            "map75": get_metric(box_metrics, "map75"),
            "map50_95": get_metric(box_metrics, "map"),
        },
    }

    output_path = args.output_dir / args.name / "metrics.json"
    save_metrics(metrics=metrics, output_path=output_path)

    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
