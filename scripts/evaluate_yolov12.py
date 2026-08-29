"""Evaluate a YOLOv12 model.

The script supports two workflows:

1. Manual CLI arguments.
2. Config-driven evaluation using a YAML experiment configuration.

Evaluation metrics are saved to a local JSON file.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from pothole_severity_detection.experiment_config import (
    ExperimentConfig,
    load_experiment_config,
)
from pothole_severity_detection.torch_utils import select_device


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv12 model.")

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to an experiment configuration file.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to YOLO model weights. Overrides config value.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to YOLO dataset configuration file. Overrides config value.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Dataset split used for evaluation. Overrides config value.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name used for output directory. Overrides config value.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size used during evaluation. Overrides config value.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Evaluation batch size. Overrides config value.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Evaluation device: auto, cpu, mps, cuda, or CUDA device index.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where evaluation metrics will be saved.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate settings without running evaluation.",
    )

    return parser.parse_args()


def get_metric(source: Any, name: str) -> float | None:
    """Safely read a numeric metric from an object."""
    value = getattr(source, name, None)

    if value is None:
        return None

    return float(value)


def resolve_existing_path(path_value: str | Path, label: str) -> Path:
    """Resolve and validate an existing local path."""
    path = Path(path_value).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    return path


def get_model_path(args: argparse.Namespace, config: ExperimentConfig | None) -> Path:
    """Resolve model weights from CLI arguments or the experiment config."""
    model_value = args.model
    if model_value is None and config is not None:
        model_value = config.model.output_weights or config.model.source

    if model_value is None:
        raise ValueError(
            "Model weights must be provided through --model or a config model section."
        )

    model_path = resolve_existing_path(model_value, "Model weights")

    if model_path.suffix != ".pt":
        raise ValueError(
            "Evaluation requires trained model weights with `.pt` extension. "
            f"Received: {model_path}"
        )

    return model_path


def get_data_path(args: argparse.Namespace, config: ExperimentConfig | None) -> Path:
    """Resolve the dataset YAML from CLI arguments or the experiment config."""
    data_value = args.data
    if data_value is None and config is not None:
        data_value = config.dataset.data_yaml

    if data_value is None:
        raise ValueError(
            "Dataset YAML must be provided through --data or a config dataset section."
        )

    return resolve_existing_path(data_value, "Dataset YAML")


def get_output_settings(
    args: argparse.Namespace,
    config: ExperimentConfig | None,
    split: str,
) -> tuple[Path, str]:
    """Resolve the evaluation output directory and run name."""
    experiment_name = "yolov12_evaluation"
    if config is not None:
        experiment_name = config.experiment.name

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        run_name = args.name or f"{experiment_name}_{split}"
        return output_dir, run_name

    evaluation_dir = config.outputs.evaluation_dir if config is not None else None
    if evaluation_dir is not None:
        resolved = Path(evaluation_dir).expanduser().resolve()
        return resolved.parent, resolved.name

    output_dir = Path("outputs/evaluation").resolve()
    run_name = args.name or f"{experiment_name}_{split}"
    return output_dir, run_name


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Save metrics dictionary as a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    """Run YOLOv12 evaluation and save metrics."""
    args = parse_args()

    config = load_experiment_config(args.config) if args.config is not None else None

    model_path = get_model_path(args=args, config=config)
    data_path = get_data_path(args=args, config=config)

    split = args.split
    if split is None and config is not None and config.evaluation is not None:
        split = config.evaluation.split
    if split is None:
        split = "test"

    device = args.device
    if device is None and config is not None and config.evaluation is not None:
        device = config.evaluation.device
    if device is None and config is not None:
        device = config.training.device
    if device is None:
        device = "cpu"
    device = select_device() if device == "auto" else str(device)

    image_size = args.imgsz
    if image_size is None and config is not None and config.evaluation is not None:
        image_size = config.evaluation.image_size
    if image_size is None and config is not None:
        image_size = config.training.image_size
    if image_size is None:
        image_size = 416
    image_size = int(image_size)

    batch_size = args.batch
    if batch_size is None and config is not None and config.evaluation is not None:
        batch_size = config.evaluation.batch_size
    if batch_size is None and config is not None:
        batch_size = config.training.batch_size
    if batch_size is None:
        batch_size = 2
    batch_size = int(batch_size)

    output_dir, run_name = get_output_settings(args=args, config=config, split=split)

    output_path = output_dir / run_name / "metrics.json"

    print("YOLOv12 evaluation configuration")
    print()
    print(f"Config: {config.path if config is not None else None}")
    print(f"Model: {model_path}")
    print(f"Dataset YAML: {data_path}")
    print(f"Split: {split}")
    print(f"Device: {device}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print(f"Run name: {run_name}")
    print(f"Metrics path: {output_path}")

    if args.dry_run:
        print()
        print("Dry run completed successfully. Evaluation was not started.")
        return

    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_path),
        split=split,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=0,
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    box_metrics = results.box

    metrics = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config.path) if config is not None else None,
        "model_path": str(model_path),
        "data_path": str(data_path),
        "split": split,
        "device": device,
        "image_size": image_size,
        "batch_size": batch_size,
        "metrics": {
            "precision": get_metric(box_metrics, "mp"),
            "recall": get_metric(box_metrics, "mr"),
            "map50": get_metric(box_metrics, "map50"),
            "map75": get_metric(box_metrics, "map75"),
            "map50_95": get_metric(box_metrics, "map"),
        },
    }

    save_metrics(metrics=metrics, output_path=output_path)

    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
