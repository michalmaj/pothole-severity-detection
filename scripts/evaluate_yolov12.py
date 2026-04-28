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

import torch
import yaml
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


def load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    """Load a YAML configuration file if provided."""
    if config_path is None:
        return {}

    resolved_config_path = config_path.expanduser().resolve()

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return config


def get_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Read a configuration section safely."""
    section = config.get(section_name, {})

    if not isinstance(section, dict):
        raise ValueError(f"Invalid configuration section: {section_name}")

    return section


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


def get_model_path(
    args: argparse.Namespace,
    model_config: dict[str, Any],
) -> Path:
    """Resolve model path from CLI arguments or experiment config."""
    model_value = (
        args.model
        or model_config.get("output_weights")
        or model_config.get("weights")
        or model_config.get("source")
    )

    if model_value is None:
        raise ValueError(
            "Model weights must be provided through --model or config model section."
        )

    model_path = resolve_existing_path(model_value, "Model weights")

    if model_path.suffix != ".pt":
        raise ValueError(
            "Evaluation requires trained model weights with `.pt` extension. "
            f"Received: {model_path}"
        )

    return model_path


def get_data_path(
    args: argparse.Namespace,
    dataset_config: dict[str, Any],
) -> Path:
    """Resolve dataset YAML path from CLI arguments or experiment config."""
    data_value = args.data or dataset_config.get("data_yaml")

    if data_value is None:
        raise ValueError(
            "Dataset YAML must be provided through --data or config dataset section."
        )

    return resolve_existing_path(data_value, "Dataset YAML")


def get_output_settings(
    args: argparse.Namespace,
    experiment_config: dict[str, Any],
    outputs_config: dict[str, Any],
    split: str,
) -> tuple[Path, str]:
    """Resolve evaluation output directory and run name."""
    experiment_name = str(experiment_config.get("name", "yolov12_evaluation"))

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        run_name = args.name or f"{experiment_name}_{split}"
        return output_dir, run_name

    evaluation_dir = outputs_config.get("evaluation_dir")

    if evaluation_dir is not None:
        resolved_evaluation_dir = Path(evaluation_dir).expanduser().resolve()
        return resolved_evaluation_dir.parent, resolved_evaluation_dir.name

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

    config_path = args.config.expanduser().resolve() if args.config else None
    config = load_yaml_config(config_path)

    experiment_config = get_section(config, "experiment")
    dataset_config = get_section(config, "dataset")
    model_config = get_section(config, "model")
    training_config = get_section(config, "training")
    evaluation_config = get_section(config, "evaluation")
    outputs_config = get_section(config, "outputs")

    model_path = get_model_path(args=args, model_config=model_config)
    data_path = get_data_path(args=args, dataset_config=dataset_config)

    split = args.split or str(evaluation_config.get("split", "test"))

    device = (
        args.device
        or evaluation_config.get("device")
        or training_config.get("device")
        or "cpu"
    )
    device = select_device() if device == "auto" else str(device)

    image_size = int(
        args.imgsz
        or evaluation_config.get("image_size")
        or training_config.get("image_size")
        or 416
    )

    batch_size = int(
        args.batch
        or evaluation_config.get("batch_size")
        or training_config.get("batch_size")
        or 2
    )

    output_dir, run_name = get_output_settings(
        args=args,
        experiment_config=experiment_config,
        outputs_config=outputs_config,
        split=split,
    )

    output_path = output_dir / run_name / "metrics.json"

    print("YOLOv12 evaluation configuration")
    print()
    print(f"Config: {config_path}")
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
        "config_path": str(config_path) if config_path else None,
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
