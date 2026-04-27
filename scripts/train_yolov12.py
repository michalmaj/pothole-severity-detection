"""Train YOLOv12 from an experiment configuration file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


def select_device() -> str:
    """Select the best available training device."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv12 from a YAML experiment configuration file."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment configuration file.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the configuration without starting training.",
    )

    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return config


def require_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a required configuration section."""
    section = config.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid configuration section: {section_name}")

    return section


def resolve_dataset_path(data_yaml: str) -> Path:
    """Resolve and validate dataset YAML path."""
    data_path = Path(data_yaml).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    return data_path


def resolve_model_source(model_source: str) -> str:
    """Resolve model source.

    YOLO model sources can be either:
    - built-in config names, for example `yolov12n.yaml`;
    - local weights paths, for example `weights/local/model.pt`.
    """
    source_path = Path(model_source).expanduser()

    if source_path.suffix == ".pt":
        resolved_path = source_path.resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"Model weights not found: {resolved_path}")

        return str(resolved_path)

    return model_source


def get_training_epochs(training_config: dict[str, Any]) -> int:
    """Get number of epochs from config.

    Standard configs use `epochs`. Fine-tuning configs may use
    `additional_epochs`, because the model continues from previous weights.
    """
    epochs = training_config.get("epochs", training_config.get("additional_epochs"))

    if epochs is None:
        raise ValueError("Training config must define `epochs` or `additional_epochs`.")

    return int(epochs)


def main() -> None:
    """Train YOLOv12 using settings from a YAML config file."""
    args = parse_args()
    config_path = args.config.resolve()

    config = load_yaml_config(config_path)

    experiment_config = require_section(config, "experiment")
    dataset_config = require_section(config, "dataset")
    model_config = require_section(config, "model")
    training_config = require_section(config, "training")
    outputs_config = config.get("outputs", {})

    experiment_name = str(experiment_config.get("name", config_path.stem))
    data_path = resolve_dataset_path(str(dataset_config["data_yaml"]))
    model_source = resolve_model_source(str(model_config["source"]))

    device = str(training_config.get("device", "auto"))
    if device == "auto":
        device = select_device()

    epochs = get_training_epochs(training_config)
    image_size = int(training_config.get("image_size", 416))
    batch_size = int(training_config.get("batch_size", 2))
    workers = int(training_config.get("workers", 0))
    amp = bool(training_config.get("amp", False))

    training_project = str(outputs_config.get("training_project", "runs/train"))
    training_name = str(outputs_config.get("training_name", experiment_name))
    exist_ok = bool(outputs_config.get("exist_ok", True))

    print("YOLOv12 training configuration")
    print()
    print(f"Config: {config_path}")
    print(f"Experiment: {experiment_name}")
    print(f"Dataset YAML: {data_path}")
    print(f"Model source: {model_source}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")
    print(f"AMP: {amp}")
    print(f"Output project: {training_project}")
    print(f"Output name: {training_name}")

    if args.dry_run:
        print()
        print("Dry run completed successfully. Training was not started.")
        return

    model = YOLO(model_source)

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=workers,
        amp=amp,
        project=training_project,
        name=training_name,
        exist_ok=exist_ok,
    )


if __name__ == "__main__":
    main()
