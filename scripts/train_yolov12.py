"""Train YOLOv12 from an experiment configuration file."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from pothole_severity_detection.experiment_config import load_experiment_config
from pothole_severity_detection.torch_utils import select_device


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


def copy_best_weights(
    training_project: str,
    training_name: str,
    destination: str | Path,
) -> None:
    """Copy the best training weights to a configured local destination."""
    source_path = Path(training_project) / training_name / "weights" / "best.pt"
    destination_path = Path(destination).expanduser()

    if not source_path.exists():
        raise FileNotFoundError(f"Best weights not found after training: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)

    print(f"Best weights copied to: {destination_path}")


def main() -> None:
    """Train YOLOv12 using settings from a YAML config file."""
    args = parse_args()
    config = load_experiment_config(args.config)

    experiment_name = config.experiment.name
    data_path = resolve_dataset_path(config.dataset.data_yaml)
    model_source = resolve_model_source(config.model.source)

    device = config.training.device
    if device == "auto":
        device = select_device()

    epochs = config.training.epochs_to_run
    image_size = config.training.image_size
    batch_size = config.training.batch_size
    workers = config.training.workers
    amp = config.training.amp
    training_overrides = config.training.augmentation_overrides()

    training_project = config.outputs.training_project
    training_name = config.training_name
    exist_ok = config.outputs.exist_ok
    output_weights = config.model.output_weights

    print("YOLOv12 training configuration")
    print()
    print(f"Config: {config.path}")
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

    if training_overrides:
        print()
        print("Additional training overrides:")
        for key, value in training_overrides.items():
            print(f"  {key}: {value}")

    if output_weights:
        print()
        print(f"Configured output weights: {output_weights}")

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
        **training_overrides,
    )

    if output_weights:
        copy_best_weights(
            training_project=training_project,
            training_name=training_name,
            destination=output_weights,
        )


if __name__ == "__main__":
    main()
