"""Evaluate a YOLOv12 model.

The script supports two workflows:

1. Manual CLI arguments.
2. Config-driven evaluation using a YAML experiment configuration.

A provenance-rich result record is written after evaluation: to
``docs/results/<experiment>.eval.yaml`` in config mode, or to
``<output-dir>/<run>/result.eval.yaml`` in pure-CLI mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from pothole_severity_detection import provenance
from pothole_severity_detection.experiment_config import (
    ExperimentConfig,
    load_experiment_config,
)
from pothole_severity_detection.results import (
    NOT_RECORDED,
    extract_box_metrics,
    write_eval_result,
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
        help="Directory where evaluation outputs will be saved.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate settings without running evaluation.",
    )

    return parser.parse_args()


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


def count_split_images(data_path: Path, split: str) -> dict[str, int | str]:
    """Best-effort count of images in the evaluation split directory."""
    images_dir = data_path.parent / split / "images"
    if images_dir.is_dir():
        images: int | str = sum(1 for item in images_dir.iterdir() if item.is_file())
    else:
        images = NOT_RECORDED
    return {"images": images, "instances": NOT_RECORDED}


def main() -> None:
    """Run YOLOv12 evaluation and write a result record."""
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

    seed = config.training.seed if config is not None else 0
    deterministic = config.training.deterministic if config is not None else True

    output_dir, run_name = get_output_settings(args=args, config=config, split=split)

    if config is not None:
        record_path = Path("docs/results") / f"{config.experiment.name}.eval.yaml"
    else:
        record_path = output_dir / run_name / "result.eval.yaml"

    print("YOLOv12 evaluation configuration")
    print()
    print(f"Config: {config.path if config is not None else None}")
    print(f"Model: {model_path}")
    print(f"Dataset YAML: {data_path}")
    print(f"Split: {split}")
    print(f"Device: {device}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Seed: {seed} (deterministic={deterministic})")
    print(f"Output directory: {output_dir}")
    print(f"Run name: {run_name}")
    print(f"Result record: {record_path}")

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
        seed=seed,
        deterministic=deterministic,
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    write_eval_result(
        record_path,
        experiment=config.experiment.name if config is not None else run_name,
        config=(
            provenance.to_repo_relative(config.path)
            if config is not None
            else NOT_RECORDED
        ),
        dataset=provenance.to_repo_relative(data_path),
        split=split,
        model=provenance.to_repo_relative(model_path),
        device=device,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        deterministic=deterministic,
        metrics=extract_box_metrics(results),
        sample_counts=count_split_images(data_path, split),
    )

    print(f"Result record saved to: {record_path}")


if __name__ == "__main__":
    main()
