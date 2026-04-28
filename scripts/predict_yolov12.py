"""Run YOLOv12 prediction on images, videos, or directories.

This script supports two workflows:

1. Manual CLI arguments.
2. Config-driven prediction using a YAML experiment configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from pothole_severity_detection.inference.detector import (
    detect_media,
    is_image_file,
    is_video_file,
)
from pothole_severity_detection.inference.model_loader import load_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv12 prediction on images, videos, or directories."
    )

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
        "--source",
        type=Path,
        default=None,
        help="Path to an image, video, or directory. Overrides config value.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where annotated outputs will be saved.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Prediction confidence threshold. Overrides config value.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for media files when source is a directory.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate settings without running prediction.",
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


def resolve_existing_path(path_value: str | Path, label: str) -> Path:
    """Resolve and validate an existing local path."""
    path = Path(path_value).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    return path


def get_model_path(args: argparse.Namespace, model_config: dict[str, Any]) -> Path:
    """Resolve model weights path from CLI arguments or experiment config."""
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
            "Prediction requires trained model weights with `.pt` extension. "
            f"Received: {model_path}"
        )

    return model_path


def get_source_path(
    args: argparse.Namespace,
    prediction_config: dict[str, Any],
) -> Path:
    """Resolve prediction source path from CLI arguments or config."""
    source_value = args.source or prediction_config.get("source")

    if source_value is None:
        raise ValueError(
            "Prediction source must be provided through --source or config "
            "prediction.source."
        )

    return resolve_existing_path(source_value, "Prediction source")


def get_output_dir(
    args: argparse.Namespace,
    prediction_config: dict[str, Any],
) -> Path:
    """Resolve output directory from CLI arguments or config."""
    output_value = (
        args.output_dir
        or prediction_config.get("output_dir")
        or Path("outputs/predictions")
    )

    return Path(output_value).expanduser().resolve()


def is_supported_media(path: Path) -> bool:
    """Check whether a file is a supported image or video."""
    return is_image_file(path) or is_video_file(path)


def collect_media_files(source: Path, recursive: bool) -> list[Path]:
    """Collect supported media files from a file or directory path."""
    if source.is_file():
        if not is_supported_media(source):
            raise ValueError(f"Unsupported media file: {source}")

        return [source]

    if source.is_dir():
        pattern = "**/*" if recursive else "*"

        return sorted(
            path
            for path in source.glob(pattern)
            if path.is_file() and is_supported_media(path)
        )

    raise FileNotFoundError(f"Source path not found: {source}")


def main() -> None:
    """Run prediction workflow."""
    args = parse_args()

    config_path = args.config.expanduser().resolve() if args.config else None
    config = load_yaml_config(config_path)

    model_config = get_section(config, "model")
    prediction_config = get_section(config, "prediction")

    model_path = get_model_path(args=args, model_config=model_config)
    source = get_source_path(args=args, prediction_config=prediction_config)
    output_dir = get_output_dir(args=args, prediction_config=prediction_config)

    confidence = float(
        args.confidence
        if args.confidence is not None
        else prediction_config.get("confidence", 0.25)
    )

    recursive = bool(args.recursive or prediction_config.get("recursive", False))

    media_files = collect_media_files(source=source, recursive=recursive)

    if not media_files:
        raise FileNotFoundError(f"No supported media files found in: {source}")

    print("YOLOv12 prediction configuration")
    print()
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {confidence}")
    print(f"Recursive: {recursive}")
    print(f"Files to process: {len(media_files)}")

    if args.dry_run:
        print()
        print("Dry run completed successfully. Prediction was not started.")
        return

    model = load_model(model_path)

    for index, media_path in enumerate(media_files, start=1):
        print(f"[{index}/{len(media_files)}] Processing: {media_path}")

        output_path = detect_media(
            model=model,
            input_path=media_path,
            output_dir=output_dir,
            confidence=confidence,
        )

        print(f"Saved: {output_path}")

    print("Prediction completed successfully.")


if __name__ == "__main__":
    main()
