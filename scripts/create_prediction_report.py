"""Create a lightweight prediction report for YOLOv12 image inference.

The report summarizes detections on an image directory and stores annotated
sample images, a JSON summary, and a Markdown report.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import yaml
from ultralytics import YOLO

from pothole_severity_detection.inference.detector import (
    draw_boxes_with_severity,
    is_image_file,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a lightweight YOLOv12 prediction report."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment configuration file.",
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional image source override.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional report output directory override.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Optional confidence threshold override.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of annotated sample images to save.",
    )

    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
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


def collect_image_files(source: Path) -> list[Path]:
    """Collect supported image files from a file or directory."""
    if source.is_file():
        if not is_image_file(source):
            raise ValueError(f"Unsupported image file: {source}")

        return [source]

    if source.is_dir():
        return sorted(
            path for path in source.glob("*") if path.is_file() and is_image_file(path)
        )

    raise FileNotFoundError(f"Source path not found: {source}")


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a dictionary as a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    """Save a Markdown report."""
    metrics = summary["summary"]

    content = f"""# Prediction Report

Generated at: {summary["timestamp_utc"]}

## Configuration

| Field | Value |
|---|---|
| Experiment | {summary["experiment_name"]} |
| Model | `{summary["model_path"]}` |
| Source | `{summary["source"]}` |
| Confidence threshold | {summary["confidence"]} |

## Summary

| Metric | Value |
|---|---:|
| Images processed | {metrics["images_processed"]} |
| Images with detections | {metrics["images_with_detections"]} |
| Total detections | {metrics["total_detections"]} |
| Average detections per image | {metrics["avg_detections_per_image"]:.3f} |
| Average confidence | {metrics["avg_confidence"]:.3f} |
| Maximum confidence | {metrics["max_confidence"]:.3f} |

## Notes

This report is based on model predictions only. It does not perform ground-truth
matching or false positive / false negative analysis yet.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    """Create prediction report."""
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_yaml_config(config_path)

    experiment_config = get_section(config, "experiment")
    model_config = get_section(config, "model")
    prediction_config = get_section(config, "prediction")

    experiment_name = str(experiment_config.get("name", config_path.stem))

    model_value = model_config.get("output_weights") or model_config.get("source")
    if model_value is None:
        raise ValueError("Config model section must define output_weights or source.")

    source_value = args.source or prediction_config.get("source")
    if source_value is None:
        raise ValueError("Prediction source must be provided.")

    confidence = float(
        args.confidence
        if args.confidence is not None
        else prediction_config.get("confidence", 0.25)
    )

    model_path = resolve_existing_path(model_value, "Model weights")
    source = resolve_existing_path(source_value, "Prediction source")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path("outputs/reports") / experiment_name
    )
    output_dir = output_dir.resolve()

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_image_files(source)

    if not image_files:
        raise FileNotFoundError(f"No supported image files found in: {source}")

    model = YOLO(str(model_path))

    image_summaries: list[dict[str, Any]] = []
    all_confidences: list[float] = []

    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Output directory: {output_dir}")
    print(f"Images to process: {len(image_files)}")

    for index, image_path in enumerate(image_files, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"[{index}/{len(image_files)}] Skipping unreadable image: {image_path}"
            )
            continue

        results = model.predict(image, conf=confidence, classes=[0], verbose=False)
        result = results[0]

        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else []
        confidences = (
            result.boxes.conf.cpu().numpy().tolist() if len(result.boxes) else []
        )

        all_confidences.extend(float(value) for value in confidences)

        annotated_image = image.copy()
        draw_boxes_with_severity(annotated_image, boxes)

        detection_count = len(boxes)
        max_confidence = max(confidences) if confidences else 0.0

        image_summaries.append(
            {
                "image": str(image_path),
                "detections": detection_count,
                "max_confidence": float(max_confidence),
            }
        )

        if index <= args.max_samples:
            sample_path = samples_dir / f"{image_path.stem}_annotated.jpg"
            cv2.imwrite(str(sample_path), annotated_image)

        print(
            f"[{index}/{len(image_files)}] {image_path.name}: "
            f"{detection_count} detections"
        )

    total_images = len(image_summaries)
    images_with_detections = sum(
        1 for image_summary in image_summaries if image_summary["detections"] > 0
    )
    total_detections = sum(
        image_summary["detections"] for image_summary in image_summaries
    )

    summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "experiment_name": experiment_name,
        "model_path": str(model_path),
        "source": str(source),
        "confidence": confidence,
        "summary": {
            "images_processed": total_images,
            "images_with_detections": images_with_detections,
            "total_detections": total_detections,
            "avg_detections_per_image": (
                total_detections / total_images if total_images else 0.0
            ),
            "avg_confidence": mean(all_confidences) if all_confidences else 0.0,
            "max_confidence": max(all_confidences) if all_confidences else 0.0,
        },
        "images": image_summaries,
    }

    save_json(summary, output_dir / "summary.json")
    save_markdown_report(summary, output_dir / "report.md")

    print()
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    print(f"Report saved to: {output_dir / 'report.md'}")
    print(f"Samples saved to: {samples_dir}")


if __name__ == "__main__":
    main()
