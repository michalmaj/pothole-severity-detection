"""Analyze YOLOv12 detection errors against YOLO-format ground-truth labels.

The script compares model predictions with YOLO labels and reports:
- true positives,
- false positives,
- false negatives,
- precision,
- recall,
- F1 score,
- matched IoU statistics.

It also saves annotated samples for qualitative error analysis.
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

from pothole_severity_detection.inference.detector import is_image_file

Box = tuple[float, float, float, float]

GT_COLOR = (0, 255, 0)
TP_COLOR = (255, 0, 0)
FP_COLOR = (0, 0, 255)
FN_COLOR = (0, 165, 255)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze YOLOv12 prediction errors against ground truth."
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
        "--model",
        type=Path,
        default=None,
        help="Optional model weights override.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional error analysis output directory override.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Optional prediction confidence threshold override.",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for matching predictions with ground truth.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Optional inference image size override.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional inference device override.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )

    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=15,
        help="Maximum number of sample images saved per error category.",
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


def collect_image_files(source: Path, limit: int | None = None) -> list[Path]:
    """Collect supported image files from a file or directory."""
    if source.is_file():
        if not is_image_file(source):
            raise ValueError(f"Unsupported image file: {source}")

        return [source]

    if source.is_dir():
        image_files = sorted(
            path for path in source.glob("*") if path.is_file() and is_image_file(path)
        )

        if limit is not None:
            return image_files[:limit]

        return image_files

    raise FileNotFoundError(f"Source path not found: {source}")


def get_label_path(image_path: Path) -> Path:
    """Infer YOLO label path for an image path."""
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"

    return image_path.with_suffix(".txt")


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
) -> Box:
    """Convert normalized YOLO xywh box to absolute xyxy coordinates."""
    box_width = width * frame_width
    box_height = height * frame_height
    center_x = x_center * frame_width
    center_y = y_center * frame_height

    x1 = center_x - box_width / 2.0
    y1 = center_y - box_height / 2.0
    x2 = center_x + box_width / 2.0
    y2 = center_y + box_height / 2.0

    return x1, y1, x2, y2


def load_ground_truth_boxes(
    label_path: Path,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Load YOLO-format ground-truth labels."""
    if not label_path.exists():
        return []

    ground_truths: list[dict[str, Any]] = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()

        if not stripped_line:
            continue

        parts = stripped_line.split()

        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        x_center, y_center, width, height = (float(value) for value in parts[1:5])

        box = yolo_to_xyxy(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        ground_truths.append(
            {
                "class_id": class_id,
                "box": box,
            }
        )

    return ground_truths


def calculate_iou(box_a: Box, box_b: Box) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def predict_boxes(
    model: YOLO,
    image: Any,
    confidence: float,
    device: str | None,
    image_size: int | None,
) -> list[dict[str, Any]]:
    """Run model prediction and return boxes with confidence scores."""
    predict_kwargs: dict[str, Any] = {
        "source": image,
        "conf": confidence,
        "classes": [0],
        "verbose": False,
    }

    if device is not None:
        predict_kwargs["device"] = device

    if image_size is not None:
        predict_kwargs["imgsz"] = image_size

    results = model.predict(**predict_kwargs)
    result = results[0]

    if not len(result.boxes):
        return []

    boxes = result.boxes.xyxy.cpu().numpy().tolist()
    confidences = result.boxes.conf.cpu().numpy().tolist()

    return [
        {
            "box": tuple(float(value) for value in box),
            "confidence": float(confidence_score),
        }
        for box, confidence_score in zip(boxes, confidences, strict=True)
    ]


def match_predictions(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    """Match predictions to ground truth using greedy IoU matching."""
    unmatched_gt_indices = set(range(len(ground_truths)))
    matches: list[dict[str, Any]] = []
    false_positives: list[int] = []

    sorted_prediction_indices = sorted(
        range(len(predictions)),
        key=lambda index: predictions[index]["confidence"],
        reverse=True,
    )

    for prediction_index in sorted_prediction_indices:
        prediction_box = predictions[prediction_index]["box"]

        best_gt_index = None
        best_iou = 0.0

        for gt_index in unmatched_gt_indices:
            gt_box = ground_truths[gt_index]["box"]
            iou = calculate_iou(prediction_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            matches.append(
                {
                    "prediction_index": prediction_index,
                    "ground_truth_index": best_gt_index,
                    "iou": best_iou,
                }
            )
            unmatched_gt_indices.remove(best_gt_index)
        else:
            false_positives.append(prediction_index)

    false_negatives = sorted(unmatched_gt_indices)

    return {
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def draw_box(image: Any, box: Box, color: tuple[int, int, int], label: str) -> None:
    """Draw a bounding box with a text label."""
    x1, y1, x2, y2 = (int(value) for value in box)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def draw_error_overlay(
    image: Any,
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    matching: dict[str, Any],
) -> Any:
    """Draw ground truth, true positives, false positives, and false negatives."""
    annotated = image.copy()

    for ground_truth in ground_truths:
        draw_box(annotated, ground_truth["box"], GT_COLOR, "GT")

    for match in matching["matches"]:
        prediction = predictions[match["prediction_index"]]
        label = f"TP {prediction['confidence']:.2f} IoU {match['iou']:.2f}"
        draw_box(annotated, prediction["box"], TP_COLOR, label)

    for prediction_index in matching["false_positives"]:
        prediction = predictions[prediction_index]
        label = f"FP {prediction['confidence']:.2f}"
        draw_box(annotated, prediction["box"], FP_COLOR, label)

    for ground_truth_index in matching["false_negatives"]:
        ground_truth = ground_truths[ground_truth_index]
        draw_box(annotated, ground_truth["box"], FN_COLOR, "FN")

    return annotated


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers."""
    if denominator == 0:
        return 0.0

    return numerator / denominator


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save dictionary as a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    """Save Markdown error analysis report."""
    totals = summary["totals"]
    metrics = summary["metrics"]

    content = f"""# YOLOv12 Error Analysis Report

Generated at: {summary["timestamp_utc"]}

## Configuration

| Field | Value |
|---|---|
| Experiment | {summary["experiment_name"]} |
| Model | `{summary["model_path"]}` |
| Source | `{summary["source"]}` |
| Confidence threshold | {summary["confidence"]} |
| IoU threshold | {summary["iou_threshold"]} |

## Detection Summary

| Metric | Value |
|---|---:|
| Images processed | {totals["images_processed"]} |
| Ground-truth boxes | {totals["ground_truth_boxes"]} |
| Predicted boxes | {totals["predicted_boxes"]} |
| True positives | {totals["true_positives"]} |
| False positives | {totals["false_positives"]} |
| False negatives | {totals["false_negatives"]} |

## Metrics

| Metric | Value |
|---|---:|
| Precision | {metrics["precision"]:.3f} |
| Recall | {metrics["recall"]:.3f} |
| F1 score | {metrics["f1_score"]:.3f} |
| Average matched IoU | {metrics["average_matched_iou"]:.3f} |

## Notes

This report uses greedy IoU matching between predictions 
and YOLO-format ground-truth labels.
A prediction is counted as a true positive when its IoU 
with an unmatched ground-truth box
is greater than or equal to the configured IoU threshold.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    """Run YOLOv12 error analysis."""
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_yaml_config(config_path)

    experiment_config = get_section(config, "experiment")
    model_config = get_section(config, "model")
    prediction_config = get_section(config, "prediction")
    training_config = get_section(config, "training")

    experiment_name = str(experiment_config.get("name", config_path.stem))

    model_value = args.model or model_config.get("output_weights")
    if model_value is None:
        raise ValueError("Model weights must be provided in config or via --model.")

    source_value = args.source or prediction_config.get("source")
    if source_value is None:
        raise ValueError(
            "Prediction source must be provided in config or via --source."
        )

    confidence = float(
        args.confidence
        if args.confidence is not None
        else prediction_config.get("confidence", 0.25)
    )

    image_size = args.imgsz or training_config.get("image_size")
    image_size = int(image_size) if image_size is not None else None

    device = (
        args.device or prediction_config.get("device") or training_config.get("device")
    )
    device = str(device) if device is not None else None

    model_path = resolve_existing_path(model_value, "Model weights")
    source = resolve_existing_path(source_value, "Prediction source")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path("outputs/error_analysis") / experiment_name
    )
    output_dir = output_dir.resolve()

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    false_positive_dir = samples_dir / "false_positives"
    false_negative_dir = samples_dir / "false_negatives"
    mixed_errors_dir = samples_dir / "mixed_errors"

    false_positive_dir.mkdir(parents=True, exist_ok=True)
    false_negative_dir.mkdir(parents=True, exist_ok=True)
    mixed_errors_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_image_files(source=source, limit=args.limit)

    if not image_files:
        raise FileNotFoundError(f"No supported image files found in: {source}")

    print("YOLOv12 error analysis configuration")
    print()
    print(f"Experiment: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {confidence}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")
    print(f"Images to process: {len(image_files)}")

    model = YOLO(str(model_path))

    image_summaries: list[dict[str, Any]] = []
    matched_ious: list[float] = []

    saved_false_positive_samples = 0
    saved_false_negative_samples = 0
    saved_mixed_error_samples = 0

    for index, image_path in enumerate(image_files, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"[{index}/{len(image_files)}] Skipping unreadable image: {image_path}"
            )
            continue

        frame_height, frame_width = image.shape[:2]
        label_path = get_label_path(image_path)

        ground_truths = load_ground_truth_boxes(
            label_path=label_path,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        predictions = predict_boxes(
            model=model,
            image=image,
            confidence=confidence,
            device=device,
            image_size=image_size,
        )

        matching = match_predictions(
            predictions=predictions,
            ground_truths=ground_truths,
            iou_threshold=args.iou_threshold,
        )

        matches = matching["matches"]
        false_positives = matching["false_positives"]
        false_negatives = matching["false_negatives"]

        matched_ious.extend(float(match["iou"]) for match in matches)

        image_summary = {
            "image": str(image_path),
            "label": str(label_path),
            "ground_truth_count": len(ground_truths),
            "prediction_count": len(predictions),
            "true_positives": len(matches),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "matched_ious": [float(match["iou"]) for match in matches],
        }

        image_summaries.append(image_summary)

        has_false_positives = bool(false_positives)
        has_false_negatives = bool(false_negatives)

        if has_false_positives or has_false_negatives:
            annotated = draw_error_overlay(
                image=image,
                predictions=predictions,
                ground_truths=ground_truths,
                matching=matching,
            )

            if (
                has_false_positives
                and not has_false_negatives
                and saved_false_positive_samples < args.max_samples_per_category
            ):
                sample_path = false_positive_dir / f"{image_path.stem}_fp.jpg"
                cv2.imwrite(str(sample_path), annotated)
                saved_false_positive_samples += 1

            elif (
                has_false_negatives
                and not has_false_positives
                and saved_false_negative_samples < args.max_samples_per_category
            ):
                sample_path = false_negative_dir / f"{image_path.stem}_fn.jpg"
                cv2.imwrite(str(sample_path), annotated)
                saved_false_negative_samples += 1

            elif saved_mixed_error_samples < args.max_samples_per_category:
                sample_path = mixed_errors_dir / f"{image_path.stem}_mixed.jpg"
                cv2.imwrite(str(sample_path), annotated)
                saved_mixed_error_samples += 1

        print(
            f"[{index}/{len(image_files)}] {image_path.name}: "
            f"GT={len(ground_truths)}, Pred={len(predictions)}, "
            f"TP={len(matches)}, FP={len(false_positives)}, "
            f"FN={len(false_negatives)}"
        )

    total_ground_truths = sum(
        image_summary["ground_truth_count"] for image_summary in image_summaries
    )
    total_predictions = sum(
        image_summary["prediction_count"] for image_summary in image_summaries
    )
    total_true_positives = sum(
        image_summary["true_positives"] for image_summary in image_summaries
    )
    total_false_positives = sum(
        image_summary["false_positives"] for image_summary in image_summaries
    )
    total_false_negatives = sum(
        image_summary["false_negatives"] for image_summary in image_summaries
    )

    precision = safe_divide(
        total_true_positives,
        total_true_positives + total_false_positives,
    )
    recall = safe_divide(
        total_true_positives,
        total_true_positives + total_false_negatives,
    )
    f1_score = safe_divide(2.0 * precision * recall, precision + recall)

    summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "experiment_name": experiment_name,
        "model_path": str(model_path),
        "source": str(source),
        "confidence": confidence,
        "iou_threshold": args.iou_threshold,
        "totals": {
            "images_processed": len(image_summaries),
            "ground_truth_boxes": total_ground_truths,
            "predicted_boxes": total_predictions,
            "true_positives": total_true_positives,
            "false_positives": total_false_positives,
            "false_negatives": total_false_negatives,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_matched_iou": mean(matched_ious) if matched_ious else 0.0,
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
