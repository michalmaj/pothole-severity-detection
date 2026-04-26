"""Run YOLOv12 prediction on images, videos, or directories.

This script provides a command-line inference workflow independent from the
Gradio application. It loads trained YOLO weights, processes one media file or
all supported files in a directory, and saves annotated outputs locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
        "--model",
        type=Path,
        default=None,
        help="Path to YOLO model weights. If omitted, the default model path is used.",
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to an image, video, or directory with media files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Directory where annotated outputs will be saved.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Prediction confidence threshold.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for media files when source is a directory.",
    )

    return parser.parse_args()


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

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()

    media_files = collect_media_files(source=source, recursive=args.recursive)

    if not media_files:
        raise FileNotFoundError(f"No supported media files found in: {source}")

    model = load_model(args.model)

    print(f"Source: {source}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Files to process: {len(media_files)}")

    for index, media_path in enumerate(media_files, start=1):
        print(f"[{index}/{len(media_files)}] Processing: {media_path}")

        output_path = detect_media(
            model=model,
            input_path=media_path,
            output_dir=output_dir,
            confidence=args.confidence,
        )

        print(f"Saved: {output_path}")

    print("Prediction completed successfully.")


if __name__ == "__main__":
    main()
