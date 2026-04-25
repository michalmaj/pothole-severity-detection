"""Pothole detection and severity visualization."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
from ultralytics import YOLO

from pothole_severity_detection.inference.severity import estimate_severity


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_video_file(path: str | Path) -> bool:
    """Check whether a path points to a supported video file."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: str | Path) -> bool:
    """Check whether a path points to a supported image file."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def draw_boxes_with_severity(frame, boxes) -> None:
    """Draw detection boxes and severity labels on an image frame."""
    frame_height, frame_width = frame.shape[:2]

    for x1, y1, x2, y2 in boxes:
        bbox_area = (x2 - x1) * (y2 - y1)
        y_center = (y1 + y2) / 2.0

        severity = estimate_severity(
            bbox_area=bbox_area,
            y_center=y_center,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))
        label_position = (int(x1), max(int(y1) - 10, 20))

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(
            frame,
            severity,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def detect_image(
    model: YOLO,
    image_path: str | Path,
    output_dir: str | Path = "outputs/inference",
    confidence: float = 0.4,
) -> Path:
    """Run pothole detection on a single image."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = model.predict(image, conf=confidence, classes=[0], verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) else []

    draw_boxes_with_severity(image, boxes)

    output_path = output_dir / f"{image_path.stem}_detected_{uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(output_path), image)

    return output_path


def detect_video(
    model: YOLO,
    video_path: str | Path,
    output_dir: str | Path = "outputs/inference",
    confidence: float = 0.4,
    max_seconds: int = 10,
) -> Path:
    """Run pothole detection on a video file."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    output_path = output_dir / f"{video_path.stem}_detected_{uuid4().hex[:8]}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    max_frames = int(fps * max_seconds)
    frame_index = 0

    while capture.isOpened():
        success, frame = capture.read()

        if not success or frame is None or frame_index >= max_frames:
            break

        results = model.predict(frame, conf=confidence, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) else []

        draw_boxes_with_severity(frame, boxes)
        writer.write(frame)

        frame_index += 1

    capture.release()
    writer.release()

    return output_path


def detect_media(
    model: YOLO,
    input_path: str | Path,
    output_dir: str | Path = "outputs/inference",
    confidence: float = 0.4,
) -> Path:
    """Run detection on an image or video file."""
    input_path = Path(input_path)

    if is_video_file(input_path):
        return detect_video(
            model=model,
            video_path=input_path,
            output_dir=output_dir,
            confidence=confidence,
        )

    if is_image_file(input_path):
        return detect_image(
            model=model,
            image_path=input_path,
            output_dir=output_dir,
            confidence=confidence,
        )

    raise ValueError(f"Unsupported media format: {input_path.suffix}")
