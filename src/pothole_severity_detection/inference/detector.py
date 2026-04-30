"""Pothole detection and severity visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from ultralytics import YOLO

from pothole_severity_detection.inference.severity import estimate_severity

Box = tuple[float, float, float, float]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SEVERITY_LEVELS = ("Low", "Medium", "High")

# OpenCV uses BGR color order.
SEVERITY_COLORS = {
    "Low": (0, 180, 0),
    "Medium": (0, 165, 255),
    "High": (0, 0, 255),
}


def is_video_file(path: str | Path) -> bool:
    """Check whether a path points to a supported video file."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: str | Path) -> bool:
    """Check whether a path points to a supported image file."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def get_severity_color(severity: str) -> tuple[int, int, int]:
    """Return BGR color for a severity label."""
    return SEVERITY_COLORS.get(severity, (255, 255, 255))


def create_empty_summary(media_type: str) -> dict[str, Any]:
    """Create an empty detection summary."""
    return {
        "media_type": media_type,
        "total_detections": 0,
        "severity_counts": {severity: 0 for severity in SEVERITY_LEVELS},
        "average_confidence": 0.0,
        "max_confidence": 0.0,
    }


def calculate_detection_details(
    frame: Any,
    boxes: list[Box],
    confidences: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Calculate severity and metadata for detection boxes."""
    frame_height, frame_width = frame.shape[:2]
    detection_details: list[dict[str, Any]] = []

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        bbox_area = (x2 - x1) * (y2 - y1)
        y_center = (y1 + y2) / 2.0

        severity = estimate_severity(
            bbox_area=bbox_area,
            y_center=y_center,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        confidence = confidences[index] if confidences is not None else None

        detection_details.append(
            {
                "box": box,
                "severity": severity,
                "confidence": confidence,
            }
        )

    return detection_details


def summarize_detection_details(
    detection_details: list[dict[str, Any]],
    media_type: str,
    frames_processed: int | None = None,
) -> dict[str, Any]:
    """Create a detection summary from detection details."""
    severity_counts = {severity: 0 for severity in SEVERITY_LEVELS}
    confidences: list[float] = []

    for detection in detection_details:
        severity = str(detection["severity"])
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

        confidence = detection.get("confidence")
        if confidence is not None:
            confidences.append(float(confidence))

    summary = {
        "media_type": media_type,
        "total_detections": len(detection_details),
        "severity_counts": severity_counts,
        "average_confidence": (
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
        "max_confidence": max(confidences) if confidences else 0.0,
    }

    if frames_processed is not None:
        summary["frames_processed"] = frames_processed

    return summary


def extract_boxes_and_confidences(result: Any) -> tuple[list[Box], list[float]]:
    """Extract boxes and confidence scores from an Ultralytics result."""
    if not len(result.boxes):
        return [], []

    boxes = [
        tuple(float(value) for value in box)
        for box in result.boxes.xyxy.cpu().numpy().tolist()
    ]
    confidences = [
        float(confidence) for confidence in result.boxes.conf.cpu().numpy().tolist()
    ]

    return boxes, confidences


def draw_detection_details(frame: Any, detection_details: list[dict[str, Any]]) -> None:
    """Draw detection boxes, severity labels, and confidence scores."""
    for detection in detection_details:
        x1, y1, x2, y2 = detection["box"]
        severity = str(detection["severity"])
        confidence = detection.get("confidence")

        color = get_severity_color(severity)
        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))
        label_position = (int(x1), max(int(y1) - 10, 20))

        if confidence is None:
            label = severity
        else:
            label = f"{severity} | conf={confidence:.2f}"

        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        cv2.putText(
            frame,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_boxes_with_severity(
    frame: Any,
    boxes: list[Box],
    confidences: list[float] | None = None,
) -> dict[str, Any]:
    """Draw detection boxes and severity labels on an image frame."""
    detection_details = calculate_detection_details(
        frame=frame,
        boxes=boxes,
        confidences=confidences,
    )
    draw_detection_details(frame=frame, detection_details=detection_details)

    return summarize_detection_details(
        detection_details=detection_details,
        media_type="image",
    )


def detect_image_with_summary(
    model: YOLO,
    image_path: str | Path,
    output_dir: str | Path = "outputs/inference",
    confidence: float = 0.4,
) -> dict[str, Any]:
    """Run pothole detection on a single image and return output metadata."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = model.predict(image, conf=confidence, classes=[0], verbose=False)
    boxes, confidences = extract_boxes_and_confidences(results[0])

    summary = draw_boxes_with_severity(
        frame=image,
        boxes=boxes,
        confidences=confidences,
    )
    summary["source"] = str(image_path)

    output_path = output_dir / f"{image_path.stem}_detected_{uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(output_path), image)

    return {
        "output_path": output_path,
        "summary": summary,
    }


def detect_image(
    model: YOLO,
    image_path: str | Path,
    output_dir: str | Path = "outputs/inference",
    confidence: float = 0.4,
) -> Path:
    """Run pothole detection on a single image."""
    result = detect_image_with_summary(
        model=model,
        image_path=image_path,
        output_dir=output_dir,
        confidence=confidence,
    )

    return Path(result["output_path"])


def detect_video_with_summary(
    model: YOLO,
    video_path: str | Path,
    output_dir: str | Path = "outputs/inference",
    confidence: float = 0.4,
    max_seconds: int = 10,
) -> dict[str, Any]:
    """Run pothole detection on a video file and return output metadata."""
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
    all_detection_details: list[dict[str, Any]] = []

    while capture.isOpened():
        success, frame = capture.read()

        if not success or frame is None or frame_index >= max_frames:
            break

        results = model.predict(frame, conf=confidence, classes=[0], verbose=False)
        boxes, confidences = extract_boxes_and_confidences(results[0])

        detection_details = calculate_detection_details(
            frame=frame,
            boxes=boxes,
            confidences=confidences,
        )
        draw_detection_details(frame=frame, detection_details=detection_details)

        all_detection_details.extend(detection_details)
        writer.write(frame)

        frame_index += 1

    capture.release()
    writer.release()

    summary = summarize_detection_details(
        detection_details=all_detection_details,
        media_type="video",
        frames_processed=frame_index,
    )
    summary["source"] = str(video_path)

    return {
        "output_path": output_path,
        "summary": summary,
    }


def detect_video(
    model: YOLO,
    video_path: str | Path,
    output_dir: str | Path = "outputs/inference",
    confidence: float = 0.4,
    max_seconds: int = 10,
) -> Path:
    """Run pothole detection on a video file."""
    result = detect_video_with_summary(
        model=model,
        video_path=video_path,
        output_dir=output_dir,
        confidence=confidence,
        max_seconds=max_seconds,
    )

    return Path(result["output_path"])


def detect_media_with_summary(
    model: YOLO,
    input_path: str | Path,
    output_dir: str | Path = "outputs/inference",
    confidence: float = 0.4,
) -> dict[str, Any]:
    """Run detection on an image or video file and return output metadata."""
    input_path = Path(input_path)

    if is_video_file(input_path):
        return detect_video_with_summary(
            model=model,
            video_path=input_path,
            output_dir=output_dir,
            confidence=confidence,
        )

    if is_image_file(input_path):
        return detect_image_with_summary(
            model=model,
            image_path=input_path,
            output_dir=output_dir,
            confidence=confidence,
        )

    raise ValueError(f"Unsupported media format: {input_path.suffix}")


def detect_media(
    model: YOLO,
    input_path: str | Path,
    output_dir: str | Path = "outputs/inference",
    confidence: float = 0.4,
) -> Path:
    """Run detection on an image or video file."""
    result = detect_media_with_summary(
        model=model,
        input_path=input_path,
        output_dir=output_dir,
        confidence=confidence,
    )

    return Path(result["output_path"])
