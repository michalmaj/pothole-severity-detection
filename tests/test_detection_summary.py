"""Tests for detection summary helpers."""

from __future__ import annotations

import numpy as np

from pothole_severity_detection.inference.detector import (
    calculate_detection_details,
    get_severity_color,
    summarize_detection_details,
)


def test_detection_summary_counts_severity_and_confidence() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [
        (1.0, 1.0, 10.0, 10.0),
        (10.0, 60.0, 90.0, 99.0),
    ]
    confidences = [0.5, 0.9]

    detection_details = calculate_detection_details(
        frame=frame,
        boxes=boxes,
        confidences=confidences,
    )
    summary = summarize_detection_details(
        detection_details=detection_details,
        media_type="image",
    )

    assert summary["total_detections"] == 2
    assert summary["severity_counts"]["Low"] == 1
    assert summary["severity_counts"]["High"] == 1
    assert summary["average_confidence"] == 0.7
    assert summary["max_confidence"] == 0.9


def test_get_severity_color_returns_fallback_for_unknown_label() -> None:
    assert get_severity_color("Unknown") == (255, 255, 255)
