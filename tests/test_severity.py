"""Tests for pothole severity estimation heuristics."""

from __future__ import annotations

import pytest

from pothole_severity_detection.inference.severity import estimate_severity


def test_estimate_severity_returns_low_for_small_upper_bbox() -> None:
    severity = estimate_severity(
        bbox_area=100.0,
        y_center=50.0,
        frame_width=1000,
        frame_height=1000,
    )

    assert severity == "Low"


def test_estimate_severity_returns_high_for_large_lower_bbox() -> None:
    severity = estimate_severity(
        bbox_area=300_000.0,
        y_center=900.0,
        frame_width=1000,
        frame_height=1000,
    )

    assert severity == "High"


def test_estimate_severity_raises_error_for_invalid_frame_area() -> None:
    with pytest.raises(ValueError, match="Frame area must be greater than zero"):
        estimate_severity(
            bbox_area=100.0,
            y_center=50.0,
            frame_width=0,
            frame_height=1000,
        )
