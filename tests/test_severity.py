"""Tests for the severity heuristic."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pothole_severity_detection.inference.severity import (
    SeverityParameters,
    estimate_severity,
    severity_score,
)


def test_estimate_severity_returns_low_for_small_upper_bbox() -> None:
    assert (
        estimate_severity(
            bbox_area=100.0, y_center=50.0, frame_width=1000, frame_height=1000
        )
        == "Low"
    )


def test_estimate_severity_returns_high_for_large_lower_bbox() -> None:
    assert (
        estimate_severity(
            bbox_area=300_000.0,
            y_center=900.0,
            frame_width=1000,
            frame_height=1000,
        )
        == "High"
    )


def test_severity_score_raises_for_invalid_frame_area() -> None:
    with pytest.raises(ValueError, match="Frame area must be greater than zero"):
        severity_score(bbox_area=100.0, y_center=50.0, frame_width=0, frame_height=1000)


def test_estimate_severity_propagates_invalid_frame_area() -> None:
    with pytest.raises(ValueError, match="Frame area must be greater than zero"):
        estimate_severity(
            bbox_area=100.0, y_center=50.0, frame_width=0, frame_height=1000
        )


def test_severity_score_matches_formula() -> None:
    # 0.6 * (200_000 / 1_000_000) + 0.4 * (600 / 1000) = 0.12 + 0.24
    score = severity_score(
        bbox_area=200_000.0,
        y_center=600.0,
        frame_width=1000,
        frame_height=1000,
    )
    assert score == pytest.approx(0.36)


def test_alpha_extremes() -> None:
    vertical_only = severity_score(
        bbox_area=100.0,
        y_center=700.0,
        frame_width=1000,
        frame_height=1000,
        alpha=0.0,
    )
    assert vertical_only == pytest.approx(0.7)

    area_only = severity_score(
        bbox_area=250_000.0,
        y_center=100.0,
        frame_width=1000,
        frame_height=1000,
        alpha=1.0,
    )
    assert area_only == pytest.approx(0.25)


def test_band_boundaries() -> None:
    # bbox_area 0 -> score is 0.4 * (y_center / frame_height)
    assert (
        estimate_severity(
            bbox_area=0.0, y_center=500.0, frame_width=1000, frame_height=1000
        )
        == "Medium"  # score == 0.2, and "< low_threshold" is Low
    )
    assert (
        estimate_severity(
            bbox_area=0.0, y_center=1000.0, frame_width=1000, frame_height=1000
        )
        == "High"  # score == 0.4
    )
    assert (
        estimate_severity(
            bbox_area=0.0, y_center=499.0, frame_width=1000, frame_height=1000
        )
        == "Low"  # score ~= 0.1996
    )
    assert (
        estimate_severity(
            bbox_area=0.0, y_center=999.0, frame_width=1000, frame_height=1000
        )
        == "Medium"  # score ~= 0.3996
    )


def test_full_frame_box_is_high() -> None:
    assert (
        estimate_severity(
            bbox_area=1_000_000.0,
            y_center=500.0,
            frame_width=1000,
            frame_height=1000,
        )
        == "High"  # 0.6 * 1.0 + 0.4 * 0.5 = 0.8
    )


def test_zero_bbox_area_is_valid() -> None:
    assert severity_score(
        bbox_area=0.0, y_center=100.0, frame_width=100, frame_height=100
    ) == pytest.approx(0.4)


def test_custom_parameters_change_banding() -> None:
    kwargs = {
        "bbox_area": 0.0,
        "y_center": 750.0,
        "frame_width": 1000,
        "frame_height": 1000,
    }
    assert estimate_severity(**kwargs) == "Medium"  # score 0.3, default thresholds
    tighter = SeverityParameters(low_threshold=0.05, medium_threshold=0.1)
    assert estimate_severity(**kwargs, params=tighter) == "High"


def test_severity_parameters_frozen() -> None:
    params = SeverityParameters()
    with pytest.raises(FrozenInstanceError):
        params.alpha = 0.9


def test_severity_parameters_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        SeverityParameters(alpha=1.5)
    with pytest.raises(ValueError, match="threshold"):
        SeverityParameters(low_threshold=0.5, medium_threshold=0.3)
