"""Tests for media path utility functions."""

from __future__ import annotations

from pathlib import Path

from pothole_severity_detection.inference.detector import (
    is_image_file,
    is_video_file,
)


def test_is_image_file_accepts_supported_image_extensions() -> None:
    assert is_image_file(Path("sample.jpg"))
    assert is_image_file(Path("sample.jpeg"))
    assert is_image_file(Path("sample.png"))
    assert is_image_file(Path("sample.bmp"))
    assert is_image_file(Path("sample.webp"))


def test_is_image_file_rejects_non_image_extensions() -> None:
    assert not is_image_file(Path("sample.txt"))
    assert not is_image_file(Path("sample.mp4"))


def test_is_video_file_accepts_supported_video_extensions() -> None:
    assert is_video_file(Path("sample.mp4"))
    assert is_video_file(Path("sample.avi"))
    assert is_video_file(Path("sample.mov"))
    assert is_video_file(Path("sample.mkv"))


def test_is_video_file_rejects_non_video_extensions() -> None:
    assert not is_video_file(Path("sample.txt"))
    assert not is_video_file(Path("sample.jpg"))
