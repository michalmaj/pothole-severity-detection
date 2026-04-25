"""Severity estimation utilities.

The current dataset contains only one object detection class: pothole.
Therefore, severity is estimated as a post-processing heuristic based on
bounding box size and vertical position in the image.
"""

from __future__ import annotations


def estimate_severity(
    bbox_area: float,
    y_center: float,
    frame_width: int,
    frame_height: int,
    alpha: float = 0.6,
) -> str:
    """Estimate pothole severity from bounding box geometry.

    Args:
        bbox_area: Area of the detected bounding box in pixels.
        y_center: Vertical center of the bounding box in pixels.
        frame_width: Width of the image frame in pixels.
        frame_height: Height of the image frame in pixels.
        alpha: Weight assigned to normalized bounding box area.

    Returns:
        Severity label: "Low", "Medium", or "High".
    """
    frame_area = frame_width * frame_height

    if frame_area <= 0:
        raise ValueError("Frame area must be greater than zero.")

    normalized_area = bbox_area / frame_area
    vertical_position = y_center / frame_height

    score = alpha * normalized_area + (1.0 - alpha) * vertical_position

    if score < 0.2:
        return "Low"

    if score < 0.4:
        return "Medium"

    return "High"
