"""Severity estimation: a transparent post-processing heuristic.

The dataset has pothole bounding boxes only -- no ground-truth severity labels --
so severity is not learned. It is estimated after detection from bounding-box
geometry. See docs/severity-heuristic.md for the derivation, the parameter
choices, and the limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

# Heuristic parameters. These are chosen defaults, NOT values calibrated against
# labelled severity data (there is none). See docs/severity-heuristic.md.
#
#   alpha  - weight on normalized bounding-box area vs. vertical position.
#            0.6 slightly favours apparent size over apparent proximity.
#   low / medium score thresholds - chosen so all three bands are populated on
#            typical forward-facing road images. No claim of optimality.
DEFAULT_ALPHA = 0.6
DEFAULT_LOW_THRESHOLD = 0.2
DEFAULT_MEDIUM_THRESHOLD = 0.4

SEVERITY_LOW = "Low"
SEVERITY_MEDIUM = "Medium"
SEVERITY_HIGH = "High"
SEVERITY_LEVELS = (SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH)


@dataclass(frozen=True)
class SeverityParameters:
    """Tunable parameters of the severity heuristic.

    Changing these is a research decision -- see docs/severity-heuristic.md.
    """

    alpha: float = DEFAULT_ALPHA
    low_threshold: float = DEFAULT_LOW_THRESHOLD
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not 0.0 <= self.low_threshold <= self.medium_threshold:
            raise ValueError(
                "thresholds must satisfy "
                "0 <= low_threshold <= medium_threshold, got "
                f"low={self.low_threshold}, medium={self.medium_threshold}"
            )


DEFAULT_SEVERITY_PARAMETERS = SeverityParameters()


def severity_score(
    bbox_area: float,
    y_center: float,
    frame_width: int,
    frame_height: int,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Return the raw severity score.

    score = alpha * (bbox_area / frame_area)
          + (1 - alpha) * (y_center / frame_height)

    The first term is the fraction of the frame the box covers (apparent size).
    The second is how far down the frame the box sits (apparent proximity for a
    roughly forward-facing camera).
    """
    frame_area = frame_width * frame_height

    if frame_area <= 0:
        raise ValueError("Frame area must be greater than zero.")

    normalized_area = bbox_area / frame_area
    vertical_position = y_center / frame_height
    return alpha * normalized_area + (1.0 - alpha) * vertical_position


def estimate_severity(
    bbox_area: float,
    y_center: float,
    frame_width: int,
    frame_height: int,
    params: SeverityParameters = DEFAULT_SEVERITY_PARAMETERS,
) -> str:
    """Band the severity score into "Low", "Medium", or "High"."""
    score = severity_score(
        bbox_area=bbox_area,
        y_center=y_center,
        frame_width=frame_width,
        frame_height=frame_height,
        alpha=params.alpha,
    )

    if score < params.low_threshold:
        return SEVERITY_LOW

    if score < params.medium_threshold:
        return SEVERITY_MEDIUM

    return SEVERITY_HIGH
