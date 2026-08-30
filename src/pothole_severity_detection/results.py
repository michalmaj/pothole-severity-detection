"""Committed experiment result records under docs/results/."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pothole_severity_detection import provenance

NOT_RECORDED = "not recorded"

_METRIC_KEYS = ("precision", "recall", "map50", "map75", "map50_95")
_COMMON_KEYS = (
    "experiment",
    "kind",
    "timestamp_utc",
    "git_commit",
    "config",
    "dataset",
    "device",
    "image_size",
    "batch_size",
    "seed",
    "deterministic",
    "versions",
)
_EVAL_KEYS = ("split", "model", "sample_counts", "metrics")
_TRAIN_KEYS = (
    "model_source",
    "output_weights",
    "epochs",
    "is_finetuning",
    "augmentation",
    "duration_seconds",
    "validation_metrics",
)
_BOX_ATTRS = {
    "precision": "mp",
    "recall": "mr",
    "map50": "map50",
    "map75": "map75",
    "map50_95": "map",
}


class ResultError(Exception):
    """Raised for a malformed result record."""


def extract_box_metrics(ultralytics_results: Any) -> dict[str, float | str]:
    """Read the box metrics from an Ultralytics results object."""
    box = getattr(ultralytics_results, "box", None)
    metrics: dict[str, float | str] = {}
    for name, attr in _BOX_ATTRS.items():
        value = getattr(box, attr, None)
        metrics[name] = float(value) if value is not None else NOT_RECORDED
    return metrics


def _common_block(
    *,
    experiment: str,
    kind: str,
    device: str,
    image_size: int,
    batch_size: int,
    seed: int,
    deterministic: bool,
    config: str,
    dataset: str,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "kind": kind,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": provenance.git_commit(),
        "config": config,
        "dataset": dataset,
        "device": device,
        "image_size": image_size,
        "batch_size": batch_size,
        "seed": seed,
        "deterministic": deterministic,
        "versions": provenance.package_versions(),
    }


def _dump(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(record, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def write_eval_result(
    path: str | Path,
    *,
    experiment: str,
    config: str,
    dataset: str,
    split: str,
    model: str,
    device: str,
    image_size: int,
    batch_size: int,
    seed: int,
    deterministic: bool,
    metrics: dict[str, float | str],
    sample_counts: dict[str, int | str],
    provenance_note: str | None = None,
) -> None:
    """Write a `kind: eval` result record."""
    record = _common_block(
        experiment=experiment,
        kind="eval",
        device=device,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        deterministic=deterministic,
        config=config,
        dataset=dataset,
    )
    record["split"] = split
    record["model"] = model
    record["sample_counts"] = sample_counts
    record["metrics"] = metrics
    if provenance_note is not None:
        record["provenance_note"] = provenance_note
    _dump(record, Path(path))


def write_train_result(
    path: str | Path,
    *,
    experiment: str,
    config: str,
    dataset: str,
    model_source: str,
    output_weights: str,
    device: str,
    image_size: int,
    batch_size: int,
    seed: int,
    deterministic: bool,
    epochs: int,
    is_finetuning: bool,
    augmentation: dict[str, Any],
    duration_seconds: float | str,
    validation_metrics: dict[str, float | str],
    provenance_note: str | None = None,
) -> None:
    """Write a `kind: train` result record."""
    record = _common_block(
        experiment=experiment,
        kind="train",
        device=device,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        deterministic=deterministic,
        config=config,
        dataset=dataset,
    )
    record["model_source"] = model_source
    record["output_weights"] = output_weights
    record["epochs"] = epochs
    record["is_finetuning"] = is_finetuning
    record["augmentation"] = augmentation
    record["duration_seconds"] = duration_seconds
    record["validation_metrics"] = validation_metrics
    if provenance_note is not None:
        record["provenance_note"] = provenance_note
    _dump(record, Path(path))


def _is_number_or_not_recorded(value: Any) -> bool:
    if value == NOT_RECORDED:
        return True
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_result_file(path: str | Path) -> None:
    """Raise ResultError if the record at `path` is malformed."""
    record_path = Path(path)
    if not record_path.is_file():
        raise ResultError(f"{record_path}: result file not found")

    data = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ResultError(f"{record_path}: result must be a mapping")

    kind = data.get("kind")
    if kind not in ("train", "eval"):
        raise ResultError(
            f"{record_path}: 'kind' must be 'train' or 'eval', got {kind!r}"
        )

    required = _COMMON_KEYS + (_EVAL_KEYS if kind == "eval" else _TRAIN_KEYS)
    missing = [key for key in required if key not in data]
    if missing:
        raise ResultError(f"{record_path}: missing key(s): {missing}")

    metrics_key = "metrics" if kind == "eval" else "validation_metrics"
    metrics = data[metrics_key]
    if not isinstance(metrics, dict):
        raise ResultError(f"{record_path}: '{metrics_key}' must be a mapping")

    missing_metrics = [key for key in _METRIC_KEYS if key not in metrics]
    if missing_metrics:
        raise ResultError(f"{record_path}: {metrics_key} missing: {missing_metrics}")
    for name, value in metrics.items():
        if not _is_number_or_not_recorded(value):
            raise ResultError(
                f"{record_path}: {metrics_key}.{name} must be a number or "
                f"'{NOT_RECORDED}', got {value!r}"
            )


def load_result(path: str | Path) -> dict[str, Any]:
    """Validate and return the record at `path`."""
    validate_result_file(path)
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
