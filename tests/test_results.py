"""Tests for the docs/results/ record format."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pothole_severity_detection import results
from pothole_severity_detection.results import (
    NOT_RECORDED,
    ResultError,
    load_result,
    validate_result_file,
)

_METRICS = {
    "precision": 0.8,
    "recall": 0.7,
    "map50": 0.78,
    "map75": 0.5,
    "map50_95": 0.45,
}


def _eval_kwargs() -> dict:
    return {
        "experiment": "unit_eval",
        "config": "configs/experiments/x.yaml",
        "dataset": "data/pothole_detection_v2/data.yaml",
        "split": "test",
        "model": "weights/local/x.pt",
        "device": "cpu",
        "image_size": 416,
        "batch_size": 2,
        "seed": 0,
        "deterministic": True,
        "metrics": dict(_METRICS),
        "sample_counts": {"images": 149, "instances": NOT_RECORDED},
    }


def _train_kwargs() -> dict:
    return {
        "experiment": "unit_train",
        "config": "configs/experiments/x.yaml",
        "dataset": "data/pothole_detection_v2/data.yaml",
        "model_source": "yolov12n.yaml",
        "output_weights": "weights/local/x.pt",
        "device": "cpu",
        "image_size": 416,
        "batch_size": 2,
        "seed": 0,
        "deterministic": True,
        "epochs": 40,
        "is_finetuning": False,
        "augmentation": {},
        "duration_seconds": 123.4,
        "validation_metrics": dict(_METRICS),
    }


def test_write_and_load_eval_record(tmp_path: Path) -> None:
    path = tmp_path / "unit_eval.eval.yaml"
    results.write_eval_result(path, **_eval_kwargs())
    record = load_result(path)
    assert record["kind"] == "eval"
    assert record["metrics"]["map50_95"] == 0.45
    assert "python" in record["versions"]


def test_write_and_load_train_record(tmp_path: Path) -> None:
    path = tmp_path / "unit_train.train.yaml"
    results.write_train_result(path, **_train_kwargs())
    record = load_result(path)
    assert record["kind"] == "train"
    assert record["duration_seconds"] == pytest.approx(123.4)


def test_validate_rejects_missing_kind(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"experiment": "x"}), encoding="utf-8")
    with pytest.raises(ResultError, match="kind"):
        validate_result_file(path)


def test_validate_rejects_unknown_kind(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"kind": "bogus"}), encoding="utf-8")
    with pytest.raises(ResultError, match="kind"):
        validate_result_file(path)


def test_validate_rejects_missing_required_key(tmp_path: Path) -> None:
    path = tmp_path / "x.eval.yaml"
    results.write_eval_result(path, **_eval_kwargs())
    data = yaml.safe_load(path.read_text())
    del data["device"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ResultError, match="device"):
        validate_result_file(path)


def test_validate_rejects_nonnumeric_metric(tmp_path: Path) -> None:
    path = tmp_path / "x.eval.yaml"
    kwargs = _eval_kwargs()
    kwargs["metrics"]["map50"] = "high"
    results.write_eval_result(path, **kwargs)
    with pytest.raises(ResultError, match="map50"):
        validate_result_file(path)


def test_validate_accepts_not_recorded_provenance(tmp_path: Path) -> None:
    path = tmp_path / "x.eval.yaml"
    results.write_eval_result(path, **_eval_kwargs())
    data = yaml.safe_load(path.read_text())
    data["git_commit"] = NOT_RECORDED
    data["versions"] = dict.fromkeys(data["versions"], NOT_RECORDED)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    validate_result_file(path)


def test_extract_box_metrics_handles_missing_attrs() -> None:
    class _Box:
        mp = 0.8
        mr = 0.7
        map50 = 0.78

    class _Results:
        box = _Box()

    metrics = results.extract_box_metrics(_Results())
    assert metrics["precision"] == 0.8
    assert metrics["map75"] == NOT_RECORDED
    assert metrics["map50_95"] == NOT_RECORDED


REPO_RESULTS = sorted(Path("docs/results").glob("*.yaml"))


@pytest.mark.parametrize("record_path", REPO_RESULTS, ids=lambda p: p.name)
def test_repo_result_files_validate(record_path: Path) -> None:
    validate_result_file(record_path)
