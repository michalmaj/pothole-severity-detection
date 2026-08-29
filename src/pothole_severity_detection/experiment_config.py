"""Typed, validated experiment-configuration schema and loader.

The schema is the single source of truth for the shape of the YAML files in
``configs/experiments/``. It validates structure, types, and cross-field rules,
but performs no filesystem existence checks: resolving and checking dataset and
weight paths stays in the workflow scripts so the schema (and its tests) run in
CI without a local dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised for any experiment-configuration problem."""


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    description: str = ""
    type: str = ""


@dataclass(frozen=True)
class DatasetSettings:
    data_yaml: str
    name: str = ""
    classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSettings:
    source: str
    architecture: str = ""
    initial_weights: str | None = None
    output_weights: str | None = None


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int | None = None
    additional_epochs: int | None = None
    device: str = "auto"
    image_size: int = 416
    batch_size: int = 2
    workers: int = 0
    amp: bool = False
    total_effective_epochs: int | None = None
    scale: float | None = None
    mosaic: float | None = None
    mixup: float | None = None
    copy_paste: float | None = None
    hsv_h: float | None = None
    hsv_s: float | None = None
    hsv_v: float | None = None
    degrees: float | None = None
    translate: float | None = None
    shear: float | None = None
    perspective: float | None = None
    fliplr: float | None = None
    flipud: float | None = None
    close_mosaic: int | None = None
    optimizer: str | None = None


@dataclass(frozen=True)
class EvaluationSettings:
    split: str = "test"
    image_size: int | None = None
    batch_size: int | None = None
    device: str | None = None
    legacy_metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class PredictionSettings:
    source: str
    output_dir: str = "outputs/predictions"
    confidence: float = 0.25
    recursive: bool = False


@dataclass(frozen=True)
class OutputsSettings:
    training_project: str = "runs/train"
    training_name: str | None = None
    evaluation_dir: str | None = None
    exist_ok: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    experiment: ExperimentSettings
    dataset: DatasetSettings
    model: ModelSettings
    training: TrainingSettings
    outputs: OutputsSettings
    evaluation: EvaluationSettings | None = None
    prediction: PredictionSettings | None = None
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #


def _as_str(value: Any, label: str, path: Path) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path}: {label}: expected a string, got {value!r}")
    return value


def _as_int(value: Any, label: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ConfigError(f"{path}: {label}: expected an integer, got {value!r}")
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(
            f"{path}: {label}: expected an integer, got {value!r}"
        ) from exc


def _as_float(value: Any, label: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{path}: {label}: expected a number, got {value!r}")
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{path}: {label}: expected a number, got {value!r}") from exc


def _as_bool(value: Any, label: str, path: Path) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: {label}: expected a boolean, got {value!r}")
    return value


def _as_str_tuple(value: Any, label: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{path}: {label}: expected a list of strings")
    return tuple(value)


def _require_mapping(value: Any, label: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(
            f"{path}: {label}: expected a mapping, got {type(value).__name__}"
        )
    return value


def _check_keys(raw: dict[str, Any], allowed: set[str], label: str, path: Path) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(
            f"{path}: {label}: unknown key(s) {unknown}; "
            f"allowed keys: {sorted(allowed)}"
        )


def _get(
    raw: dict[str, Any],
    key: str,
    coerce: Callable[[Any, str, Path], Any],
    label: str,
    path: Path,
    *,
    default: Any,
    required: bool = False,
) -> Any:
    if key not in raw or raw[key] is None:
        if required:
            raise ConfigError(f"{path}: {label}: missing required key '{key}'")
        return default
    return coerce(raw[key], f"{label}.{key}", path)


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #

_TRAINING_KEYS = {
    "epochs",
    "additional_epochs",
    "device",
    "image_size",
    "batch_size",
    "workers",
    "amp",
    "total_effective_epochs",
    "scale",
    "mosaic",
    "mixup",
    "copy_paste",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
    "close_mosaic",
    "optimizer",
}

_FLOAT_AUGMENTATION_KEYS = (
    "scale",
    "mosaic",
    "mixup",
    "copy_paste",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
)


def _build_experiment(raw: dict[str, Any], path: Path) -> ExperimentSettings:
    _check_keys(raw, {"name", "description", "type"}, "experiment", path)
    return ExperimentSettings(
        name=_get(
            raw, "name", _as_str, "experiment", path, default=None, required=True
        ),
        description=_get(raw, "description", _as_str, "experiment", path, default=""),
        type=_get(raw, "type", _as_str, "experiment", path, default=""),
    )


def _build_dataset(raw: dict[str, Any], path: Path) -> DatasetSettings:
    _check_keys(raw, {"data_yaml", "name", "classes"}, "dataset", path)
    data_yaml = _get(
        raw, "data_yaml", _as_str, "dataset", path, default=None, required=True
    )
    return DatasetSettings(
        data_yaml=data_yaml,
        name=_get(raw, "name", _as_str, "dataset", path, default=""),
        classes=_get(raw, "classes", _as_str_tuple, "dataset", path, default=()),
    )


def _build_model(raw: dict[str, Any], path: Path) -> ModelSettings:
    _check_keys(
        raw,
        {"source", "architecture", "initial_weights", "output_weights"},
        "model",
        path,
    )
    return ModelSettings(
        source=_get(raw, "source", _as_str, "model", path, default=None, required=True),
        architecture=_get(raw, "architecture", _as_str, "model", path, default=""),
        initial_weights=_get(
            raw, "initial_weights", _as_str, "model", path, default=None
        ),
        output_weights=_get(
            raw, "output_weights", _as_str, "model", path, default=None
        ),
    )


def _build_training(raw: dict[str, Any], path: Path) -> TrainingSettings:
    _check_keys(raw, _TRAINING_KEYS, "training", path)

    def num(key: str, coerce: Callable[[Any, str, Path], Any]) -> Any:
        return _get(raw, key, coerce, "training", path, default=None)

    augmentation = {key: num(key, _as_float) for key in _FLOAT_AUGMENTATION_KEYS}

    return TrainingSettings(
        epochs=num("epochs", _as_int),
        additional_epochs=num("additional_epochs", _as_int),
        device=_get(raw, "device", _as_str, "training", path, default="auto"),
        image_size=_get(raw, "image_size", _as_int, "training", path, default=416),
        batch_size=_get(raw, "batch_size", _as_int, "training", path, default=2),
        workers=_get(raw, "workers", _as_int, "training", path, default=0),
        amp=_get(raw, "amp", _as_bool, "training", path, default=False),
        total_effective_epochs=num("total_effective_epochs", _as_int),
        close_mosaic=num("close_mosaic", _as_int),
        optimizer=num("optimizer", _as_str),
        **augmentation,
    )


def _build_evaluation(raw: dict[str, Any], path: Path) -> EvaluationSettings:
    _check_keys(
        raw,
        {"split", "image_size", "batch_size", "device", "metrics"},
        "evaluation",
        path,
    )
    metrics = raw.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        raise ConfigError(
            f"{path}: evaluation.metrics: expected a mapping, "
            f"got {type(metrics).__name__}"
        )
    return EvaluationSettings(
        split=_get(raw, "split", _as_str, "evaluation", path, default="test"),
        image_size=_get(raw, "image_size", _as_int, "evaluation", path, default=None),
        batch_size=_get(raw, "batch_size", _as_int, "evaluation", path, default=None),
        device=_get(raw, "device", _as_str, "evaluation", path, default=None),
        legacy_metrics=metrics,
    )


def _build_prediction(raw: dict[str, Any], path: Path) -> PredictionSettings:
    _check_keys(
        raw,
        {"source", "output_dir", "confidence", "recursive"},
        "prediction",
        path,
    )
    return PredictionSettings(
        source=_get(
            raw,
            "source",
            _as_str,
            "prediction",
            path,
            default=None,
            required=True,
        ),
        output_dir=_get(
            raw,
            "output_dir",
            _as_str,
            "prediction",
            path,
            default="outputs/predictions",
        ),
        confidence=_get(raw, "confidence", _as_float, "prediction", path, default=0.25),
        recursive=_get(raw, "recursive", _as_bool, "prediction", path, default=False),
    )


def _build_outputs(raw: dict[str, Any], path: Path) -> OutputsSettings:
    _check_keys(
        raw,
        {"training_project", "training_name", "evaluation_dir", "exist_ok"},
        "outputs",
        path,
    )
    return OutputsSettings(
        training_project=_get(
            raw,
            "training_project",
            _as_str,
            "outputs",
            path,
            default="runs/train",
        ),
        training_name=_get(
            raw, "training_name", _as_str, "outputs", path, default=None
        ),
        evaluation_dir=_get(
            raw, "evaluation_dir", _as_str, "outputs", path, default=None
        ),
        exist_ok=_get(raw, "exist_ok", _as_bool, "outputs", path, default=True),
    )


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

_KNOWN_SECTIONS = {
    "experiment",
    "dataset",
    "model",
    "training",
    "evaluation",
    "prediction",
    "outputs",
    "notes",
}
_REQUIRED_SECTIONS = ("experiment", "dataset", "model", "training")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load, validate, and return an experiment configuration.

    Raises:
        ConfigError: if the file is missing/unreadable, the YAML is malformed,
            a required section is absent, an unknown section or key is present,
            a value has the wrong type, or a cross-field rule is violated.
    """
    config_path = Path(path)

    if not config_path.is_file():
        raise ConfigError(f"{config_path}: configuration file not found")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")

    unknown_sections = sorted(set(raw) - _KNOWN_SECTIONS)
    if unknown_sections:
        raise ConfigError(
            f"{config_path}: unknown section(s) {unknown_sections}; "
            f"allowed sections: {sorted(_KNOWN_SECTIONS)}"
        )

    for section in _REQUIRED_SECTIONS:
        if raw.get(section) is None:
            raise ConfigError(f"{config_path}: missing required section '{section}'")

    evaluation = None
    if raw.get("evaluation") is not None:
        evaluation = _build_evaluation(
            _require_mapping(raw["evaluation"], "evaluation", config_path),
            config_path,
        )

    prediction = None
    if raw.get("prediction") is not None:
        prediction = _build_prediction(
            _require_mapping(raw["prediction"], "prediction", config_path),
            config_path,
        )

    outputs = OutputsSettings()
    if raw.get("outputs") is not None:
        outputs = _build_outputs(
            _require_mapping(raw["outputs"], "outputs", config_path),
            config_path,
        )

    notes: tuple[str, ...] = ()
    if raw.get("notes") is not None:
        notes = _as_str_tuple(raw["notes"], "notes", config_path)

    return ExperimentConfig(
        path=config_path,
        experiment=_build_experiment(
            _require_mapping(raw["experiment"], "experiment", config_path),
            config_path,
        ),
        dataset=_build_dataset(
            _require_mapping(raw["dataset"], "dataset", config_path),
            config_path,
        ),
        model=_build_model(
            _require_mapping(raw["model"], "model", config_path), config_path
        ),
        training=_build_training(
            _require_mapping(raw["training"], "training", config_path),
            config_path,
        ),
        outputs=outputs,
        evaluation=evaluation,
        prediction=prediction,
        notes=notes,
    )
