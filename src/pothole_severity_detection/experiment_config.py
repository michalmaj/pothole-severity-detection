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
    seed: int = 0
    deterministic: bool = True
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

    @property
    def epochs_to_run(self) -> int:
        if self.epochs is not None:
            return self.epochs
        if self.additional_epochs is not None:
            return self.additional_epochs
        raise ConfigError("training: neither 'epochs' nor 'additional_epochs' is set")

    @property
    def is_finetuning(self) -> bool:
        return self.additional_epochs is not None

    def augmentation_overrides(self) -> dict[str, float | int | str]:
        keys = (
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
        )
        return {
            key: getattr(self, key) for key in keys if getattr(self, key) is not None
        }


@dataclass(frozen=True)
class EvaluationSettings:
    split: str = "test"
    image_size: int | None = None
    batch_size: int | None = None
    device: str | None = None


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

    @property
    def training_name(self) -> str:
        return self.outputs.training_name or self.experiment.name


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


class _Reader:
    """Read and coerce keys from one config section with consistent errors."""

    def __init__(self, raw: dict[str, Any], label: str, path: Path) -> None:
        self._raw = raw
        self._label = label
        self._path = path

    def check_keys(self, allowed: set[str]) -> None:
        _check_keys(self._raw, allowed, self._label, self._path)

    def _value(
        self,
        key: str,
        coerce: Callable[[Any, str, Path], Any],
        default: Any,
        required: bool,
    ) -> Any:
        if key not in self._raw or self._raw[key] is None:
            if required:
                raise ConfigError(
                    f"{self._path}: {self._label}: missing required key '{key}'"
                )
            return default
        return coerce(self._raw[key], f"{self._label}.{key}", self._path)

    def str_(self, key: str, default: Any = None, *, required: bool = False) -> Any:
        return self._value(key, _as_str, default, required)

    def int_(self, key: str, default: Any = None) -> Any:
        return self._value(key, _as_int, default, False)

    def float_(self, key: str, default: Any = None) -> Any:
        return self._value(key, _as_float, default, False)

    def bool_(self, key: str, default: bool = False) -> bool:
        return self._value(key, _as_bool, default, False)

    def str_tuple(self, key: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
        return self._value(key, _as_str_tuple, default, False)


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
    "seed",
    "deterministic",
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
    r = _Reader(raw, "experiment", path)
    r.check_keys({"name", "description", "type"})
    return ExperimentSettings(
        name=r.str_("name", required=True),
        description=r.str_("description", ""),
        type=r.str_("type", ""),
    )


def _build_dataset(raw: dict[str, Any], path: Path) -> DatasetSettings:
    r = _Reader(raw, "dataset", path)
    r.check_keys({"data_yaml", "name", "classes"})
    data_yaml = r.str_("data_yaml", required=True)
    if not data_yaml.endswith(".yaml"):
        raise ConfigError(
            f"{path}: dataset.data_yaml: must end with '.yaml', got {data_yaml!r}"
        )
    return DatasetSettings(
        data_yaml=data_yaml,
        name=r.str_("name", ""),
        classes=r.str_tuple("classes"),
    )


def _build_model(raw: dict[str, Any], path: Path) -> ModelSettings:
    r = _Reader(raw, "model", path)
    r.check_keys({"source", "architecture", "initial_weights", "output_weights"})
    return ModelSettings(
        source=r.str_("source", required=True),
        architecture=r.str_("architecture", ""),
        initial_weights=r.str_("initial_weights"),
        output_weights=r.str_("output_weights"),
    )


def _build_training(raw: dict[str, Any], path: Path) -> TrainingSettings:
    r = _Reader(raw, "training", path)
    r.check_keys(_TRAINING_KEYS)
    augmentation = {key: r.float_(key) for key in _FLOAT_AUGMENTATION_KEYS}
    return TrainingSettings(
        epochs=r.int_("epochs"),
        additional_epochs=r.int_("additional_epochs"),
        device=r.str_("device", "auto"),
        image_size=r.int_("image_size", 416),
        batch_size=r.int_("batch_size", 2),
        workers=r.int_("workers", 0),
        amp=r.bool_("amp"),
        seed=r.int_("seed", 0),
        deterministic=r.bool_("deterministic", True),
        total_effective_epochs=r.int_("total_effective_epochs"),
        close_mosaic=r.int_("close_mosaic"),
        optimizer=r.str_("optimizer"),
        **augmentation,
    )


def _build_evaluation(raw: dict[str, Any], path: Path) -> EvaluationSettings:
    r = _Reader(raw, "evaluation", path)
    r.check_keys({"split", "image_size", "batch_size", "device"})
    split = r.str_("split", "test")
    if split not in {"train", "val", "test"}:
        raise ConfigError(
            f"{path}: evaluation.split: must be 'train', 'val', or 'test', "
            f"got {split!r}"
        )
    return EvaluationSettings(
        split=split,
        image_size=r.int_("image_size"),
        batch_size=r.int_("batch_size"),
        device=r.str_("device"),
    )


def _build_prediction(raw: dict[str, Any], path: Path) -> PredictionSettings:
    r = _Reader(raw, "prediction", path)
    r.check_keys({"source", "output_dir", "confidence", "recursive"})
    return PredictionSettings(
        source=r.str_("source", required=True),
        output_dir=r.str_("output_dir", "outputs/predictions"),
        confidence=r.float_("confidence", 0.25),
        recursive=r.bool_("recursive"),
    )


def _build_outputs(raw: dict[str, Any], path: Path) -> OutputsSettings:
    r = _Reader(raw, "outputs", path)
    r.check_keys({"training_project", "training_name", "evaluation_dir", "exist_ok"})
    return OutputsSettings(
        training_project=r.str_("training_project", "runs/train"),
        training_name=r.str_("training_name"),
        evaluation_dir=r.str_("evaluation_dir"),
        exist_ok=r.bool_("exist_ok", True),
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


def _validate_training_epochs(training: TrainingSettings, path: Path) -> None:
    has_epochs = training.epochs is not None
    has_additional = training.additional_epochs is not None
    if has_epochs and has_additional:
        raise ConfigError(
            f"{path}: training: set only one of 'epochs' or "
            f"'additional_epochs', not both"
        )
    if not has_epochs and not has_additional:
        raise ConfigError(
            f"{path}: training: set exactly one of 'epochs' or 'additional_epochs'"
        )


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

    training = _build_training(
        _require_mapping(raw["training"], "training", config_path),
        config_path,
    )
    _validate_training_epochs(training, config_path)

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
        training=training,
        outputs=outputs,
        evaluation=evaluation,
        prediction=prediction,
        notes=notes,
    )
