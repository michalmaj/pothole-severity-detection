"""Tests for the experiment configuration schema and loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pothole_severity_detection.experiment_config import (
    ConfigError,
    load_experiment_config,
)

MINIMAL: dict = {
    "experiment": {"name": "unit_test_exp"},
    "dataset": {"data_yaml": "data/pothole_detection_v2/data.yaml"},
    "model": {"source": "yolov12n.yaml"},
    "training": {"epochs": 1},
}


def write_config(directory: Path, data: dict) -> Path:
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_minimal_config_loads_with_defaults(tmp_path: Path) -> None:
    config = load_experiment_config(write_config(tmp_path, MINIMAL))

    assert config.experiment.name == "unit_test_exp"
    assert config.dataset.data_yaml.endswith(".yaml")
    assert config.model.source == "yolov12n.yaml"
    assert config.training.image_size == 416
    assert config.training.batch_size == 2
    assert config.training.device == "auto"
    assert config.outputs.training_project == "runs/train"
    assert config.outputs.exist_ok is True
    assert config.evaluation is None
    assert config.prediction is None
    assert config.notes == ()


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_experiment_config(tmp_path / "nope.yaml")


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_experiment_config(path)


def test_missing_required_section_raises(tmp_path: Path) -> None:
    data = {k: v for k, v in MINIMAL.items() if k != "model"}
    with pytest.raises(ConfigError, match="model"):
        load_experiment_config(write_config(tmp_path, data))


def test_unknown_section_raises(tmp_path: Path) -> None:
    data = MINIMAL | {"telemetry": {"enabled": True}}
    with pytest.raises(ConfigError, match="telemetry"):
        load_experiment_config(write_config(tmp_path, data))


def test_unknown_key_in_section_raises(tmp_path: Path) -> None:
    data = MINIMAL | {"training": {"epochs": 1, "frobnicate": 3}}
    with pytest.raises(ConfigError, match="frobnicate"):
        load_experiment_config(write_config(tmp_path, data))


def test_wrong_type_raises(tmp_path: Path) -> None:
    data = MINIMAL | {"training": {"epochs": "not-a-number"}}
    with pytest.raises(ConfigError, match="training.epochs"):
        load_experiment_config(write_config(tmp_path, data))


def test_explicit_null_optional_is_treated_as_default(tmp_path: Path) -> None:
    data = MINIMAL | {"model": {"source": "yolov12n.yaml", "initial_weights": None}}
    config = load_experiment_config(write_config(tmp_path, data))
    assert config.model.initial_weights is None


def test_optional_sections_are_parsed(tmp_path: Path) -> None:
    data = MINIMAL | {
        "evaluation": {"split": "test", "batch_size": 2},
        "prediction": {"source": "data/pothole_detection_v2/test/images"},
        "outputs": {"training_project": "runs/experiments"},
        "notes": ["first note", "second note"],
    }
    config = load_experiment_config(write_config(tmp_path, data))
    assert config.evaluation.split == "test"
    assert config.evaluation.batch_size == 2
    assert config.prediction.source.endswith("images")
    assert config.prediction.confidence == 0.25
    assert config.outputs.training_project == "runs/experiments"
    assert config.notes == ("first note", "second note")


def test_prediction_section_requires_source(tmp_path: Path) -> None:
    data = MINIMAL | {"prediction": {"confidence": 0.3}}
    with pytest.raises(ConfigError, match="prediction.*source"):
        load_experiment_config(write_config(tmp_path, data))


def test_augmentation_fields_are_parsed(tmp_path: Path) -> None:
    data = MINIMAL | {
        "training": {"additional_epochs": 60, "scale": 0.25, "mosaic": 0.2}
    }
    config = load_experiment_config(write_config(tmp_path, data))
    assert config.training.scale == 0.25
    assert config.training.mosaic == 0.2
    assert config.training.mixup is None


def test_both_epochs_keys_raises(tmp_path: Path) -> None:
    data = MINIMAL | {"training": {"epochs": 40, "additional_epochs": 60}}
    with pytest.raises(ConfigError, match="epochs"):
        load_experiment_config(write_config(tmp_path, data))


def test_no_epochs_key_raises(tmp_path: Path) -> None:
    data = MINIMAL | {"training": {"image_size": 416}}
    with pytest.raises(ConfigError, match="epochs"):
        load_experiment_config(write_config(tmp_path, data))


def test_data_yaml_must_end_with_yaml(tmp_path: Path) -> None:
    data = MINIMAL | {"dataset": {"data_yaml": "data/pothole/data.json"}}
    with pytest.raises(ConfigError, match="data_yaml"):
        load_experiment_config(write_config(tmp_path, data))


def test_evaluation_split_is_validated(tmp_path: Path) -> None:
    data = MINIMAL | {"evaluation": {"split": "holdout"}}
    with pytest.raises(ConfigError, match="split"):
        load_experiment_config(write_config(tmp_path, data))


def test_epochs_to_run_and_is_finetuning(tmp_path: Path) -> None:
    fresh = load_experiment_config(write_config(tmp_path, MINIMAL))
    assert fresh.training.epochs_to_run == 1
    assert fresh.training.is_finetuning is False

    data = MINIMAL | {"training": {"additional_epochs": 60}}
    tuned = load_experiment_config(write_config(tmp_path, data))
    assert tuned.training.epochs_to_run == 60
    assert tuned.training.is_finetuning is True


def test_augmentation_overrides_only_includes_set_keys(tmp_path: Path) -> None:
    data = MINIMAL | {
        "training": {
            "additional_epochs": 60,
            "scale": 0.25,
            "mosaic": 0.2,
            "mixup": 0.0,
        }
    }
    config = load_experiment_config(write_config(tmp_path, data))
    assert config.training.augmentation_overrides() == {
        "scale": 0.25,
        "mosaic": 0.2,
        "mixup": 0.0,
    }


def test_training_name_falls_back_to_experiment_name(tmp_path: Path) -> None:
    fresh = load_experiment_config(write_config(tmp_path, MINIMAL))
    assert fresh.training_name == "unit_test_exp"

    data = MINIMAL | {"outputs": {"training_name": "explicit_name"}}
    named = load_experiment_config(write_config(tmp_path, data))
    assert named.training_name == "explicit_name"


def test_seed_and_deterministic(tmp_path: Path) -> None:
    default = load_experiment_config(write_config(tmp_path, MINIMAL))
    assert default.training.seed == 0
    assert default.training.deterministic is True

    data = MINIMAL | {"training": {"epochs": 1, "seed": 7, "deterministic": False}}
    custom = load_experiment_config(write_config(tmp_path, data))
    assert custom.training.seed == 7
    assert custom.training.deterministic is False


def test_evaluation_metrics_key_is_rejected(tmp_path: Path) -> None:
    data = MINIMAL | {"evaluation": {"split": "test", "metrics": {"precision": 0.8}}}
    with pytest.raises(ConfigError, match="metrics"):
        load_experiment_config(write_config(tmp_path, data))


def test_config_is_immutable(tmp_path: Path) -> None:
    from dataclasses import FrozenInstanceError

    config = load_experiment_config(write_config(tmp_path, MINIMAL))
    with pytest.raises(FrozenInstanceError):
        config.training.image_size = 999


REPO_CONFIG_DIR = Path("configs/experiments")
REPO_CONFIGS = sorted(REPO_CONFIG_DIR.glob("*.yaml"))


def test_repo_config_dir_is_populated() -> None:
    assert REPO_CONFIGS, f"no configs found in {REPO_CONFIG_DIR}"


@pytest.mark.parametrize("config_path", REPO_CONFIGS, ids=lambda p: p.name)
def test_repo_config_loads(config_path: Path) -> None:
    load_experiment_config(config_path)
