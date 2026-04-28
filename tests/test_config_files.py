"""Tests for experiment configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path("configs/experiments")

REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "dataset",
    "model",
    "training",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert isinstance(config, dict)
    return config


def test_experiment_config_directory_exists() -> None:
    assert CONFIG_DIR.exists()
    assert CONFIG_DIR.is_dir()


def test_all_experiment_configs_have_required_sections() -> None:
    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))

    assert config_paths, "No experiment configuration files found."

    for config_path in config_paths:
        config = load_config(config_path)
        missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(config)

        assert not missing_keys, (
            f"{config_path} is missing required sections: {sorted(missing_keys)}"
        )


def test_experiment_configs_define_dataset_yaml() -> None:
    for config_path in CONFIG_DIR.glob("*.yaml"):
        config = load_config(config_path)
        dataset = config["dataset"]

        assert "data_yaml" in dataset
        assert dataset["data_yaml"].endswith(".yaml")


def test_prediction_configs_define_output_directory_when_present() -> None:
    for config_path in CONFIG_DIR.glob("*.yaml"):
        config = load_config(config_path)
        prediction = config.get("prediction")

        if prediction is None:
            continue

        assert "source" in prediction
        assert "output_dir" in prediction
        assert "confidence" in prediction
