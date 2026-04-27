"""Inspect an experiment configuration file.

This script loads a YAML experiment configuration and prints the most important
training, evaluation, and output settings. It is a lightweight first step toward
a config-driven ML workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "dataset",
    "model",
    "training",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect a YAML experiment configuration file."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment configuration file.",
    )

    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required top-level configuration keys."""
    missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(config)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing required top-level configuration keys: {missing}")


def get_nested(
    config: dict[str, Any], section: str, key: str, default: Any = None
) -> Any:
    """Read a nested configuration value safely."""
    section_data = config.get(section, {})

    if not isinstance(section_data, dict):
        return default

    return section_data.get(key, default)


def main() -> None:
    """Load, validate, and print experiment configuration details."""
    args = parse_args()
    config_path = args.config.resolve()

    config = load_yaml_config(config_path)
    validate_config(config)

    experiment_name = get_nested(config, "experiment", "name", "unknown")
    experiment_type = get_nested(config, "experiment", "type", "unknown")
    dataset_yaml = get_nested(config, "dataset", "data_yaml", "unknown")
    model_source = get_nested(config, "model", "source", "unknown")

    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    outputs = config.get("outputs", {})

    print("Experiment configuration loaded successfully.")
    print()
    print(f"Config path: {config_path}")
    print(f"Experiment name: {experiment_name}")
    print(f"Experiment type: {experiment_type}")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Model source: {model_source}")
    print()
    print("Training settings:")
    for key, value in training.items():
        print(f"  {key}: {value}")

    if evaluation:
        print()
        print("Evaluation settings:")
        for key, value in evaluation.items():
            print(f"  {key}: {value}")

    if outputs:
        print()
        print("Output settings:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
