"""Inspect an experiment configuration file: load, validate, and print it.

This is the canonical check that a YAML file in ``configs/experiments/``
conforms to the experiment configuration schema.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from pothole_severity_detection.experiment_config import (
    ConfigError,
    load_experiment_config,
)


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


def print_section(name: str, section: object | None) -> None:
    """Print one configuration section, or a placeholder when it is absent."""
    print(f"{name}:")

    if section is None:
        print("  (not set)")
        return

    for field in dataclasses.fields(section):
        print(f"  {field.name}: {getattr(section, field.name)}")


def main() -> None:
    """Load, validate, and print experiment configuration details."""
    args = parse_args()

    try:
        config = load_experiment_config(args.config)
    except ConfigError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Config path: {config.path}")
    print(f"Resolved training name: {config.training_name}")
    print()

    print_section("experiment", config.experiment)
    print_section("dataset", config.dataset)
    print_section("model", config.model)
    print_section("training", config.training)
    print_section("evaluation", config.evaluation)
    print_section("prediction", config.prediction)
    print_section("outputs", config.outputs)

    if config.notes:
        print("notes:")
        for note in config.notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
