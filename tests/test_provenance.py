"""Tests for environment provenance helpers."""

from __future__ import annotations

import re
from pathlib import Path

from pothole_severity_detection import provenance


def test_repo_root_contains_this_package() -> None:
    root = provenance.repo_root()
    assert (root / "src" / "pothole_severity_detection" / "provenance.py").is_file()


def test_git_commit_is_nonempty_and_plausible() -> None:
    value = provenance.git_commit()
    assert value
    assert value == "not available" or re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?", value)


def test_package_versions_has_core_keys() -> None:
    versions = provenance.package_versions()
    for key in ("python", "torch", "ultralytics", "ultralytics_commit"):
        assert key in versions
        assert versions[key]


def test_ultralytics_commit_is_hex_or_not_available() -> None:
    value = provenance.ultralytics_commit()
    assert value == "not available" or re.fullmatch(r"[0-9a-f]{40}", value)


def test_to_repo_relative_strips_root_for_inside_paths() -> None:
    inside = provenance.repo_root() / "configs" / "README.md"
    assert provenance.to_repo_relative(inside) == "configs/README.md"


def test_to_repo_relative_passes_through_outside_paths() -> None:
    result = provenance.to_repo_relative("/etc/hosts")
    assert Path(result).is_absolute()
