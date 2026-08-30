"""Tests for the docs/reports/ structure."""

from __future__ import annotations

from pathlib import Path

import pytest

REPORTS_DIR = Path("docs/reports")
REQUIRED_HEADINGS = (
    "Objective",
    "Baseline",
    "Experimental setup",
    "Change under test",
    "Results",
    "Observed facts",
    "Interpretation",
    "Limitations",
    "Decision",
)

_REPORT_FILES = sorted(
    path for path in REPORTS_DIR.glob("*.md") if path.name != "README.md"
)


def _headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.lstrip().startswith("#")]


def test_reports_dir_has_template_and_readme() -> None:
    assert (REPORTS_DIR / "TEMPLATE.md").is_file()
    assert (REPORTS_DIR / "README.md").is_file()


@pytest.mark.parametrize("report_path", _REPORT_FILES, ids=lambda p: p.name)
def test_report_has_required_headings(report_path: Path) -> None:
    headings = _headings(report_path.read_text(encoding="utf-8"))
    for required in REQUIRED_HEADINGS:
        assert any(required in heading for heading in headings), (
            f"{report_path.name} is missing a heading containing '{required}'"
        )
