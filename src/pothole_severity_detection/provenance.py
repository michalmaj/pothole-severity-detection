"""Environment provenance for experiment result records."""

from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path

_NOT_AVAILABLE = "not available"
_MODULE_DIR = Path(__file__).resolve().parent


def _run_git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=_MODULE_DIR,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def repo_root() -> Path:
    """Return the repository root, falling back to a `.git` walk then cwd."""
    top_level = _run_git("rev-parse", "--show-toplevel")
    if top_level:
        return Path(top_level)
    for parent in (_MODULE_DIR, *_MODULE_DIR.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def git_commit() -> str:
    """Return the current commit sha, suffixed `-dirty` for an unclean tree."""
    sha = _run_git("rev-parse", "HEAD")
    if not sha:
        return _NOT_AVAILABLE
    status = _run_git("status", "--porcelain")
    return f"{sha}-dirty" if status else sha


def ultralytics_commit() -> str:
    """Return the pinned commit of the sunsmarterjie/yolov12 fork from uv.lock."""
    try:
        text = (repo_root() / "uv.lock").read_text(encoding="utf-8")
    except OSError:
        return _NOT_AVAILABLE
    match = re.search(r"sunsmarterjie/yolov12\.git#([0-9a-f]{40})", text)
    return match.group(1) if match else _NOT_AVAILABLE


def package_versions() -> dict[str, str]:
    """Return the versions relevant to reproducing a run."""
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "ultralytics": str(ultralytics.__version__),
        "ultralytics_commit": ultralytics_commit(),
    }


def to_repo_relative(path: str | Path) -> str:
    """Return `path` relative to the repo root, or its absolute form if outside."""
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root().resolve()))
    except ValueError:
        return str(resolved)
