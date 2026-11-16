"""Run metadata saved next to every result: config, git commit, versions, machine.

With this, any number in the thesis can be traced back to the exact code and settings that made it.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

PACKAGES = ("audiomt", "numpy", "scipy", "scikit-learn", "librosa", "torch", "transformers")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_info(config: dict[str, Any], command: str) -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "command": command,
        "started_utc": now(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "config": config,
        "versions": {p: _version(p) for p in PACKAGES},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
