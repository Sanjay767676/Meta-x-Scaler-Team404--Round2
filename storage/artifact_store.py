"""Persistent artifact storage helpers."""

from __future__ import annotations

import json
import os
from typing import Any

from config import RUN_ARTIFACTS_DIR, ensure_runtime_dirs


def ensure_artifact_dirs() -> None:
    """Ensure all runtime and artifact directories exist."""
    ensure_runtime_dirs()


def save_run_artifact(filename: str, payload: dict[str, Any]) -> str:
    """Save deterministic JSON artifact for audit and replay."""
    ensure_artifact_dirs()
    path = os.path.join(RUN_ARTIFACTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
