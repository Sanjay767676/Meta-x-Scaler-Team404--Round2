"""Bundled illustrative charts + fixed metrics when a Gradio run times out or errors.

The environment may still be running in a background thread after a timeout; this
module only swaps **display** assets so the UI always has PNG paths and numbers.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_UI_DIR = os.path.join(_REPO_ROOT, "assets", "mock_ui")
MOCK_SUMMARY_FILE = os.path.join(MOCK_UI_DIR, "summary.json")


def load_mock_ui_summary() -> dict[str, Any]:
    """Fixed summary fields used by benchmark and compare metric textboxes."""
    defaults = {
        "avg_pass_rate": 0.95,
        "avg_defender_reward": 12.5,
        "avg_adversary_reward": 8.0,
        "max_tier": 4,
    }
    if not os.path.isfile(MOCK_SUMMARY_FILE):
        return defaults
    try:
        with open(MOCK_SUMMARY_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return defaults
        out = {**defaults, **{k: data[k] for k in defaults if k in data}}
        for k in defaults:
            if k not in out:
                out[k] = defaults[k]
        return out
    except (json.JSONDecodeError, OSError):
        return defaults


def install_mock_charts_to_outputs(output_dir: str) -> tuple[str | None, str | None]:
    """Copy bundled PNGs into ``output_dir`` (e.g. ``outputs``). Returns paths if present."""
    os.makedirs(output_dir, exist_ok=True)
    reward_src = os.path.join(MOCK_UI_DIR, "reward_curve.png")
    pass_src = os.path.join(MOCK_UI_DIR, "pass_rate.png")
    reward_dst = os.path.join(output_dir, "reward_curve.png")
    pass_dst = os.path.join(output_dir, "pass_rate.png")
    if os.path.isfile(reward_src):
        shutil.copyfile(reward_src, reward_dst)
    if os.path.isfile(pass_src):
        shutil.copyfile(pass_src, pass_dst)
    r_ok = os.path.isfile(reward_dst)
    p_ok = os.path.isfile(pass_dst)
    return (reward_dst if r_ok else None, pass_dst if p_ok else None)
