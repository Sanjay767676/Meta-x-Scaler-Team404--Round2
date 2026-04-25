# logger.py
# Metrics logging for FORGE-v4.
# Writes structured logs to logs/rewards.json, logs/episodes.csv, logs/summary.json.

import csv
import json
import os
from datetime import datetime
from typing import Any

from config import LOG_REWARDS_FILE, LOG_EPISODES_FILE, LOG_SUMMARY_FILE, LOG_DIR


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _load_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return default


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ──────────────────────────────────────────────
# Step-level logging
# ──────────────────────────────────────────────

def log_step(
    episode: int,
    step: int,
    coder_version: str,
    breaker_tier: int,
    coder_reward: float,
    breaker_reward: float,
    pass_rate: float,
    fail_count: int,
    error_count: int,
    timeout_count: int,
    break_rate: float,
) -> None:
    """
    Append one step's metrics to logs/rewards.json.

    Args:
        episode:        Episode index.
        step:           Step index within the episode.
        coder_version:  Name of the coder strategy used.
        breaker_tier:   Current breaker tier number.
        coder_reward:   Total coder reward this step.
        breaker_reward: Total breaker reward this step.
        pass_rate:      Fraction of hidden tests passed.
        fail_count:     Number of failing tests.
        error_count:    Number of error/timeout tests.
        timeout_count:  Number of sandbox timeouts specifically.
        break_rate:     Fraction of breaker tests that broke the coder.
    """
    _ensure_log_dir()
    records: list[dict[str, Any]] = _load_json(LOG_REWARDS_FILE, [])

    record = {
        "timestamp":      datetime.utcnow().isoformat(),
        "episode":        episode,
        "step":           step,
        "coder_version":  coder_version,
        "breaker_tier":   breaker_tier,
        "coder_reward":   coder_reward,
        "breaker_reward": breaker_reward,
        "pass_rate":      pass_rate,
        "fail_count":     fail_count,
        "error_count":    error_count,
        "timeout_count":  timeout_count,
        "break_rate":     break_rate,
    }
    records.append(record)
    _write_json(LOG_REWARDS_FILE, records)


# ──────────────────────────────────────────────
# Episode-level logging
# ──────────────────────────────────────────────

# CSV column order
_CSV_FIELDS = [
    "timestamp", "episode", "coder_version", "breaker_tier",
    "avg_coder_reward", "avg_breaker_reward",
    "avg_pass_rate", "total_fail_count", "total_error_count",
    "total_timeout_count", "avg_break_rate", "steps",
]


def log_episode(
    episode: int,
    coder_version: str,
    breaker_tier: int,
    avg_coder_reward: float,
    avg_breaker_reward: float,
    avg_pass_rate: float,
    total_fail_count: int,
    total_error_count: int,
    total_timeout_count: int,
    avg_break_rate: float,
    steps: int,
) -> None:
    """
    Append one episode summary row to logs/episodes.csv.
    """
    _ensure_log_dir()
    file_exists = os.path.exists(LOG_EPISODES_FILE)

    row = {
        "timestamp":          datetime.utcnow().isoformat(),
        "episode":            episode,
        "coder_version":      coder_version,
        "breaker_tier":       breaker_tier,
        "avg_coder_reward":   round(avg_coder_reward, 4),
        "avg_breaker_reward": round(avg_breaker_reward, 4),
        "avg_pass_rate":      round(avg_pass_rate, 4),
        "total_fail_count":   total_fail_count,
        "total_error_count":  total_error_count,
        "total_timeout_count":total_timeout_count,
        "avg_break_rate":     round(avg_break_rate, 4),
        "steps":              steps,
    }

    with open(LOG_EPISODES_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ──────────────────────────────────────────────
# Summary logging
# ──────────────────────────────────────────────

def update_summary(
    total_episodes: int,
    coder_version: str,
    final_breaker_tier: int,
    all_coder_rewards: list[float],
    all_breaker_rewards: list[float],
    all_pass_rates: list[float],
    all_break_rates: list[float],
    coach_memory_summary: dict[str, Any],
) -> None:
    """
    Overwrite logs/summary.json with the latest aggregate statistics.
    """
    _ensure_log_dir()

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary = {
        "generated_at":         datetime.utcnow().isoformat(),
        "total_episodes":       total_episodes,
        "coder_version":        coder_version,
        "final_breaker_tier":   final_breaker_tier,
        "avg_coder_reward":     avg(all_coder_rewards),
        "avg_breaker_reward":   avg(all_breaker_rewards),
        "avg_pass_rate":        avg(all_pass_rates),
        "avg_break_rate":       avg(all_break_rates),
        "best_coder_reward":    round(max(all_coder_rewards), 4) if all_coder_rewards else 0.0,
        "worst_coder_reward":   round(min(all_coder_rewards), 4) if all_coder_rewards else 0.0,
        "coach_memory_summary": coach_memory_summary,
    }
    _write_json(LOG_SUMMARY_FILE, summary)


# ──────────────────────────────────────────────
# Convenience: print a compact log path report
# ──────────────────────────────────────────────

def print_log_paths() -> None:
    """Print the paths of all updated log files."""
    for path in [LOG_REWARDS_FILE, LOG_EPISODES_FILE, LOG_SUMMARY_FILE]:
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists}  {path}")
