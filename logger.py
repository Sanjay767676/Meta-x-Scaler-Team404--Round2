# logger.py
# Metrics logging for FORGE-v4.
# Writes structured logs to logs/rewards.json, logs/episodes.csv, logs/summary.json.

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from config import LOG_REWARDS_FILE, LOG_EPISODES_FILE, LOG_SUMMARY_FILE, LOG_DIR
from storage.artifact_store import save_run_artifact


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _ensure_log_dir() -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def _load_json(path: str, default: Any) -> Any:
    p = Path(path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return default


def _write_json(path: str, data: Any) -> None:
    p = Path(path)
    tmp_path = p.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(str(tmp_path), str(p))


# ──────────────────────────────────────────────
# Step-level logging
# ──────────────────────────────────────────────

def log_steps_batch(records_to_add: list[dict[str, Any]]) -> None:
    """
    Append a batch of step records to logs/rewards.json efficiently.
    """
    if not records_to_add:
        return
        
    _ensure_log_dir()
    records: list[dict[str, Any]] = _load_json(LOG_REWARDS_FILE, [])
    
    now = datetime.utcnow().isoformat()
    for r in records_to_add:
        r.setdefault("timestamp", now)
        # Ensure rounding for clean logs
        for k in ["coder_reward", "breaker_reward", "pass_rate", "break_rate"]:
            if k in r:
                r[k] = round(float(r[k]), 4)
                
    records.extend(records_to_add)
    _write_json(LOG_REWARDS_FILE, records)


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
    Single step logger (calls batch internally for backwards compatibility).
    """
    record = {
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
    log_steps_batch([record])


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
    p = Path(LOG_EPISODES_FILE)
    file_exists = p.exists()

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

    with open(p, "a", newline="", encoding="utf-8") as f:
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


def write_episode_report(episode: int, payload: dict[str, Any]) -> str:
    """Write deterministic per-episode artifact for auditability and replay."""
    filename = f"episode_{episode:04d}.json"
    return save_run_artifact(filename=filename, payload=payload)


# ──────────────────────────────────────────────
# Convenience: print a compact log path report
# ──────────────────────────────────────────────

def print_log_paths() -> None:
    """Print the paths of all updated log files."""
    for path in [LOG_REWARDS_FILE, LOG_EPISODES_FILE, LOG_SUMMARY_FILE]:
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {exists}  {path}")
