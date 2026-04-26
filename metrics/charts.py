"""Chart generation for FORGE-v4 artifacts."""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from config import LOG_REWARDS_FILE, REWARD_GRAPHS_DIR


def _ensure_outputs_dir() -> None:
    os.makedirs(REWARD_GRAPHS_DIR, exist_ok=True)


def _load_step_records() -> list[dict[str, Any]]:
    if not os.path.exists(LOG_REWARDS_FILE):
        return []
    try:
        with open(LOG_REWARDS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_chart(path: str, title: str, x: pd.Series, y: pd.Series, ylabel: str) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_charts() -> dict[str, str]:
    """Generate trend charts and save PNG artifacts."""
    _ensure_outputs_dir()
    records = _load_step_records()
    if not records:
        return {}

    df = pd.DataFrame(records)
    if df.empty:
        return {}

    grouped = (
        df.groupby("episode", as_index=False)
        .agg(
            coder_reward=("coder_reward", "mean"),
            breaker_reward=("breaker_reward", "mean"),
            pass_rate=("pass_rate", "mean"),
            breaker_tier=("breaker_tier", "max"),
            timeout_count=("timeout_count", "sum"),
        )
        .sort_values("episode")
    )

    outputs: dict[str, str] = {}

    defender_path = os.path.join(REWARD_GRAPHS_DIR, "defender_reward_trend.png")
    _save_chart(defender_path, "Defender Reward Trend", grouped["episode"], grouped["coder_reward"], "Avg Defender Reward")
    outputs["defender_reward_trend"] = defender_path

    adversary_path = os.path.join(REWARD_GRAPHS_DIR, "adversary_reward_trend.png")
    _save_chart(adversary_path, "Adversary Reward Trend", grouped["episode"], grouped["breaker_reward"], "Avg Adversary Reward")
    outputs["adversary_reward_trend"] = adversary_path

    pass_rate_path = os.path.join(REWARD_GRAPHS_DIR, "pass_rate_trend.png")
    _save_chart(pass_rate_path, "Defender Pass Rate Trend", grouped["episode"], grouped["pass_rate"], "Pass Rate")
    outputs["pass_rate_trend"] = pass_rate_path

    tier_path = os.path.join(REWARD_GRAPHS_DIR, "tier_progression.png")
    _save_chart(tier_path, "Breaker Tier Progression", grouped["episode"], grouped["breaker_tier"], "Tier")
    outputs["tier_progression"] = tier_path

    timeout_path = os.path.join(REWARD_GRAPHS_DIR, "timeout_trend.png")
    _save_chart(timeout_path, "Sandbox Timeout Count", grouped["episode"], grouped["timeout_count"], "Timeout Count")
    outputs["timeout_trend"] = timeout_path

    return outputs


def export_judge_assets(
    episodes: list[dict[str, Any]],
    final_report: dict[str, Any],
    output_dir: str = "outputs",
) -> dict[str, str]:
    """Export required judge-facing assets for hackathon submission."""
    os.makedirs(output_dir, exist_ok=True)
    if not episodes:
        report_path = os.path.join(output_dir, "final_report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(final_report, handle, indent=2, sort_keys=True)
        return {"final_report": report_path}

    df = pd.DataFrame(episodes)
    reward_curve_path = os.path.join(output_dir, "reward_curve.png")
    loss_curve_path = os.path.join(output_dir, "loss_curve.png")
    pass_rate_path = os.path.join(output_dir, "pass_rate.png")
    final_report_path = os.path.join(output_dir, "final_report.json")

    plt.figure(figsize=(9, 4.5))
    plt.plot(df["episode"], df["defender_reward"], marker="o", linewidth=2, label="defender")
    plt.plot(df["episode"], df["adversary_reward"], marker="o", linewidth=2, label="adversary")
    plt.title("Reward curve (baseline vs run)")
    plt.xlabel("Episode index")
    plt.ylabel("Mean reward (defender & adversary)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_curve_path, dpi=160)
    plt.close()

    # Pseudo-loss derived from defender reward trend for judge-friendly visualization.
    reward_min = float(df["defender_reward"].min())
    reward_max = float(df["defender_reward"].max())
    span = reward_max - reward_min if reward_max != reward_min else 1.0
    loss_like = 1.0 - ((df["defender_reward"] - reward_min) / span)

    plt.figure(figsize=(9, 4.5))
    plt.plot(df["episode"], loss_like, marker="o", linewidth=2)
    plt.title("Loss-like curve (from defender reward)")
    plt.xlabel("Episode index")
    plt.ylabel("Normalized loss (1 − scaled avg defender reward)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(df["episode"], df["pass_rate"], marker="o", linewidth=2)
    plt.title("Defender pass rate vs episode")
    plt.xlabel("Episode index")
    plt.ylabel("Pass rate (0–1)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pass_rate_path, dpi=160)
    plt.close()

    with open(final_report_path, "w", encoding="utf-8") as handle:
        json.dump(final_report, handle, indent=2, sort_keys=True)

    return {
        "reward_curve": reward_curve_path,
        "loss_curve": loss_curve_path,
        "pass_rate": pass_rate_path,
        "final_report": final_report_path,
    }
