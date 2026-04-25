"""Coach memory storage and summarization for FORGE-v4."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from config import MEMORY_FILE, MEMORY_MAX_LESSONS


class CoachMemory:
    """
    Persistent memory that accumulates lessons learned across training episodes.

    Lessons are stored as a list of dicts in a JSON file and loaded on startup.
    Each lesson includes a human-readable "coach_note" derived from the metrics
    so the history is understandable without post-processing.
    """

    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self.lessons: list[dict[str, Any]] = []
        self._ensure_data_dir()
        self.load()

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def add_lesson(
        self,
        episode: int,
        agent: str,
        observation: str,
        coder_reward: float,
        breaker_reward: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a lesson from one episode step.

        Args:
            episode:        Episode index.
            agent:          "coder" | "breaker" | "env".
            observation:    Raw observation string from the environment.
            coder_reward:   Total coder reward for this step.
            breaker_reward: Total breaker reward for this step.
            extra:          Optional metadata (pass_rate, fail_count, etc.).
        """
        coach_note = self._derive_coach_note(episode, extra or {})

        reward_delta = round(coder_reward - breaker_reward, 4)
        reward_weight = round(abs(coder_reward) + abs(breaker_reward), 4)

        lesson = {
            "timestamp":      datetime.utcnow().isoformat(),
            "episode":        episode,
            "agent":          agent,
            "observation":    observation,
            "coach_note":     coach_note,
            "coder_reward":   coder_reward,
            "breaker_reward": breaker_reward,
            "reward_delta":   reward_delta,
            "reward_weight":  reward_weight,
        }
        if extra:
            lesson["extra"] = extra

        self.lessons.append(lesson)
        if len(self.lessons) > MEMORY_MAX_LESSONS:
            self.lessons = self.lessons[-MEMORY_MAX_LESSONS:]
        self.save()

    def get_lessons(
        self,
        agent: str | None = None,
        last_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve stored lessons, optionally filtered by agent and/or limited to the last N.
        """
        result = self.lessons
        if agent is not None:
            result = [l for l in result if l.get("agent") == agent]
        if last_n is not None:
            result = result[-last_n:]
        return result

    def get_coach_notes(self, last_n: int = 5) -> list[str]:
        """Return the most recent human-readable coach notes."""
        return [l["coach_note"] for l in self.lessons[-last_n:] if l.get("coach_note")]

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of stored lessons."""
        if not self.lessons:
            return {
                "total_lessons": 0,
                "episodes_seen": 0,
                "weighted_signal": 0.0,
                "top_lessons": [],
            }

        episodes = {l["episode"] for l in self.lessons}
        coder_rewards   = [l["coder_reward"]   for l in self.lessons]
        breaker_rewards = [l["breaker_reward"] for l in self.lessons]

        weighted_signal = sum((l.get("reward_delta", 0.0)) * max(1.0, l.get("reward_weight", 0.0)) for l in self.lessons)
        top_lessons = sorted(self.lessons, key=lambda item: item.get("reward_weight", 0.0), reverse=True)[:3]

        return {
            "total_lessons":      len(self.lessons),
            "episodes_seen":      len(episodes),
            "avg_coder_reward":   round(sum(coder_rewards)   / len(coder_rewards),   4),
            "avg_breaker_reward": round(sum(breaker_rewards) / len(breaker_rewards), 4),
            "weighted_signal":    round(weighted_signal, 4),
            "recent_coach_notes": self.get_coach_notes(last_n=3),
            "top_lessons": [
                {
                    "episode": lesson.get("episode"),
                    "coach_note": lesson.get("coach_note", ""),
                    "reward_weight": lesson.get("reward_weight", 0.0),
                }
                for lesson in top_lessons
            ],
        }

    def clear(self) -> None:
        """Wipe all stored lessons (use with caution)."""
        self.lessons = []
        self.save()

    # ──────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────

    def save(self) -> None:
        """Persist lessons to JSON file."""
        tmp_path = f"{self.filepath}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.lessons, f, indent=2)
        os.replace(tmp_path, self.filepath)

    def load(self) -> None:
        """Load lessons from JSON file if it exists."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.lessons = loaded if isinstance(loaded, list) else []
            except (json.JSONDecodeError, IOError):
                self.lessons = []
        else:
            self.lessons = []

    # ──────────────────────────────────────────────
    # Coach note derivation
    # ──────────────────────────────────────────────

    def _derive_coach_note(self, episode: int, extra: dict[str, Any]) -> str:
        """
        Generate a human-readable coaching note from step metadata.

        Examples:
            "Episode 4: Coder failed on duplicates → handle duplicate values safely"
            "Episode 8: Coder timed out on large arrays → avoid O(n²) for large inputs"
            "Episode 2: Strong performance (pass_rate=1.00) → keep current strategy"
        """
        pass_rate     = extra.get("coder_pass_rate",    None)
        fail_count    = extra.get("fail_count",          0)
        error_count   = extra.get("error_count",         0)
        timeout_count = extra.get("timeout_count",       0)
        breaker_tier  = extra.get("breaker_tier",        1)
        coder_version = extra.get("coder_version",      "unknown")
        recent_case   = extra.get("recent_breaker_case", [])

        prefix = f"Episode {episode}"

        # Timeout note
        if timeout_count > 0:
            return (
                f"{prefix}: Coder timed out on {timeout_count} test(s)"
                f" [tier={breaker_tier}] → avoid O(n²) or infinite loops for large inputs"
            )

        # Error note
        if error_count > 0 and pass_rate is not None and pass_rate < 0.5:
            return (
                f"{prefix}: Coder raised errors on {error_count} test(s)"
                f" → add input validation and handle edge cases"
            )

        # Negative/duplicate failure detection from recent breaker case
        if fail_count > 0 and recent_case:
            has_neg  = any(x < 0 for x in recent_case)
            has_dups = len(recent_case) != len(set(recent_case))
            is_large = len(recent_case) >= 10

            if has_neg and has_dups:
                return (
                    f"{prefix}: Coder ({coder_version}) failed on negatives+duplicates"
                    f" → ensure sort key uses true value, not abs()"
                )
            if has_neg:
                return (
                    f"{prefix}: Coder ({coder_version}) failed on negative values"
                    f" → handle negative integers in comparison logic"
                )
            if has_dups:
                return (
                    f"{prefix}: Coder ({coder_version}) failed on duplicate values"
                    f" → ensure stable sort handles equal elements correctly"
                )
            if is_large:
                return (
                    f"{prefix}: Coder ({coder_version}) failed on large array (n={len(recent_case)})"
                    f" → consider O(n log n) algorithm"
                )
            return (
                f"{prefix}: Coder ({coder_version}) failed {fail_count} test(s)"
                f" at breaker {breaker_tier} → review edge case handling"
            )

        # Good performance
        if pass_rate is not None and pass_rate >= 0.8:
            return (
                f"{prefix}: Strong performance (pass_rate={pass_rate:.2f})"
                f" [{coder_version}] → breaker should escalate tier"
            )

        # Generic fallback
        pr = f"{pass_rate:.2f}" if pass_rate is not None else "N/A"
        return f"{prefix}: pass_rate={pr}, fail={fail_count}, errors={error_count}"

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _ensure_data_dir(self) -> None:
        """Create the directory for the memory file if it doesn't exist."""
        directory = os.path.dirname(self.filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
