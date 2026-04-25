# memory.py
# Coach Memory system for FORGE-v4.
# Stores lessons learned across episodes in a JSON file.

import json
import os
from datetime import datetime
from typing import Any
from config import MEMORY_FILE


class CoachMemory:
    """
    Persistent memory that accumulates lessons learned across training episodes.

    Lessons are stored as a list of dicts in a JSON file and loaded on startup.
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
            observation:    Human-readable description of what happened.
            coder_reward:   Total coder reward for this step.
            breaker_reward: Total breaker reward for this step.
            extra:          Optional additional metadata.
        """
        lesson = {
            "timestamp":      datetime.utcnow().isoformat(),
            "episode":        episode,
            "agent":          agent,
            "observation":    observation,
            "coder_reward":   coder_reward,
            "breaker_reward": breaker_reward,
        }
        if extra:
            lesson["extra"] = extra

        self.lessons.append(lesson)
        self.save()

    def get_lessons(self, agent: str | None = None, last_n: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve stored lessons, optionally filtered by agent and/or limited to the last N.

        Args:
            agent:  Filter to a specific agent ("coder", "breaker", "env"), or None for all.
            last_n: Return only the last N lessons if provided.

        Returns:
            List of lesson dicts.
        """
        result = self.lessons
        if agent is not None:
            result = [l for l in result if l.get("agent") == agent]
        if last_n is not None:
            result = result[-last_n:]
        return result

    def summary(self) -> dict[str, Any]:
        """
        Return a high-level summary of stored lessons.
        """
        if not self.lessons:
            return {"total_lessons": 0, "episodes_seen": 0}

        episodes = {l["episode"] for l in self.lessons}
        coder_rewards = [l["coder_reward"] for l in self.lessons]
        breaker_rewards = [l["breaker_reward"] for l in self.lessons]

        return {
            "total_lessons":      len(self.lessons),
            "episodes_seen":      len(episodes),
            "avg_coder_reward":   round(sum(coder_rewards) / len(coder_rewards), 4),
            "avg_breaker_reward": round(sum(breaker_rewards) / len(breaker_rewards), 4),
        }

    def clear(self) -> None:
        """
        Wipe all stored lessons (use with caution).
        """
        self.lessons = []
        self.save()

    # ──────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────

    def save(self) -> None:
        """Persist lessons to JSON file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.lessons, f, indent=2)

    def load(self) -> None:
        """Load lessons from JSON file if it exists."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.lessons = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Start fresh if file is corrupted
                self.lessons = []
        else:
            self.lessons = []

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _ensure_data_dir(self) -> None:
        """Create the directory for the memory file if it doesn't exist."""
        directory = os.path.dirname(self.filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
