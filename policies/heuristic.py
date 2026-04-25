"""Heuristic policy implementation backed by local strategy templates."""

from __future__ import annotations

from typing import Any

from agents import get_coder_code
from policies.base import CodeCandidate, CoderPolicy


class HeuristicPolicy(CoderPolicy):
    """Policy that uses deterministic built-in strategies from agents.py."""

    name = "heuristic"

    def __init__(self, strategy: str = "weak_coder_v2") -> None:
        self.strategy = strategy

    def generate_candidates(self, state: dict[str, Any], num_candidates: int) -> list[CodeCandidate]:
        episode = int(state.get("episode", 1))
        candidates: list[CodeCandidate] = []

        # Prioritize weak versions to ensure baseline fails on edge cases
        # Removed improving_coder from default search to ensure it stays a weak baseline
        strategy_order = [self.strategy, "weak_coder_v1", "weak_coder_v2", "weak_coder_v3"]
        if self.strategy == "improving_coder":
            # If user explicitly asked for improving_coder, we use it, but 
            # for the heuristic baseline we usually want something that fails.
            pass
        else:
            # If we are in "heuristic" mode, we should stick to buggy versions
            strategy_order = ["weak_coder_v1", "weak_coder_v2", "weak_coder_v3"]
        seen: set[str] = set()
        for strategy in strategy_order:
            if strategy in seen:
                continue
            seen.add(strategy)
            try:
                code = get_coder_code(strategy, episode=episode)
            except ValueError:
                continue
            candidates.append(
                CodeCandidate(code=code, source=f"heuristic:{strategy}", metadata={"strategy": strategy})
            )
            if len(candidates) >= num_candidates:
                break

        return candidates
