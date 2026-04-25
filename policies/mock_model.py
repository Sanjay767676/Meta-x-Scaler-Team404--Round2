"""Mock fallback policy for stable offline benchmark runs."""

from __future__ import annotations

from typing import Any

from llm_agent import get_provider
from policies.base import CodeCandidate, CoderPolicy


class MockModelPolicy(CoderPolicy):
    """Always uses the mock provider regardless of external dependencies."""

    name = "mock"

    def __init__(self) -> None:
        self.provider = get_provider("mock")

    def generate_candidates(self, state: dict[str, Any], num_candidates: int) -> list[CodeCandidate]:
        prompt = state.get("problem_description", "Write solution(arr) that sorts integers ascending.")
        candidates: list[CodeCandidate] = []
        for idx in range(max(1, num_candidates)):
            response = self.provider.generate(prompt=prompt)
            candidates.append(
                CodeCandidate(
                    code=response.content,
                    source="mock:model",
                    metadata={"candidate_idx": idx, "model": response.model},
                )
            )
        return candidates
