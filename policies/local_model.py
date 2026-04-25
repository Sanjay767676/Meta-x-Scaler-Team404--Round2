"""Local model fallback policy for offline operation."""

from __future__ import annotations

from typing import Any

from llm_agent import extract_python_code, get_provider
from policies.base import CodeCandidate, CoderPolicy


class LocalModelPolicy(CoderPolicy):
    """Generate candidates from local provider path without network dependency."""

    name = "local"

    def __init__(self) -> None:
        self.provider = get_provider("huggingface_local")

    def generate_candidates(self, state: dict[str, Any], num_candidates: int) -> list[CodeCandidate]:
        prompt = state.get("problem_description", "Write solution(arr) that sorts integers ascending.")
        candidates: list[CodeCandidate] = []
        for idx in range(max(1, num_candidates)):
            response = self.provider.generate(prompt=prompt)
            code = extract_python_code(response.content)
            candidates.append(
                CodeCandidate(
                    code=code,
                    source="local:model",
                    metadata={"candidate_idx": idx, "model": response.model},
                )
            )
        return candidates
