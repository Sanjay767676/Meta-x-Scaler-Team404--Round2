"""API-backed policy implementation using llm_agent providers."""

from __future__ import annotations

from typing import Any

from llm_agent import get_provider
from policies.base import CodeCandidate, CoderPolicy


class APIModelPolicy(CoderPolicy):
    """Generate candidates through configured API providers."""

    name = "api"

    def __init__(self, provider_name: str = "openrouter") -> None:
        self.provider_name = provider_name
        self.provider = get_provider(provider_name)

    def generate_candidates(self, state: dict[str, Any], num_candidates: int) -> list[CodeCandidate]:
        prompt = state.get("problem_description", "Write solution(arr) that sorts integers ascending.")
        system_prompt = (
            "You are FORGE defender model. Return Python code only and define solution(arr)."
        )

        candidates: list[CodeCandidate] = []
        for idx in range(max(1, num_candidates)):
            response = self.provider.generate(prompt=prompt, system_prompt=system_prompt)
            code = response.content.strip()
            if not code:
                continue
            candidates.append(
                CodeCandidate(
                    code=code,
                    source=f"api:{response.provider}",
                    metadata={"model": response.model, "candidate_idx": idx},
                )
            )
        return candidates
