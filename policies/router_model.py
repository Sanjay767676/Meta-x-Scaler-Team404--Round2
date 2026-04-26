"""Defender policy using the FORGE inference router (auto: NIM → OpenRouter → mock)."""

from __future__ import annotations

from typing import Any

from forge.providers.router import get_inference_router
from llm_agent import extract_python_code
from memory import CoachMemory
from policies.base import CodeCandidate, CoderPolicy


class RouterModelPolicy(CoderPolicy):
    """Generate candidates via modular router; never crashes — mock ends the chain."""

    name = "model"

    def __init__(
        self,
        memory: CoachMemory | None = None,
        mode: str | None = None,
    ) -> None:
        from config import CODE_PROVIDER_MODE

        self.memory = memory or CoachMemory()
        self.mode = (mode or CODE_PROVIDER_MODE).strip().lower()
        self.router = get_inference_router()
        self.system_prompt = (
            "You are a FORGE defender model trained on adversarial sorting failures. "
            "Return Python code only and define solution(arr) that handles duplicates, negatives, and edge cases."
        )

    def generate_candidates(self, state: dict[str, Any], num_candidates: int) -> list[CodeCandidate]:
        prompt = state.get("problem_description", "Write solution(arr) that sorts integers ascending.")
        candidates: list[CodeCandidate] = []
        for idx in range(max(1, num_candidates)):
            response = self.router.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                mode=self.mode,
            )
            code = extract_python_code(response.content)
            if not code:
                continue
            candidates.append(
                CodeCandidate(
                    code=code,
                    source=f"router:{response.provider}",
                    metadata={"model": response.model, "candidate_idx": idx},
                )
            )
        if not candidates:
            fallback = self.router.generate(prompt, self.system_prompt, mode="mock")
            code = extract_python_code(fallback.content)
            if code:
                candidates.append(
                    CodeCandidate(
                        code=code,
                        source="router:mock_fallback",
                        metadata={"candidate_idx": 0},
                    )
                )
        return candidates
