"""Deterministic baseline provider — always returns valid FORGE `solution(arr)` code."""

from __future__ import annotations

import logging

from forge.llm_types import LLMResponse

logger = logging.getLogger("forge.offline_baseline")

# FORGE sandbox requires callable solution(arr), not solve(arr).
BASELINE_SOLUTION = (
    "def solution(arr):\n"
    "    return sorted(arr)\n"
)


class MockProvider:
    name = "offline"

    def __init__(self, model_id: str = "forge-baseline") -> None:
        self.model_id = model_id

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        logger.info("[offline_baseline] returning deterministic sorted(arr) solution")
        return LLMResponse(
            provider=self.name,
            model=self.model_id,
            content=BASELINE_SOLUTION,
            raw={"mode": "deterministic_baseline"},
        )
