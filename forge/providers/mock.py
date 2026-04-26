"""Deterministic final fallback — always returns valid FORGE `solution(arr)` code."""

from __future__ import annotations

import logging

from forge.llm_types import LLMResponse

logger = logging.getLogger("forge.mock")

# FORGE sandbox requires callable solution(arr), not solve(arr).
MOCK_SOLUTION = (
    "def solution(arr):\n"
    "    return sorted(arr)\n"
)


class MockProvider:
    name = "mock"

    def __init__(self, model_id: str = "forge-mock") -> None:
        self.model_id = model_id

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        logger.info("[mock] returning deterministic sorted(arr) solution")
        return LLMResponse(
            provider=self.name,
            model=self.model_id,
            content=MOCK_SOLUTION,
            raw={"mode": "deterministic_mock"},
        )
