"""OpenRouter chat completions (REST)."""

from __future__ import annotations

import logging
from typing import Any

import requests

from forge.llm_types import LLMResponse

logger = logging.getLogger("forge.openrouter")


class OpenRouterProvider:
    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_id: str,
        timeout_sec: float,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_sec = timeout_sec

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a Python coding model."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        logger.info("[openrouter] POST %s/chat/completions model=%s", self.base_url, self.model_id)
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not content:
            raise ValueError("OpenRouter returned empty content")
        return LLMResponse(
            provider=self.name,
            model=self.model_id,
            content=content,
            raw={"status": "ok"},
        )
