"""NVIDIA NIM OpenAI-compatible chat completions."""

from __future__ import annotations

import logging
from typing import Any

import requests

from forge.llm_types import LLMResponse

logger = logging.getLogger("forge.nim")

NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"


class NIMProvider:
    name = "nim"

    def __init__(
        self,
        api_key: str,
        model_id: str,
        base_url: str = NIM_DEFAULT_BASE,
        timeout_sec: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        if not self.api_key:
            raise ValueError("NIM_API_KEY / NVIDIA_API_KEY is not set")

        sys_prompt = system_prompt or (
            "You are a FORGE defender. Output Python only; define solution(arr) for robust sorting."
        )
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        logger.info("[nim] POST %s/chat/completions model=%s", self.base_url, self.model_id)
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
            raise ValueError("NIM returned empty content")
        return LLMResponse(
            provider=self.name,
            model=self.model_id,
            content=content,
            raw={"status": "ok"},
        )
