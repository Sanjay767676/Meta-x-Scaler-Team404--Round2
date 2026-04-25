"""Provider-ready LLM abstraction for FORGE-v4.

This module intentionally supports dry-run mode by default so the project works
without API keys during hackathons.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from config import (
    HF_LOCAL_MODEL_ID,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


@dataclass
class LLMResponse:
    """Normalized response payload for any provider."""

    provider: str
    model: str
    content: str
    raw: dict[str, Any]


class BaseLLMProvider:
    """Base provider contract."""

    name = "base"

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        raise NotImplementedError


class MockFallbackProvider(BaseLLMProvider):
    """Deterministic fallback provider for offline/dev use."""

    name = "mock"

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        content = (
            "def solution(arr):\n"
            "    # Professional robust sorting implementation\n"
            "    # Uses Python's built-in Timsort (O(n log n)) which is stable and correct\n"
            "    if not isinstance(arr, list):\n"
            "        return []\n"
            "    return sorted(list(arr))\n"
        )
        return LLMResponse(
            provider=self.name,
            model=LLM_MODEL,
            content=content,
            raw={"mode": "local_fallback", "system_prompt": system_prompt, "prompt": prompt[:200]},
        )


class HuggingFaceLocalProvider(BaseLLMProvider):
    """Local HF model provider with safe fallback when dependencies are unavailable."""

    name = "huggingface_local"

    def __init__(self, model_id: str = HF_LOCAL_MODEL_ID):
        self.model_id = model_id
        self._pipeline = None
        self._load_error = ""

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None or self._load_error:
            return
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline(
                task="text-generation",
                model=self.model_id,
                max_new_tokens=200,
                do_sample=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        self._ensure_pipeline()
        if self._pipeline is None:
            fallback = MockFallbackProvider().generate(prompt=prompt, system_prompt=system_prompt)
            fallback.raw["hf_local_error"] = self._load_error
            fallback.raw["requested_model"] = self.model_id
            return fallback

        try:
            full_prompt = f"{system_prompt}\n\n{prompt}".strip()
            response = self._pipeline(full_prompt)
            generated = response[0].get("generated_text", "") if response else ""
            code = generated.strip()
            if not code:
                raise ValueError("HF local model returned empty text")
            return LLMResponse(
                provider=self.name,
                model=self.model_id,
                content=code,
                raw={"mode": "hf_local", "requested_model": self.model_id},
            )
        except Exception as exc:  # noqa: BLE001
            fallback = MockFallbackProvider().generate(prompt=prompt, system_prompt=system_prompt)
            fallback.raw["hf_local_runtime_error"] = f"{type(exc).__name__}: {exc}"
            return fallback


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider with graceful fallback when network/key is unavailable."""

    name = "openrouter"

    def __init__(self, api_key: str = OPENROUTER_API_KEY, base_url: str = OPENROUTER_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            return MockFallbackProvider().generate(prompt=prompt, system_prompt=system_prompt)

        try:
            import requests  # type: ignore

            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt or "You are a Python coding model."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
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
                model=LLM_MODEL,
                content=content,
                raw={"status": "ok", "base_url": self.base_url},
            )
        except Exception as exc:  # noqa: BLE001
            fallback = MockFallbackProvider().generate(prompt=prompt, system_prompt=system_prompt)
            fallback.raw["openrouter_error"] = f"{type(exc).__name__}: {exc}"
            return fallback


def get_provider(provider_name: str = LLM_PROVIDER) -> BaseLLMProvider:
    """Factory for active provider."""
    name = (provider_name or "").strip().lower()
    if name == "openrouter":
        return OpenRouterProvider()
    if name in ("huggingface_local", "hf_local", "local"):
        return HuggingFaceLocalProvider()
    return MockFallbackProvider()


def generate_code(prompt: str, system_prompt: str = "") -> str:
    """Convenience API for future trainer/env integration."""
    provider = get_provider()
    response = provider.generate(prompt=prompt, system_prompt=system_prompt)
    return response.content
