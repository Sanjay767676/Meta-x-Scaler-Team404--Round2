"""Provider-ready LLM abstraction for FORGE-v4.

This module intentionally supports dry-run mode by default so the project works
without API keys during hackathons.
"""

from __future__ import annotations

import os
from typing import Any

from config import (
    HF_LOCAL_MODEL_ID,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    NIM_MODEL,
    NVIDIA_API_KEY,
)
from forge.llm_types import LLMResponse


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
                "model": OPENROUTER_MODEL,
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
                model=OPENROUTER_MODEL,
                content=content,
                raw={"status": "ok", "base_url": self.base_url},
            )
        except Exception as exc:  # noqa: BLE001
            fallback = MockFallbackProvider().generate(prompt=prompt, system_prompt=system_prompt)
            fallback.raw["openrouter_error"] = f"{type(exc).__name__}: {exc}"
            return fallback


class HuggingFaceAPIProvider(BaseLLMProvider):
    """Hugging Face Inference API provider."""

    name = "hf_api"

    def __init__(self, api_key: str = "", model_id: str = "qwen/Qwen2.5-Coder-7B-Instruct"):
        self.api_key = api_key or os.getenv("HF_TOKEN", "")
        self.model_id = model_id

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        if not self.api_key:
            return MockFallbackProvider().generate(prompt=prompt, system_prompt=system_prompt)

        try:
            import requests  # type: ignore

            headers = {"Authorization": f"Bearer {self.api_key}"}
            api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            
            # Simple prompt formatting for instruction models
            full_prompt = f"<|im_start|>system\n{system_prompt or 'You are a Python coding model.'}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            payload = {
                "inputs": full_prompt,
                "parameters": {"max_new_tokens": 512, "temperature": 0.2},
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # HF API returns list of results
            content = data[0].get("generated_text", "") if isinstance(data, list) else data.get("generated_text", "")
            
            # Strip the prompt if the model returns it
            if content.startswith(full_prompt):
                content = content[len(full_prompt):].strip()
            
            return LLMResponse(
                provider=self.name,
                model=self.model_id,
                content=content.strip(),
                raw={"status": "ok", "api_url": api_url},
            )
        except Exception as exc:  # noqa: BLE001
            fallback = MockFallbackProvider().generate(prompt=prompt, system_prompt=system_prompt)
            fallback.raw["hf_api_error"] = f"{type(exc).__name__}: {exc}"
            return fallback


class NvidiaNIMProvider(BaseLLMProvider):
    """NVIDIA NIM provider with OpenAI-compatible API."""

    name = "nim"

    def __init__(self, api_key: str = NVIDIA_API_KEY, model_id: str = NIM_MODEL):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.model_id = model_id
        self.base_url = "https://integrate.api.nvidia.com/v1"

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        if not self.api_key:
            return MockFallbackProvider().generate(prompt=prompt, system_prompt=system_prompt)

        # Default system prompt for trained behavior if none provided
        sys_prompt = system_prompt or (
            "You are a defender model trained on FORGE adversarial sorting failures. "
            "Produce robust Python solutions that handle edge cases (negatives, duplicates, large inputs) efficiently."
        )

        try:
            import requests  # type: ignore

            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=45,
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
                raise ValueError("NVIDIA NIM returned empty content")
            
            return LLMResponse(
                provider=self.name,
                model=self.model_id,
                content=content,
                raw={"status": "ok", "model": self.model_id, "provider": "nvidia_nim"},
            )
        except Exception as exc:  # noqa: BLE001
            fallback = MockFallbackProvider().generate(prompt=prompt, system_prompt=sys_prompt)
            fallback.raw["nim_error"] = f"{type(exc).__name__}: {exc}"
            return fallback


def get_provider(provider_name: str = LLM_PROVIDER) -> BaseLLMProvider:
    """Factory for active provider."""
    name = (provider_name or "").strip().lower()
    if name == "openrouter":
        return OpenRouterProvider()
    if name in ("hf_api", "huggingface_api"):
        return HuggingFaceAPIProvider()
    if name in ("huggingface_local", "hf_local", "local"):
        return HuggingFaceLocalProvider()
    if name == "nim":
        return NvidiaNIMProvider()
    return MockFallbackProvider()


def extract_python_code(text: str) -> str:
    """Extract code block between triple backticks if present, else return whole text."""
    if not text:
        return ""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def generate_code(prompt: str, provider: str = "auto", system_prompt: str = "") -> str:
    """Unified entry: modular router (auto) or explicit backend (see forge/providers/router.py)."""
    from forge.providers.router import get_inference_router

    router = get_inference_router()
    response = router.generate(prompt=prompt, system_prompt=system_prompt, mode=provider)
    return extract_python_code(response.content)
