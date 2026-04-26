"""Ordered inference router.

* **auto:** nim → openrouter → mock only. **custom_hf is not in `auto`** — local PyTorch
  loads are slow on CPU Spaces; pick **custom_hf** explicitly when you want the adapter.
* **Explicit modes** call one provider (with fallback to mock where implemented).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Callable

from forge.llm_types import LLMResponse
from forge.providers.hf_custom import HFCustomProvider
from forge.providers.mock import MockProvider
from forge.providers.nim import NIMProvider
from forge.providers.openrouter import OpenRouterProvider

logger = logging.getLogger("forge.router")

_ROUTER_SINGLETON: "InferenceRouter | None" = None


def get_inference_router() -> "InferenceRouter":
    global _ROUTER_SINGLETON
    if _ROUTER_SINGLETON is None:
        _ROUTER_SINGLETON = InferenceRouter()
    return _ROUTER_SINGLETON


def _with_timeout(fn: Callable[[], LLMResponse], seconds: float) -> LLMResponse:
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=seconds)
        except FuturesTimeout as exc:
            raise TimeoutError(f"Inference exceeded {seconds}s") from exc


class InferenceRouter:
    """Try providers in order for mode=auto; single provider for explicit modes."""

    def __init__(self) -> None:
        from config import (
            BASE_MODEL_ID,
            HF_ADAPTER_REPO,
            HF_TOKEN,
            NIM_API_KEY,
            NIM_BASE_URL,
            NIM_MODEL,
            OPENROUTER_API_KEY,
            OPENROUTER_BASE_URL,
            OPENROUTER_MODEL,
            ROUTER_HF_TIMEOUT_SEC,
            ROUTER_NIM_TIMEOUT_SEC,
            ROUTER_OPENROUTER_TIMEOUT_SEC,
        )

        self._mock = MockProvider()
        self._hf = HFCustomProvider(
            base_model_id=BASE_MODEL_ID,
            adapter_id=HF_ADAPTER_REPO,
            hf_token=HF_TOKEN or None,
        )
        self._nim = NIMProvider(
            api_key=NIM_API_KEY,
            model_id=NIM_MODEL,
            base_url=NIM_BASE_URL,
            timeout_sec=ROUTER_NIM_TIMEOUT_SEC,
        )
        self._openrouter = OpenRouterProvider(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            model_id=OPENROUTER_MODEL,
            timeout_sec=ROUTER_OPENROUTER_TIMEOUT_SEC,
        )
        self._timeouts = {
            "custom_hf": ROUTER_HF_TIMEOUT_SEC,
            "nim": ROUTER_NIM_TIMEOUT_SEC,
            "openrouter": ROUTER_OPENROUTER_TIMEOUT_SEC,
        }

    def generate(self, prompt: str, system_prompt: str = "", mode: str = "auto") -> LLMResponse:
        mode = (mode or "auto").strip().lower()
        if mode == "mock":
            return self._mock.generate(prompt, system_prompt)

        if mode == "custom_hf":
            return self._try_hf(prompt, system_prompt, fallback=True)
        if mode == "nim":
            return self._try_nim(prompt, system_prompt, fallback=True)
        if mode == "openrouter":
            return self._try_openrouter(prompt, system_prompt, fallback=True)

        # auto: cloud APIs only, then mock. (custom_hf excluded — it can block minutes on CPU.)
        for label, fn in (
            ("nim", lambda: self._try_nim(prompt, system_prompt, fallback=False)),
            ("openrouter", lambda: self._try_openrouter(prompt, system_prompt, fallback=False)),
        ):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[router] %s failed (%s), trying next provider", label, exc)

        logger.warning("[router] all remote providers failed; using mock")
        return self._mock.generate(prompt, system_prompt)

    def _try_hf(self, prompt: str, system_prompt: str, fallback: bool) -> LLMResponse:
        try:
            sec = self._timeouts["custom_hf"]

            def call() -> LLMResponse:
                return self._hf.generate(prompt, system_prompt)

            return _with_timeout(call, sec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[router] custom_hf error: %s", exc)
            if fallback:
                return self._mock.generate(prompt, system_prompt)
            raise

    def _try_nim(self, prompt: str, system_prompt: str, fallback: bool) -> LLMResponse:
        try:
            sec = self._timeouts["nim"]

            def call() -> LLMResponse:
                return self._nim.generate(prompt, system_prompt)

            return _with_timeout(call, sec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[router] nim error: %s", exc)
            if fallback:
                return self._mock.generate(prompt, system_prompt)
            raise

    def _try_openrouter(self, prompt: str, system_prompt: str, fallback: bool) -> LLMResponse:
        try:
            sec = self._timeouts["openrouter"]

            def call() -> LLMResponse:
                return self._openrouter.generate(prompt, system_prompt)

            return _with_timeout(call, sec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[router] openrouter error: %s", exc)
            if fallback:
                return self._mock.generate(prompt, system_prompt)
            raise
