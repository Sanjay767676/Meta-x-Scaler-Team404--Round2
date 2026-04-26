"""Ordered inference router.

* **auto:** nim → openrouter → (optional **custom_hf** if `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
  is set) → offline baseline. Without a Hub token, local HF is skipped so CPU Spaces stay responsive.
* **Explicit modes** call one provider (with fallback to offline baseline where implemented).
"""

from __future__ import annotations

import logging
import os
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


def _cuda_ok_for_custom_hf() -> bool:
    """Local HF is only practical on GPU unless explicitly overridden (debug)."""
    if os.getenv("FORGE_ALLOW_CUSTOM_HF_CPU", "").strip().lower() in ("1", "true", "yes"):
        return True
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


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
            FORGE_HF_MAX_NEW_TOKENS,
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
            max_new_tokens=FORGE_HF_MAX_NEW_TOKENS,
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
        self._include_hf_in_auto = bool((HF_TOKEN or "").strip())

    def generate(self, prompt: str, system_prompt: str = "", mode: str = "auto") -> LLMResponse:
        mode = (mode or "auto").strip().lower()
        if mode in ("offline", "mock"):
            return self._mock.generate(prompt, system_prompt)

        if mode == "custom_hf":
            if not _cuda_ok_for_custom_hf():
                logger.info("[router] custom_hf: no CUDA — offline baseline (set FORGE_ALLOW_CUSTOM_HF_CPU=1 to force CPU load)")
                return self._mock.generate(prompt, system_prompt)
            return self._try_hf(prompt, system_prompt, fallback=True)
        if mode == "nim":
            return self._try_nim(prompt, system_prompt, fallback=True)
        if mode == "openrouter":
            return self._try_openrouter(prompt, system_prompt, fallback=True)

        # auto: cloud APIs first; local HF only if a Hub token is configured (still last before offline baseline).
        auto_steps: list[tuple[str, Callable[[], LLMResponse]]] = [
            ("nim", lambda: self._try_nim(prompt, system_prompt, fallback=False)),
            ("openrouter", lambda: self._try_openrouter(prompt, system_prompt, fallback=False)),
        ]
        if self._include_hf_in_auto:
            auto_steps.append(
                ("custom_hf", lambda: self._try_hf(prompt, system_prompt, fallback=False)),
            )
        for label, fn in auto_steps:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[router] %s failed (%s), trying next provider", label, exc)

        logger.warning("[router] all remote providers failed; using offline baseline")
        return self._mock.generate(prompt, system_prompt)

    def _try_hf(self, prompt: str, system_prompt: str, fallback: bool) -> LLMResponse:
        if not _cuda_ok_for_custom_hf():
            if fallback:
                logger.info("[router] custom_hf skipped in chain: no CUDA")
                return self._mock.generate(prompt, system_prompt)
            raise RuntimeError("custom_hf requires CUDA (or FORGE_ALLOW_CUSTOM_HF_CPU=1)")
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
