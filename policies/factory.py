"""Factory utilities for selecting coder policy implementations."""

from __future__ import annotations

from policies.api_model import APIModelPolicy
from policies.base import CoderPolicy
from policies.heuristic import HeuristicPolicy
from policies.local_model import LocalModelPolicy
from policies.mock_model import MockModelPolicy


def build_policy(name: str, strategy: str = "improving_coder") -> CoderPolicy:
    """Build a policy by short name.

    Supported names: heuristic | api | local | mock | model.
    """
    normalized = (name or "").strip().lower()
    if normalized == "api":
        return APIModelPolicy()
    if normalized == "local":
        return LocalModelPolicy()
    if normalized == "mock":
        return MockModelPolicy()
    if normalized == "model":
        from config import LLM_PROVIDER

        if LLM_PROVIDER == "openrouter":
            return APIModelPolicy(provider_name="openrouter")
        if LLM_PROVIDER in ("huggingface_local", "hf_local", "local"):
            return LocalModelPolicy()
        return MockModelPolicy()
    return HeuristicPolicy(strategy=strategy)
