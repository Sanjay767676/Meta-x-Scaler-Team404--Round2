"""Factory utilities for selecting coder policy implementations."""

from __future__ import annotations

from policies.api_model import APIModelPolicy
from policies.base import CoderPolicy
from policies.heuristic import HeuristicPolicy
from policies.local_model import LocalModelPolicy
from policies.mock_model import MockModelPolicy


from memory import CoachMemory


def build_policy(
    name: str,
    strategy: str = "improving_coder",
    memory: CoachMemory | None = None,
    forge_provider: str | None = None,
) -> CoderPolicy:
    """Build a policy by short name.

    Supported names: heuristic | api | local | mock | model.
    """
    normalized = (name or "").strip().lower()
    if normalized == "api":
        return APIModelPolicy()
    if normalized == "local":
        return LocalModelPolicy()
    if normalized == "mock":
        return MockModelPolicy(memory=memory)
    if normalized == "model":
        from policies.router_model import RouterModelPolicy

        # HF custom → NIM → OpenRouter → mock (see forge/providers/router.py)
        return RouterModelPolicy(memory=memory, mode=forge_provider)
    return HeuristicPolicy(strategy=strategy)
