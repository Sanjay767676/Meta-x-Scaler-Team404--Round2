"""Shared LLM response type for FORGE inference providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Normalized response from any inference backend."""

    provider: str
    model: str
    content: str
    raw: dict[str, Any] = field(default_factory=dict)
