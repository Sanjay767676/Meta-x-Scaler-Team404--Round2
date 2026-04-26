"""FORGE-v4 adapter: wraps FORGEEnv with the official OpenEnv ``Environment`` ABC.

Install the framework: ``pip install openenv-core`` (import path ``openenv.core``).
This module does not replace ``FORGEEnv`` or the FastAPI server; it lets trainers
and tooling that expect ``openenv.core.Environment`` use the same logic.
"""

from __future__ import annotations

from typing import Any, Optional

from env import FORGEEnv
from memory import CoachMemory

try:
    from openenv.core import Environment
    from openenv.core.generic_client import GenericAction
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install OpenEnv: pip install openenv-core>=0.2.3"
    ) from exc


class FORGEOpenEnvironment(Environment[GenericAction, dict[str, Any], dict[str, Any]]):
    """Gym-style OpenEnv wrapper around :class:`FORGEEnv`."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, memory: CoachMemory | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._env = FORGEEnv(memory=memory)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # OpenEnv API allows seed/episode_id; core FORGE reproducibility uses config.GLOBAL_RANDOM_SEED.
        _ = (seed, episode_id, kwargs)
        return self._env.reset()

    def step(
        self,
        action: GenericAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        _ = timeout_s
        return self._env.step(dict(action))

    @property
    def state(self) -> dict[str, Any]:
        return self._env.get_state()
