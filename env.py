"""Core OpenEnv-like environment for FORGE-v4."""

from __future__ import annotations

import uuid
from typing import Any

from agents import BreakerAgent
import config
from logger import log_step
from memory import CoachMemory
from rewards import breaker_reward, coder_reward
from services.candidate_evaluator import evaluate_candidates
from sandbox import run_code_against_tests
from tasks import generate_task


class FORGEEnv:
    """
    Two-agent adversarial environment for code generation tasks.

    Agents:
        - Coder:   submits Python code defining solution(arr).
        - Breaker: submits adversarial test cases via a BreakerAgent.

    Episode flow:
        1. reset()               → returns initial state
        2. step(action) × N     → coder vs breaker, rewards, memory, logs
        3. done=True             → call reset() for next episode

    Action format passed to step():
        {
            "coder_code":    str,   # Python source defining solution(arr)
            "coder_version": str,   # label, e.g. "weak_coder_v1"
        }
    The BreakerAgent is managed internally by the environment.

    State returned by get_state() / reset() / step():
        {
            "task_id":              str,
            "problem_description":  str,
            "episode":              int,
            "episode_step":         int,
            "done":                 bool,
            "coder_version":        str,
            "current_tier":         int,
            "recent_breaker_case":  list[int],
            "pass_rate_history":    list[float],
            "coach_memory_summary": dict,
            "public_example":       dict,
        }

    step() returns:
        {
            "state":          dict,
            "coder_reward":   dict,   # from rewards.coder_reward()
            "breaker_reward": dict,   # from rewards.breaker_reward()
            "done":           bool,
            "info":           dict,   # diagnostics
        }

    Explicit step() flow:
        1. Run coder code against hidden tests in sandbox
        2. Run breaker tests against coder code in sandbox
        3. Assign coder_reward and breaker_reward
        4. Update coach memory with structured lesson
        5. Log step metrics to logs/rewards.json
        6. Advance breaker tier based on break_rate
        7. Return next_state, rewards, done, info
    """

    def __init__(self, memory: CoachMemory | None = None):
        self.memory = memory or CoachMemory()
        self.breaker = BreakerAgent()
        self.episode: int = 0
        self.step_count: int = 0
        self.current_task: dict[str, Any] = {}
        self.done: bool = True

        # Caching to avoid redundant sandbox runs
        self._eval_cache: dict[str, dict[str, Any]] = {}

        # Tracked across the episode
        self._coder_version: str = "unknown"
        self._pass_rate_history: list[float] = []
        self._recent_breaker_case: list[int] = []
        self._last_coder_pass_rate: float = 0.0
        self._last_timeout_count: int = 0
        self._last_candidate_rankings: list[dict[str, Any]] = []

    # ──────────────────────────────────────────────
    # Core env methods
    # ──────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        """
        Start a new episode. Generates a fresh task and resets counters.
        """
        self.episode += 1
        self.step_count = 0
        self.done = False

        self._eval_cache = {} # Clear cache for new task
        self._coder_version = "unknown"
        self._pass_rate_history = []
        self._recent_breaker_case = []
        self._last_coder_pass_rate = 0.0
        self._last_timeout_count = 0
        self._last_candidate_rankings = []

        self.current_task = generate_task()
        self.current_task["task_id"] = str(uuid.uuid4())[:8]

        return self.get_state()

    def _validate_action(self, action: dict[str, Any]) -> tuple[str, str, list[str]]:
        """Validate action and normalize coder fields and optional candidates."""
        if not isinstance(action, dict):
            raise TypeError("Action must be a dict with coder_code and coder_version.")

        coder_code = action.get("coder_code", "")
        coder_version = action.get("coder_version", "unknown")
        candidate_solutions = action.get("candidate_solutions") or []
        if not isinstance(coder_code, str):
            raise TypeError("action['coder_code'] must be a string.")
        if not isinstance(coder_version, str):
            raise TypeError("action['coder_version'] must be a string.")
        if not isinstance(candidate_solutions, list):
            raise TypeError("action['candidate_solutions'] must be a list[str].")
        normalized_candidates: list[str] = []
        for candidate in candidate_solutions:
            if isinstance(candidate, str) and candidate.strip():
                normalized_candidates.append(candidate)
        return coder_code, coder_version, normalized_candidates

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Advance the environment by one step."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        self.step_count += 1
        coder_code, coder_version, candidate_solutions = self._validate_action(action)
        self._coder_version = coder_version

        # ── 1. Get breaker tests for this step ───────────────────────────
        breaker_tests = self.breaker.get_tests(n_per_tier=2)
        if breaker_tests:
            self._recent_breaker_case = breaker_tests[-1]["input"]

        hidden_tests = self.current_task.get("hidden_tests", [])
        if candidate_solutions:
            candidate_eval = evaluate_candidates(candidates=candidate_solutions, tests=hidden_tests)
            self._last_candidate_rankings = candidate_eval["rankings"]
            selected_code = candidate_eval["best_code"] or coder_code
            selected_candidate_index = candidate_eval.get("best_index", -1)
            selection_reason = candidate_eval.get("selection_reason", "unknown")
        else:
            self._last_candidate_rankings = []
            selected_code = coder_code
            selected_candidate_index = -1
            selection_reason = "single_candidate"

        # ── 2 & 3. Run sandbox + compute rewards ──────────────────────────
        coder_info = self._evaluate_coder(selected_code)
        breaker_info = self._evaluate_breaker(selected_code, breaker_tests, coder_info)

        self._pass_rate_history.append(coder_info["pass_rate"])
        self._last_coder_pass_rate = coder_info["pass_rate"]
        self._last_timeout_count = coder_info["timeout_count"]

        # ── 4. Update coach memory ────────────────────────────────────────
        self.memory.add_lesson(
            episode=self.episode,
            agent="env",
            observation=(
                f"Step {self.step_count}: "
                f"coder={coder_version}, "
                f"pass_rate={coder_info['pass_rate']:.2f}, "
                f"breaker_tier={self.breaker.current_tier}, "
                f"break_rate={breaker_info['break_rate']:.2f}"
            ),
            coder_reward=coder_info["total_reward"],
            breaker_reward=breaker_info["total_reward"],
            extra={
                "task_id":             self.current_task.get("task_id", ""),
                "problem_description": self.current_task.get("prompt", ""),
                "step":                self.step_count,
                "coder_version":       coder_version,
                "breaker_tier":        self.breaker.current_tier,
                "coder_pass_rate":     coder_info["pass_rate"],
                "fail_count":          coder_info["fail_count"],
                "error_count":         coder_info["error_count"],
                "timeout_count":       coder_info["timeout_count"],
                "breaker_break_rate":  breaker_info["break_rate"],
                "recent_breaker_case": self._recent_breaker_case,
                "candidate_count":     len(candidate_solutions),
                "candidate_rankings":  self._last_candidate_rankings,
            },
        )

        # ── 5. Log step metrics ───────────────────────────────────────────
        log_step(
            episode=self.episode,
            step=self.step_count,
            coder_version=coder_version,
            breaker_tier=self.breaker.current_tier,
            coder_reward=coder_info["total_reward"],
            breaker_reward=breaker_info["total_reward"],
            pass_rate=coder_info["pass_rate"],
            fail_count=coder_info["fail_count"],
            error_count=coder_info["error_count"],
            timeout_count=coder_info["timeout_count"],
            break_rate=breaker_info["break_rate"],
        )

        # ── 6. Advance breaker tier ────────────────────────────────────────
        self.breaker.update_tier(breaker_info["break_rate"], coder_info["pass_rate"], self.episode)

        # ── 7. Check done + return ────────────────────────────────────────
        if self.step_count >= config.STEPS_PER_EPISODE:
            self.done = True

        return {
            "state": self.get_state(),
            "coder_reward": coder_info,
            "breaker_reward": breaker_info,
            "done": self.done,
            "info": {
                "episode": self.episode,
                "step": self.step_count,
                "coder_version": coder_version,
                "breaker_tier": self.breaker.current_tier,
                "breaker_tier_name": self.breaker.tier_name,
                "timeout_count": coder_info["timeout_count"],
                "candidate_rankings": self._last_candidate_rankings,
                "selected_candidate_index": selected_candidate_index,
                "selection_reason": selection_reason,
            },
        }

    def get_state(self) -> dict[str, Any]:
        """Return the current observable state of the environment."""
        return {
            "task_id": self.current_task.get("task_id", ""),
            "problem_description": self.current_task.get("prompt", ""),
            "episode": self.episode,
            "episode_step": self.step_count,
            "done": self.done,
            "coder_version": self._coder_version,
            "current_tier": self.breaker.current_tier,
            "recent_breaker_case": self._recent_breaker_case,
            "pass_rate_history": list(self._pass_rate_history),
            "last_pass_rate": self._last_coder_pass_rate,
            "last_timeout_count": self._last_timeout_count,
            "last_candidate_rankings": self._last_candidate_rankings,
            "coach_memory_summary": self.memory.summary(),
            "public_example": self.current_task.get("public_example", {}),
        }

    def _evaluate_coder(self, code: str) -> dict[str, Any]:
        """Run the coder's code against hidden tests and compute reward."""
        if code in self._eval_cache:
            return self._eval_cache[code]

        hidden_tests = self.current_task.get("hidden_tests", [])
        if not code or not hidden_tests:
            dummy = [{"status": "error"} for _ in hidden_tests or [{}]]
            return coder_reward(dummy)

        results = run_code_against_tests(code, hidden_tests)
        info = coder_reward(results)
        
        # Cache the result
        self._eval_cache[code] = info
        return info

    def _evaluate_breaker(
        self,
        coder_code: str,
        breaker_tests: list[dict[str, Any]],
        coder_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the coder's code against breaker adversarial tests."""
        # Breaker tests change every step, so we don't cache this as easily
        if not coder_code or not breaker_tests:
            dummy = [{"status": "pass"} for _ in breaker_tests or [{}]]
            return breaker_reward(dummy, coder_base_pass_rate=coder_info["pass_rate"])

        results = run_code_against_tests(coder_code, breaker_tests)
        return breaker_reward(results, coder_base_pass_rate=coder_info["pass_rate"])
