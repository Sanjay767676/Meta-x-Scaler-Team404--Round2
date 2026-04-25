# env.py
# Main OpenEnv-style reinforcement learning environment for FORGE-v4.
# Manages the interaction between the Coder Agent, Breaker Agent, and Sandbox.

from typing import Any
from tasks import generate_task, generate_breaker_task
from sandbox import run_code_against_tests
from rewards import coder_reward, breaker_reward
from memory import CoachMemory
from config import STEPS_PER_EPISODE


class FORGEEnv:
    """
    Two-agent adversarial environment for code generation tasks.

    Agents:
        - Coder:   writes Python code to solve array-sorting tasks.
        - Breaker: generates adversarial test cases to break the Coder's solution.

    Episode flow:
        1. reset()           → returns the initial task state
        2. step(action)      × STEPS_PER_EPISODE steps
        3. Rewards assigned to both agents at each step

    Action format:
        {
            "coder_code":        str | None,   # Python source defining solution(arr)
            "breaker_tests":     list | None,  # List of {"input": [...]} dicts
        }
    """

    def __init__(self, memory: CoachMemory | None = None):
        self.memory = memory or CoachMemory()
        self.episode: int = 0
        self.step_count: int = 0
        self.current_task: dict[str, Any] = {}
        self.done: bool = True
        self._last_coder_code: str = ""
        self._last_coder_pass_rate: float = 0.0

    # ──────────────────────────────────────────────
    # Core env methods
    # ──────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        """
        Start a new episode.

        Returns:
            Initial state dict containing the task prompt and public example.
        """
        self.episode += 1
        self.step_count = 0
        self.done = False
        self._last_coder_code = ""
        self._last_coder_pass_rate = 0.0

        self.current_task = generate_task()

        state = self.get_state()
        return state

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Advance the environment by one step.

        Args:
            action: dict with optional keys:
                "coder_code"    – Python source defining solution(arr)
                "breaker_tests" – list of {"input": [...]} dicts

        Returns:
            {
                "state":          current env state,
                "coder_reward":   coder reward info dict,
                "breaker_reward": breaker reward info dict,
                "done":           bool (True when episode ends),
                "info":           extra diagnostics,
            }
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        self.step_count += 1
        coder_code    = action.get("coder_code", "")
        breaker_tests = action.get("breaker_tests", [])

        # ── Evaluate Coder ────────────────────────────────────────────────
        coder_info = self._evaluate_coder(coder_code)

        # ── Evaluate Breaker ──────────────────────────────────────────────
        breaker_info = self._evaluate_breaker(coder_code, breaker_tests, coder_info)

        # ── Log to Coach Memory ───────────────────────────────────────────
        self.memory.add_lesson(
            episode=self.episode,
            agent="env",
            observation=(
                f"Step {self.step_count}: "
                f"coder pass_rate={coder_info['pass_rate']:.2f}, "
                f"breaker break_rate={breaker_info['break_rate']:.2f}"
            ),
            coder_reward=coder_info["total_reward"],
            breaker_reward=breaker_info["total_reward"],
            extra={
                "step": self.step_count,
                "coder_pass_rate": coder_info["pass_rate"],
                "breaker_break_rate": breaker_info["break_rate"],
            },
        )

        # ── Check done ────────────────────────────────────────────────────
        if self.step_count >= STEPS_PER_EPISODE:
            self.done = True

        return {
            "state":          self.get_state(),
            "coder_reward":   coder_info,
            "breaker_reward": breaker_info,
            "done":           self.done,
            "info": {
                "episode":    self.episode,
                "step":       self.step_count,
            },
        }

    def get_state(self) -> dict[str, Any]:
        """
        Return the current observable state of the environment.
        """
        return {
            "episode":        self.episode,
            "step":           self.step_count,
            "done":           self.done,
            "task_prompt":    self.current_task.get("prompt", ""),
            "public_example": self.current_task.get("public_example", {}),
            "last_pass_rate": self._last_coder_pass_rate,
        }

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _evaluate_coder(self, code: str) -> dict[str, Any]:
        """Run the coder's code against hidden tests and compute reward."""
        hidden_tests = self.current_task.get("hidden_tests", [])

        if not code or not hidden_tests:
            # No code submitted — max penalty
            dummy_results = [{"status": "error"} for _ in hidden_tests or [{}]]
            info = coder_reward(dummy_results)
        else:
            results = run_code_against_tests(code, hidden_tests)
            info = coder_reward(results)

        # Cache for Breaker quality multiplier
        self._last_coder_code = code
        self._last_coder_pass_rate = info["pass_rate"]
        return info

    def _evaluate_breaker(
        self,
        coder_code: str,
        breaker_tests: list[dict[str, Any]],
        coder_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the coder's code against the breaker's adversarial tests."""
        if not coder_code or not breaker_tests:
            # No submission from one of the agents
            dummy = [{"status": "pass"} for _ in breaker_tests or [{}]]
            return breaker_reward(dummy, coder_base_pass_rate=coder_info["pass_rate"])

        results = run_code_against_tests(coder_code, breaker_tests)
        return breaker_reward(results, coder_base_pass_rate=coder_info["pass_rate"])
