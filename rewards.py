"""Reward functions for FORGE-v4 agents."""

from typing import Any

from config import (
    BREAKER_BREAK_REWARD,
    BREAKER_ERROR_BREAK_BONUS,
    BREAKER_FAIL_PENALTY,
    BREAKER_TIMEOUT_BREAK_BONUS,
    CODER_ERROR_PENALTY,
    CODER_FAIL_PENALTY,
    CODER_PASS_REWARD,
    CODER_PERFECT_RUN_BONUS,
    CODER_TIMEOUT_PENALTY,
)


def coder_reward(test_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute defender reward from hidden-test outcomes.

    Timeout outcomes are tracked separately from general runtime errors so logs
    and charts reflect where code quality is failing.
    """
    breakdown: list[float] = []
    pass_count = 0
    fail_count = 0
    error_count = 0
    timeout_count = 0

    for result in test_results:
        status = result.get("status", "error")
        if status == "pass":
            breakdown.append(CODER_PASS_REWARD)
            pass_count += 1
        elif status == "timeout":
            breakdown.append(CODER_TIMEOUT_PENALTY)
            timeout_count += 1
        elif status == "fail":
            breakdown.append(CODER_FAIL_PENALTY)
            fail_count += 1
        else:
            breakdown.append(CODER_ERROR_PENALTY)
            error_count += 1

    total_tests = len(test_results)
    pass_rate = pass_count / total_tests if total_tests else 0.0

    total_reward = sum(breakdown)
    if total_tests and pass_count == total_tests:
        total_reward += CODER_PERFECT_RUN_BONUS

    return {
        "total_reward": round(total_reward, 4),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "error_count": error_count,
        "timeout_count": timeout_count,
        "pass_rate": round(pass_rate, 4),
        "robustness_score": round(pass_rate - (timeout_count / max(total_tests, 1)) * 0.25, 4),
        "breakdown": [round(value, 4) for value in breakdown],
    }


def breaker_reward(adversarial_results: list[dict[str, Any]], coder_base_pass_rate: float) -> dict[str, Any]:
    """Compute adversary reward based on how effectively it breaks the coder."""
    breakdown: list[float] = []
    breaks = 0
    passes = 0
    timeout_breaks = 0
    error_breaks = 0

    # Breaking a stronger coder should be worth more.
    quality_multiplier = max(1.0, 1.0 + coder_base_pass_rate)

    for result in adversarial_results:
        status = result.get("status", "error")
        if status == "pass":
            passes += 1
            breakdown.append(BREAKER_FAIL_PENALTY)
            continue

        breaks += 1
        reward = BREAKER_BREAK_REWARD * quality_multiplier
        if status == "timeout":
            timeout_breaks += 1
            reward += BREAKER_TIMEOUT_BREAK_BONUS
        elif status == "error":
            error_breaks += 1
            reward += BREAKER_ERROR_BREAK_BONUS
        breakdown.append(reward)

    total_tests = len(adversarial_results)
    break_rate = breaks / total_tests if total_tests else 0.0
    total_reward = sum(breakdown)

    return {
        "total_reward": round(total_reward, 4),
        "breaks": breaks,
        "passes": passes,
        "timeout_breaks": timeout_breaks,
        "error_breaks": error_breaks,
        "break_rate": round(break_rate, 4),
        "breakdown": [round(value, 4) for value in breakdown],
    }
