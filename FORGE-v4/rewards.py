# rewards.py
# Reward functions for the Coder Agent and the Breaker Agent in FORGE-v4.

from typing import Any
from config import (
    CODER_PASS_REWARD,
    CODER_FAIL_PENALTY,
    CODER_ERROR_PENALTY,
    BREAKER_BREAK_REWARD,
    BREAKER_FAIL_PENALTY,
)


def coder_reward(test_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute the Coder agent's reward from sandbox test results.

    Args:
        test_results: list of result dicts from sandbox.run_code_against_tests().
            Each dict has a "status" key: "pass" | "fail" | "error" | "timeout".

    Returns:
        {
            "total_reward": float,
            "pass_count":   int,
            "fail_count":   int,
            "error_count":  int,
            "pass_rate":    float,   # fraction of tests passed
            "breakdown":    list of per-test reward floats,
        }
    """
    breakdown = []
    pass_count = fail_count = error_count = 0

    for r in test_results:
        status = r.get("status", "error")
        if status == "pass":
            breakdown.append(CODER_PASS_REWARD)
            pass_count += 1
        elif status in ("error", "timeout"):
            breakdown.append(CODER_ERROR_PENALTY)
            error_count += 1
        else:  # "fail"
            breakdown.append(CODER_FAIL_PENALTY)
            fail_count += 1

    total = sum(breakdown)
    n = len(test_results)
    pass_rate = pass_count / n if n > 0 else 0.0

    return {
        "total_reward": round(total, 4),
        "pass_count":   pass_count,
        "fail_count":   fail_count,
        "error_count":  error_count,
        "pass_rate":    round(pass_rate, 4),
        "breakdown":    breakdown,
    }


def breaker_reward(
    adversarial_results: list[dict[str, Any]],
    coder_base_pass_rate: float,
) -> dict[str, Any]:
    """
    Compute the Breaker agent's reward.

    The Breaker earns credit for tests that break the coder (non-pass outcomes).
    It is penalised for tests that the coder still passes, because those tests
    are not adversarial enough.

    Args:
        adversarial_results: results when the coder's code is run against the
                             Breaker's adversarial test cases.
        coder_base_pass_rate: the coder's pass-rate on the standard hidden tests
                              (used to scale the Breaker's reward — breaking a
                              strong coder is worth more).

    Returns:
        {
            "total_reward": float,
            "breaks":       int,   # number of tests that broke the coder
            "passes":       int,   # number of tests the coder still passed
            "break_rate":   float,
            "breakdown":    list of per-test reward floats,
        }
    """
    breakdown = []
    breaks = passes = 0

    # A higher-quality coder means a bigger multiplier for breaking them
    quality_multiplier = max(1.0, 1.0 + coder_base_pass_rate)

    for r in adversarial_results:
        status = r.get("status", "error")
        if status != "pass":
            # Breaker successfully broke the coder
            reward = BREAKER_BREAK_REWARD * quality_multiplier
            breakdown.append(round(reward, 4))
            breaks += 1
        else:
            # Coder survived — penalise the Breaker
            breakdown.append(BREAKER_FAIL_PENALTY)
            passes += 1

    total = sum(breakdown)
    n = len(adversarial_results)
    break_rate = breaks / n if n > 0 else 0.0

    return {
        "total_reward": round(total, 4),
        "breaks":       breaks,
        "passes":       passes,
        "break_rate":   round(break_rate, 4),
        "breakdown":    breakdown,
    }
