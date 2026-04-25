"""Task generation utilities for FORGE-v4."""

import random
from typing import Any

from config import ARRAY_VALUE_RANGE, MAX_ARRAY_SIZE, MIN_ARRAY_SIZE, NUM_HIDDEN_TESTS


def generate_task() -> dict[str, Any]:
    """
    Generate a single sorting task.

    Returns a dict with:
        - prompt: natural-language task description
        - public_example: one visible (input, expected_output) pair
        - hidden_tests: list of (input, expected_output) pairs kept secret from agents
    """
    size = random.randint(MIN_ARRAY_SIZE, MAX_ARRAY_SIZE)
    arr = [random.randint(*ARRAY_VALUE_RANGE) for _ in range(size)]

    public_example = {
        "input": arr,
        "expected_output": sorted(arr),
    }

    hidden_tests = _generate_hidden_tests(NUM_HIDDEN_TESTS)

    task = {
        "prompt": (
            "Write a Python function named `solution(arr)` that takes a list of integers "
            "and returns a new list sorted in ascending order. "
            "Do not use `arr.sort()` in-place — return a new sorted list.\n\n"
            f"Example:\n  Input:  {arr}\n  Output: {sorted(arr)}"
        ),
        "public_example": public_example,
        "hidden_tests": hidden_tests,
    }
    return task


def _generate_hidden_tests(n: int) -> list[dict[str, Any]]:
    """Generate exactly n hidden tests with both random and edge-case coverage."""
    tests: list[dict[str, Any]] = []
    random_slots = max(0, n - 3)

    for _ in range(random_slots):
        size = random.randint(MIN_ARRAY_SIZE, MAX_ARRAY_SIZE)
        arr = [random.randint(*ARRAY_VALUE_RANGE) for _ in range(size)]
        tests.append({"input": arr, "expected_output": sorted(arr)})

    # Edge case: already-sorted array
    arr = sorted([random.randint(*ARRAY_VALUE_RANGE) for _ in range(5)])
    tests.append({"input": arr, "expected_output": sorted(arr)})

    # Edge case: reverse-sorted array
    arr = sorted([random.randint(*ARRAY_VALUE_RANGE) for _ in range(5)], reverse=True)
    tests.append({"input": arr, "expected_output": sorted(arr)})

    # Edge case: single element
    arr = [random.randint(*ARRAY_VALUE_RANGE)]
    tests.append({"input": arr, "expected_output": sorted(arr)})

    return tests[:n]


def generate_breaker_task(original_task: dict[str, Any]) -> dict[str, Any]:
    """
    Given an existing task, produce adversarial test cases for the Breaker agent.

    The Breaker is asked to produce arrays that are likely to break a naive solution.
    Returns a dict with the adversarial prompt and a set of candidate adversarial arrays.
    """
    adversarial_candidates = [
        # All identical elements
        [0] * random.randint(3, 8),
        # All negative values
        [random.randint(-100, -1) for _ in range(random.randint(3, 8))],
        # Large array
        [random.randint(*ARRAY_VALUE_RANGE) for _ in range(MAX_ARRAY_SIZE)],
        # Duplicate-heavy array
        [random.choice([1, 2, 3]) for _ in range(random.randint(4, 10))],
        # Mixed positive/negative with duplicates
        [random.randint(-5, 5) for _ in range(random.randint(4, 12))],
    ]

    adversarial_tests = [
        {"input": arr, "expected_output": sorted(arr)}
        for arr in adversarial_candidates
    ]

    breaker_task = {
        "prompt": (
            "You are the Breaker agent. Generate adversarial integer arrays that are "
            "likely to expose flaws in a naive sorting implementation. "
            "Focus on edge cases: duplicates, negatives, large inputs, already-sorted, "
            "reverse-sorted, and single-element arrays."
        ),
        "adversarial_tests": adversarial_tests,
    }
    return breaker_task
