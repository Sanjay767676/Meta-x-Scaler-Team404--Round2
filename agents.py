# agents.py
# Coder strategies and tiered Breaker agent for FORGE-v4.
#
# Coder strategies:
#   weak_coder_v1     — bubble sort (O(n²), slow on large arrays)
#   weak_coder_v2     — selection sort with a subtle bug on negatives
#   improving_coder   — picks strategy based on episode count
#
# Breaker agent:
#   BreakerAgent      — tiered adversarial test case generator

import random
from typing import Any
from config import (
    ARRAY_VALUE_RANGE,
    MAX_ARRAY_SIZE,
    BREAKER_TIER_UNLOCK_RATE,
    BREAKER_TIER3_MIN_EPISODE,
    BREAKER_TIER4_MIN_EPISODE,
    IMPROVING_CODER_TIER1_UNTIL,
    IMPROVING_CODER_TIER2_UNTIL,
)


# ══════════════════════════════════════════════
#  CODER STRATEGIES
# ══════════════════════════════════════════════

# Each strategy returns a Python source string that defines solution(arr).

WEAK_CODER_V1_CODE = '''
def solution(arr):
    """Bubble sort — fails on negatives and large arrays."""
    a = list(arr)
    n = len(a)
    for i in range(n):
        for j in range(n - i - 1):
            # BUG: Fails on negatives by using abs() for comparison
            if abs(a[j]) > abs(a[j + 1]):
                a[j], a[j + 1] = a[j + 1], a[j]
    # BUG: Fails on large arrays by returning truncated result
    if len(a) > 15: return a[:10]
    return a
'''

WEAK_CODER_V2_CODE = '''
def solution(arr):
    """
    Selection sort — fails on duplicates and large values.
    """
    # BUG: destroys duplicates
    a = list(set(arr)) 
    # BUG: fails on values > 50 by capping them
    a = [min(x, 50) for x in a]
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]
    return a
'''

WEAK_CODER_V3_CODE = '''
def solution(arr):
    """
    Boundary value bug — fails on zeros and mixed signs.
    """
    # BUG: removes all zeros
    # BUG: fails to handle mixed signs correctly (only returns positives)
    a = sorted([x for x in arr if x > 0]) 
    return a
'''

IMPROVING_CODER_TEMPLATE = '''
def solution(arr):
    """
    Improving coder — strategy selected by episode {episode}.
    Episode <= {tier1_until}: bubble sort (weakest)
    Episode <= {tier2_until}: selection sort (medium)
    Episode >  {tier2_until}: built-in sorted (strongest)
    """
    episode = {episode}
    a = list(arr)

    if episode <= {tier1_until}:
        # Bubble sort
        n = len(a)
        for i in range(n):
            for j in range(n - i - 1):
                if a[j] > a[j + 1]:
                    a[j], a[j + 1] = a[j + 1], a[j]
        return a
    elif episode <= {tier2_until}:
        # Selection sort with abs() bug
        n = len(a)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if abs(a[j]) < abs(a[min_idx]):
                    min_idx = j
            a[i], a[min_idx] = a[min_idx], a[i]
        return a
    else:
        # Strong solution
        return sorted(a)
'''


def get_coder_code(version: str, episode: int = 1) -> str:
    """
    Return the Python source code for the given coder version.

    Args:
        version: "weak_coder_v1" | "weak_coder_v2" | "weak_coder_v3" | "improving_coder"
        episode: current episode number (used by improving_coder)
    """
    if version == "weak_coder_v1":
        return WEAK_CODER_V1_CODE

    if version == "weak_coder_v2":
        return WEAK_CODER_V2_CODE

    if version == "weak_coder_v3":
        return WEAK_CODER_V3_CODE

    if version == "improving_coder":
        return IMPROVING_CODER_TEMPLATE.format(
            episode=episode,
            tier1_until=IMPROVING_CODER_TIER1_UNTIL,
            tier2_until=IMPROVING_CODER_TIER2_UNTIL,
        )

    raise ValueError(f"Unknown coder version: {version!r}")


def coder_version_label(version: str, episode: int) -> str:
    """Human-readable label for what strategy the coder is using this episode."""
    if version == "weak_coder_v1":
        return "weak_coder_v1 (bubble sort / abs-bug)"
    if version == "weak_coder_v2":
        return "weak_coder_v2 (selection sort / set-bug)"
    if version == "weak_coder_v3":
        return "weak_coder_v3 (filter-bug / boundary)"
    if version == "improving_coder":
        if episode <= IMPROVING_CODER_TIER1_UNTIL:
            return f"improving_coder → bubble sort  (ep {episode} ≤ {IMPROVING_CODER_TIER1_UNTIL})"
        if episode <= IMPROVING_CODER_TIER2_UNTIL:
            return f"improving_coder → selection sort (ep {episode} ≤ {IMPROVING_CODER_TIER2_UNTIL})"
        return f"improving_coder → sorted()  (ep {episode} > {IMPROVING_CODER_TIER2_UNTIL})"
    return version


# ══════════════════════════════════════════════
#  TIERED BREAKER AGENT
# ══════════════════════════════════════════════

# Test case banks per tier
_TIER1_CASES: list[list[int]] = [
    [10, 5, 8],
    [-1, -5, -2], # Negatives
    [1, 1, 1],    # Duplicates
    [0, 5, 2],    # Zeros (to trigger weak_coder_v3)
]

_TIER2_CASES: list[list[int]] = [
    [1, 1, 1, 1],                               # all duplicates
    [2, 2, 1, 1, 3, 3],                         # duplicate pairs
    [-5, -1, -3, -7, -2],                       # all negatives
    [-3, 0, 3, -1, 1],                          # mixed sign + zero
    [1, 2, 3, 4, 5],                            # already sorted
    [5, 4, 3, 2, 1],                            # reverse sorted
    [0, 0, 0],                                  # all zeros
]

_TIER3_CASES: list[list[int]] = [
    list(range(MAX_ARRAY_SIZE, 0, -1)),                      # full reverse
    [random.choice([1, 2]) for _ in range(MAX_ARRAY_SIZE)], # heavy duplicates
    [random.randint(-100, 100) for _ in range(MAX_ARRAY_SIZE)],  # large random
    [0] * MAX_ARRAY_SIZE,                                    # all zeros, large
    list(range(MAX_ARRAY_SIZE)),                             # sorted ascending, large
    [random.randint(-10, 10) for _ in range(MAX_ARRAY_SIZE)], # mixed, large
]

_TIER4_CASES: list[list[int]] = [
    [-100, 100],                                             # boundary values only
    [100, 100, 100, -100, -100, -100],                      # boundary duplicates
    [-100] * 10 + [100] * 10,                               # boundary mixed
    list(range(-10, 11)),                                    # full range small
    [random.randint(-100, 100) for _ in range(MAX_ARRAY_SIZE)],  # stress random
    [ARRAY_VALUE_RANGE[0], 0, ARRAY_VALUE_RANGE[1]] * 3,     # boundary/zero combo
]


class BreakerAgent:
    """
    Adversarial test-case generator with four tiers of difficulty.

    Tier unlocking rules:
        Tier 2 → always available from episode 1
        Tier 3 → unlocks when break_rate >= BREAKER_TIER_UNLOCK_RATE
                 AND episode >= BREAKER_TIER3_MIN_EPISODE
        Tier 4 → unlocks when at tier 3 AND episode >= BREAKER_TIER4_MIN_EPISODE

    The agent samples cases from all unlocked tiers, weighted toward the
    current (highest) tier for maximum adversarial pressure.
    """

    def __init__(self) -> None:
        self.current_tier: int = 1
        self._recent_break_rates: list[float] = []

    def update_tier(self, break_rate: float, episode: int) -> None:
        """
        Update the current tier based on recent performance and episode count.

        Args:
            break_rate: Breaker's break_rate from the last step.
            episode:    Current episode number.
        """
        self._recent_break_rates.append(break_rate)
        # Use rolling window of last 3 steps to smooth noise
        recent = self._recent_break_rates[-3:]
        avg_break = sum(recent) / len(recent)

        if self.current_tier == 1 and avg_break >= BREAKER_TIER_UNLOCK_RATE:
            self.current_tier = 2

        if self.current_tier == 2 and (
            avg_break >= BREAKER_TIER_UNLOCK_RATE
            and episode >= BREAKER_TIER3_MIN_EPISODE
        ):
            self.current_tier = 3

        if self.current_tier == 3 and episode >= BREAKER_TIER4_MIN_EPISODE:
            self.current_tier = 4

    def get_tests(self, n_per_tier: int = 2) -> list[dict[str, Any]]:
        """
        Return adversarial test cases sampled from all unlocked tiers,
        with extra weight on the current highest tier.

        Args:
            n_per_tier: Number of cases to sample from each unlocked tier.

        Returns:
            List of {"input": [...], "expected_output": [...]} dicts.
        """
        pools: list[tuple[int, list[list[int]]]] = [
            (1, _TIER1_CASES),
            (2, _TIER2_CASES),
            (3, _TIER3_CASES),
            (4, _TIER4_CASES),
        ]

        seen: set[tuple[int, ...]] = set()
        results: list[dict[str, Any]] = []
        for tier_num, pool in pools:
            if tier_num > self.current_tier:
                break
            
            # Sample more from the highest tier
            k = n_per_tier * 2 if tier_num == self.current_tier else n_per_tier
            k = min(k, len(pool))
            sampled = random.sample(pool, k)
            
            # Weighted scoring: higher tiers give significantly more points
            weight = 1.0 + (tier_num - 1) * 1.5 # Tier 1: 1.0, 2: 2.5, 3: 4.0, 4: 5.5
            
            for arr in sampled:
                key = tuple(arr)
                if key not in seen:
                    seen.add(key)
                    results.append({"input": arr, "expected_output": sorted(arr), "weight": weight})

        return results

    @property
    def tier_name(self) -> str:
        """Human-readable tier label."""
        from config import BREAKER_TIER_NAMES
        return BREAKER_TIER_NAMES.get(self.current_tier, f"Tier-{self.current_tier}")
