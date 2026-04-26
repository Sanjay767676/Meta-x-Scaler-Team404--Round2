from typing import Any
from memory import CoachMemory
from llm_agent import get_provider
from policies.base import CodeCandidate, CoderPolicy


class MockModelPolicy(CoderPolicy):
    """
    Simulates an improving LLM policy that reads memory to handle edge cases.
    """

    name = "offline"

    def __init__(self, memory: CoachMemory | None = None) -> None:
        self.provider = get_provider("offline")
        self.memory = memory or CoachMemory()

    def generate_candidates(self, state: dict[str, Any], num_candidates: int) -> list[CodeCandidate]:
        """
        Generates candidates based on memory of past failures.
        """
        # 1. Read memory for failures
        summary = self.memory.summary()
        recent_notes = summary.get("recent_coach_notes", [])
        
        # Analyze failures
        has_neg_fail = any("negative" in n.lower() for n in recent_notes)
        has_dup_fail = any("duplicate" in n.lower() for n in recent_notes)
        has_large_fail = any("large" in n.lower() or "timeout" in n.lower() for n in recent_notes)

        candidates: list[CodeCandidate] = []
        n = max(3, num_candidates) # Default to 3 candidates for better realism

        for i in range(n):
            # Simulate different strategies based on memory
            if has_neg_fail or has_dup_fail or has_large_fail:
                # Memory-informed "Strong" strategy
                code = (
                    "def solution(arr):\n"
                    "    # MEMORY-INFORMED: Handling negatives, duplicates, and efficiency\n"
                    "    # Uses Python's Timsort which is O(n log n) and handles all edge cases.\n"
                    "    if not isinstance(arr, list): return []\n"
                    "    return sorted(list(arr))\n"
                )
                source = "baseline:model:robust"
            else:
                # Default "Naive" strategy for early episodes
                if i == 0:
                    # One slightly buggy candidate to simulate "thinking"
                    code = (
                        "def solution(arr):\n"
                        "    # Naive quicksort (potentially buggy on recursion depth)\n"
                        "    if not arr: return []\n"
                        "    pivot = arr[0]\n"
                        "    less = [x for x in arr[1:] if x < pivot]\n"
                        "    greater = [x for x in arr[1:] if x >= pivot]\n"
                        "    return solution(less) + [pivot] + solution(greater)\n"
                    )
                    source = "baseline:model:naive"
                else:
                    code = (
                        "def solution(arr):\n"
                        "    return sorted(list(arr))\n"
                    )
                    source = "baseline:model:standard"

            candidates.append(
                CodeCandidate(
                    code=code,
                    source=source,
                    metadata={"candidate_idx": i, "memory_aware": True}
                )
            )

        return candidates
