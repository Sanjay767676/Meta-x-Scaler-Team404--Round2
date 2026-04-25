"""Candidate evaluation service for multi-sample defender generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sandbox import run_code_against_tests


@dataclass
class CandidateScore:
    """Scored candidate details used for ranking and logging."""

    index: int
    code: str
    pass_count: int
    total_tests: int
    pass_rate: float
    avg_runtime_ms: float
    timeout_count: int
    error_count: int
    fail_count: int
    robustness: float
    code_quality: float
    composite_score: float


def _code_quality_score(code: str) -> float:
    """Simple static heuristic used for tie-breaking and reportability."""
    quality = 0.0
    stripped = code.strip()
    if "def solution(" in stripped:
        quality += 0.4
    if '"""' in stripped or "'''" in stripped:
        quality += 0.2
    line_count = len([line for line in stripped.splitlines() if line.strip()])
    if 3 <= line_count <= 80:
        quality += 0.2
    if "sorted(" in stripped or "for " in stripped:
        quality += 0.2
    return round(min(1.0, quality), 4)


def _score_single_candidate(index: int, code: str, tests: list[dict[str, Any]]) -> CandidateScore:
    results = run_code_against_tests(code=code, tests=tests)
    total = len(results)
    pass_count = sum(1 for r in results if r.get("status") == "pass")
    timeout_count = sum(1 for r in results if r.get("status") == "timeout")
    error_count = sum(1 for r in results if r.get("status") == "error")
    fail_count = sum(1 for r in results if r.get("status") == "fail")

    runtime_values = [float(r.get("execution_ms", 0.0)) for r in results]
    avg_runtime = sum(runtime_values) / len(runtime_values) if runtime_values else 0.0
    pass_rate = pass_count / total if total else 0.0

    robustness = pass_rate - (timeout_count / max(total, 1)) * 0.5 - (error_count / max(total, 1)) * 0.25
    quality = _code_quality_score(code)

    # Lower runtime is better, normalized against 1s baseline.
    runtime_component = max(0.0, 1.0 - min(avg_runtime, 1000.0) / 1000.0)

    composite = 0.6 * pass_rate + 0.2 * max(0.0, robustness) + 0.1 * quality + 0.1 * runtime_component

    return CandidateScore(
        index=index,
        code=code,
        pass_count=pass_count,
        total_tests=total,
        pass_rate=round(pass_rate, 4),
        avg_runtime_ms=round(avg_runtime, 2),
        timeout_count=timeout_count,
        error_count=error_count,
        fail_count=fail_count,
        robustness=round(robustness, 4),
        code_quality=quality,
        composite_score=round(composite, 4),
    )


def evaluate_candidates(candidates: list[str], tests: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate and rank candidates, returning winner and ranking table."""
    if not candidates:
        return {
            "best_code": "",
            "best_index": -1,
            "rankings": [],
            "selection_reason": "no_candidates",
        }

    scored = [_score_single_candidate(idx, code, tests) for idx, code in enumerate(candidates)]
    ranked = sorted(
        scored,
        key=lambda s: (s.composite_score, s.pass_count, -s.avg_runtime_ms, s.code_quality),
        reverse=True,
    )
    best = ranked[0]

    return {
        "best_code": best.code,
        "best_index": best.index,
        "selection_reason": "max_composite_score",
        "rankings": [
            {
                "index": item.index,
                "pass_count": item.pass_count,
                "total_tests": item.total_tests,
                "pass_rate": item.pass_rate,
                "avg_runtime_ms": item.avg_runtime_ms,
                "timeout_count": item.timeout_count,
                "error_count": item.error_count,
                "fail_count": item.fail_count,
                "robustness": item.robustness,
                "code_quality": item.code_quality,
                "composite_score": item.composite_score,
            }
            for item in ranked
        ],
    }
