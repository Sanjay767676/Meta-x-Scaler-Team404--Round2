"""Sandbox runner for evaluating generated Python solutions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any

from config import (
    SANDBOX_MAX_CODE_CHARS,
    SANDBOX_MAX_OUTPUT_CHARS,
    SANDBOX_MEMORY_LIMIT_MB,
    SANDBOX_TIMEOUT_SECONDS,
)

_RESULT_START = "__FORGE_RESULT_START__"
_RESULT_END = "__FORGE_RESULT_END__"


def _build_runner_script(code: str, test_input: list[int], expected: list[int]) -> str:
    """Wrap candidate code with a guarded evaluator script."""
    return textwrap.dedent(
        f"""
import json

try:
    import resource  # Unix only
except Exception:
    resource = None

if resource is not None:
    try:
        limit_bytes = int({SANDBOX_MEMORY_LIMIT_MB}) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception:
        pass

{code}

test_input = {test_input!r}
expected = {expected!r}

payload = {{
    "status": "error",
    "output": None,
    "expected": expected,
    "error_msg": "Unknown execution failure",
}}

try:
    if "solution" not in globals() or not callable(solution):
        raise NameError("Generated code must define callable solution(arr)")
    output = solution(test_input)
    payload["output"] = output
    payload["status"] = "pass" if output == expected else "fail"
    payload["error_msg"] = ""
except Exception as exc:
    payload["status"] = "error"
    payload["error_msg"] = f"{{type(exc).__name__}}: {{exc}}"

print("{_RESULT_START}")
print(json.dumps(payload))
print("{_RESULT_END}")
"""
    )


def _safe_text(value: str) -> str:
    """Normalize and cap potentially large process output."""
    cleaned = value.replace("\x00", "")
    if len(cleaned) > SANDBOX_MAX_OUTPUT_CHARS:
        return cleaned[:SANDBOX_MAX_OUTPUT_CHARS]
    return cleaned


def _extract_json_payload(stdout: str) -> dict[str, Any] | None:
    """Extract the framed JSON payload from potentially noisy stdout."""
    start_idx = stdout.rfind(_RESULT_START)
    end_idx = stdout.rfind(_RESULT_END)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None
    json_text = stdout[start_idx + len(_RESULT_START) : end_idx].strip()
    if not json_text:
        return None
    return json.loads(json_text)


def run_code(code: str, test_input: list[int], expected_output: list[int] | None = None) -> dict[str, Any]:
    """Execute generated code against a single test input."""
    expected = sorted(test_input) if expected_output is None else expected_output

    if not code.strip():
        return {
            "status": "error",
            "output": None,
            "expected": expected,
            "error_msg": "Empty coder code.",
            "error_type": "validation",
            "execution_ms": 0,
        }
    if len(code) > SANDBOX_MAX_CODE_CHARS:
        return {
            "status": "error",
            "output": None,
            "expected": expected,
            "error_msg": f"Code exceeds SANDBOX_MAX_CODE_CHARS={SANDBOX_MAX_CODE_CHARS}.",
            "error_type": "validation",
            "execution_ms": 0,
        }

    runner = _build_runner_script(code=code, test_input=test_input, expected=expected)
    started_at = time.perf_counter()

    temp_script_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_forge_runner.py", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(runner)
            temp_script_path = temp_file.name

        proc = subprocess.run(
            [sys.executable, "-I", temp_script_path],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            cwd=os.getcwd(),
        )
        execution_ms = int((time.perf_counter() - started_at) * 1000)

        stdout = _safe_text(proc.stdout)
        stderr = _safe_text(proc.stderr)

        payload = _extract_json_payload(stdout)
        if payload is None:
            return {
                "status": "error",
                "output": None,
                "expected": expected,
                "error_msg": (stderr or stdout or "No structured payload returned.")[:SANDBOX_MAX_OUTPUT_CHARS],
                "error_type": "unstructured_output",
                "execution_ms": execution_ms,
            }

        payload.setdefault("expected", expected)
        payload.setdefault("output", None)
        payload.setdefault("error_msg", "")
        payload["execution_ms"] = execution_ms
        payload["error_type"] = "" if payload.get("status") in ("pass", "fail") else "runtime"
        return payload
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "output": None,
            "expected": expected,
            "error_msg": f"Code exceeded {SANDBOX_TIMEOUT_SECONDS}s timeout.",
            "error_type": "timeout",
            "execution_ms": int((time.perf_counter() - started_at) * 1000),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "output": None,
            "expected": expected,
            "error_msg": f"Sandbox failure: {type(exc).__name__}: {exc}",
            "error_type": "sandbox",
            "execution_ms": int((time.perf_counter() - started_at) * 1000),
        }
    finally:
        if temp_script_path and os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
            except OSError:
                pass


def run_code_against_tests(code: str, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run generated code against a list of test dictionaries."""
    results: list[dict[str, Any]] = []
    for test in tests:
        test_input = list(test.get("input", []))
        expected = test.get("expected_output")
        result = run_code(code=code, test_input=test_input, expected_output=expected)
        # Preserve weight if present
        if "weight" in test:
            result["weight"] = test["weight"]
        results.append(result)
    return results
