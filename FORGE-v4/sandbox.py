# sandbox.py
# Safely execute agent-generated Python code in a restricted subprocess.
# Returns structured pass/fail/error results with timeout handling.

import subprocess
import sys
import textwrap
import json
import os
from typing import Any
from config import SANDBOX_TIMEOUT_SECONDS, SANDBOX_MAX_OUTPUT_CHARS


def run_code(code: str, test_input: list[int]) -> dict[str, Any]:
    """
    Execute agent-generated code against a single test input.

    Args:
        code: Python source code that defines a `solution(arr)` function.
        test_input: The integer list to pass to `solution`.

    Returns:
        {
            "status":   "pass" | "fail" | "error" | "timeout",
            "output":   the value returned by solution(arr), or None,
            "expected": sorted(test_input),
            "error_msg": exception string if status is "error", else "",
        }
    """
    expected = sorted(test_input)

    # Build a self-contained runner script
    runner = textwrap.dedent(f"""
import json, sys

{code}

test_input = {test_input!r}
expected   = {expected!r}
try:
    result = solution(test_input)
    if result == expected:
        print(json.dumps({{"status": "pass", "output": result, "expected": expected, "error_msg": ""}}))
    else:
        print(json.dumps({{"status": "fail", "output": result, "expected": expected, "error_msg": ""}}))
except Exception as exc:
    print(json.dumps({{"status": "error", "output": None, "expected": expected, "error_msg": str(exc)}}))
""")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT_SECONDS,
        )
        raw = proc.stdout.strip()

        # Truncate excessive output
        if len(raw) > SANDBOX_MAX_OUTPUT_CHARS:
            raw = raw[:SANDBOX_MAX_OUTPUT_CHARS]

        if raw:
            result = json.loads(raw)
        else:
            # No stdout — treat stderr as the error message
            err = proc.stderr.strip()[:SANDBOX_MAX_OUTPUT_CHARS]
            result = {
                "status": "error",
                "output": None,
                "expected": expected,
                "error_msg": err or "No output produced.",
            }

    except subprocess.TimeoutExpired:
        result = {
            "status": "timeout",
            "output": None,
            "expected": expected,
            "error_msg": f"Code exceeded {SANDBOX_TIMEOUT_SECONDS}s timeout.",
        }
    except json.JSONDecodeError as exc:
        result = {
            "status": "error",
            "output": None,
            "expected": expected,
            "error_msg": f"JSON decode error: {exc}  raw='{raw}'",
        }

    return result


def run_code_against_tests(code: str, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Run agent code against a list of test cases.

    Args:
        code: Python source defining `solution(arr)`.
        tests: list of {"input": [...], "expected_output": [...]} dicts.

    Returns:
        List of result dicts, one per test.
    """
    results = []
    for test in tests:
        result = run_code(code, test["input"])
        results.append(result)
    return results
