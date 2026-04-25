"""Sandbox runner for evaluating generated Python solutions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from config import (
    SANDBOX_MAX_CODE_CHARS,
    SANDBOX_MAX_OUTPUT_CHARS,
    SANDBOX_MEMORY_LIMIT_MB,
    SANDBOX_TIMEOUT_SECONDS,
)

_RESULT_START = "__FORGE_RESULT_START__"
_RESULT_END = "__FORGE_RESULT_END__"


def _build_batch_runner_script(code: str, test_cases: list[dict[str, Any]]) -> str:
    """Wrap candidate code with a loop that evaluates multiple test cases in one process."""
    return textwrap.dedent(
        f"""
import json
import time

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

# Candidate code
{code}

test_cases = {test_cases!r}
results = []

for idx, test in enumerate(test_cases):
    test_input = test.get("input", [])
    expected = test.get("expected_output")
    if expected is None:
        expected = sorted(test_input)
    
    payload = {{
        "status": "error",
        "output": None,
        "expected": expected,
        "error_msg": "Unknown execution failure",
        "weight": test.get("weight", 1.0)
    }}
    
    try:
        start_t = time.perf_counter()
        if "solution" not in globals() or not callable(solution):
            raise NameError("Generated code must define callable solution(arr)")
        output = solution(test_input)
        end_t = time.perf_counter()
        
        payload["output"] = output
        payload["status"] = "pass" if output == expected else "fail"
        payload["error_msg"] = ""
        payload["execution_ms"] = int((end_t - start_t) * 1000)
    except Exception as exc:
        payload["status"] = "error"
        payload["error_msg"] = f"{{type(exc).__name__}}: {{exc}}"
        payload["execution_ms"] = 0
    
    results.append(payload)

print("{_RESULT_START}")
print(json.dumps(results))
print("{_RESULT_END}")
"""
    )


def _safe_text(value: str) -> str:
    """Normalize and cap potentially large process output."""
    cleaned = value.replace("\x00", "")
    if len(cleaned) > SANDBOX_MAX_OUTPUT_CHARS:
        return cleaned[:SANDBOX_MAX_OUTPUT_CHARS]
    return cleaned


def _extract_batch_payload(stdout: str) -> list[dict[str, Any]] | None:
    """Extract the framed JSON list of results from potentially noisy stdout."""
    start_idx = stdout.rfind(_RESULT_START)
    end_idx = stdout.rfind(_RESULT_END)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None
    json_text = stdout[start_idx + len(_RESULT_START) : end_idx].strip()
    if not json_text:
        return None
    try:
        data = json.loads(json_text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def run_code_against_tests(code: str, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run generated code against a list of tests in a single subprocess."""
    if not code.strip():
        return [{"status": "error", "error_msg": "Empty code", "weight": t.get("weight", 1.0)} for t in tests]
    
    if len(code) > SANDBOX_MAX_CODE_CHARS:
        return [{"status": "error", "error_msg": "Code too large", "weight": t.get("weight", 1.0)} for t in tests]

    runner = _build_batch_runner_script(code=code, test_cases=tests)
    started_at = time.perf_counter()

    temp_script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_forge_batch.py", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(runner)
            temp_script_path = Path(temp_file.name)

        # Total timeout is increased for batch to allow all tests to run
        # but internal tests are still individually timed implicitly by total time
        batch_timeout = SANDBOX_TIMEOUT_SECONDS * 2 + len(tests) * 0.1 
        
        proc = subprocess.run(
            [sys.executable, "-I", str(temp_script_path)],
            capture_output=True,
            text=True,
            timeout=batch_timeout,
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            cwd=os.getcwd(),
        )
        
        stdout = _safe_text(proc.stdout)
        stderr = _safe_text(proc.stderr)

        results = _extract_batch_payload(stdout)
        if results is None:
            # Fallback for catastrophic failure
            error_msg = (stderr or stdout or "Catastrophic sandbox failure").splitlines()[0]
            return [{"status": "error", "error_msg": error_msg, "weight": t.get("weight", 1.0)} for t in tests]

        return results

    except subprocess.TimeoutExpired:
        return [{"status": "timeout", "error_msg": "Batch timeout", "weight": t.get("weight", 1.0)} for t in tests]
    except Exception as exc:
        return [{"status": "error", "error_msg": f"Sandbox error: {exc}", "weight": t.get("weight", 1.0)} for t in tests]
    finally:
        if temp_script_path and temp_script_path.exists():
            try:
                temp_script_path.unlink()
            except OSError:
                pass
