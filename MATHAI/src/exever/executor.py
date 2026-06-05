"""Sandboxed Python/SymPy execution for verification scripts.

Executes verification code in an isolated subprocess with:
- Timeout protection (30s default)
- Output capture (stdout, stderr)
- Assertion error detection and step localization
- Process pool management
"""
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ExecutionResult:
    """Result of executing a verification script."""
    success: bool  # All assertions passed
    stdout: str = ""
    stderr: str = ""
    assertion_error: bool = False
    error_step: int = -1  # 0-indexed step that failed (-1 = no assertion error)
    error_message: str = ""
    runtime_error: bool = False
    timeout: bool = False
    return_code: int = -1
    assertions_found: int = 0  # Total assert statements in script
    answer_extracted: str = ""  # From "ANSWER: ..." in stdout

    @property
    def verdict(self) -> str:
        if self.success:
            return "ALL_PASS"
        if self.assertion_error:
            return f"FAIL_STEP_{self.error_step}"
        if self.timeout:
            return "TIMEOUT"
        return "ERROR"


def execute_verification_script(
    script: str,
    timeout: int = 30,
    extra_imports: str = "",
) -> ExecutionResult:
    """Execute a verification script in a sandboxed subprocess.

    Args:
        script: Python/SymPy verification code to execute.
        timeout: Maximum execution time in seconds.
        extra_imports: Additional imports to prepend.

    Returns:
        ExecutionResult with detailed outcome information.
    """
    # Prepend standard imports if not present
    if "from sympy import" not in script and "import sympy" not in script:
        script = "from sympy import *\n" + script

    if extra_imports:
        script = extra_imports + "\n" + script

    # Count assertions (including indented ones inside blocks)
    assertions_found = sum(
        1 for line in script.split("\n")
        if line.strip().startswith("assert ")
        or line.strip().startswith("assert(")
    )

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="exever_",
    ) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Extract answer from stdout
        answer = ""
        answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", stdout)
        if answer_match:
            answer = answer_match.group(1).strip()

        if result.returncode == 0:
            return ExecutionResult(
                success=True,
                stdout=stdout,
                stderr=stderr,
                return_code=0,
                assertions_found=assertions_found,
                answer_extracted=answer,
            )

        # Check for assertion error
        if "AssertionError" in stderr or "AssertionError" in stdout:
            error_msg = stderr if stderr else stdout
            step = _extract_step_from_assertion_error(error_msg)
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                assertion_error=True,
                error_step=step,
                error_message=_extract_assertion_message(error_msg),
                return_code=result.returncode,
                assertions_found=assertions_found,
                answer_extracted=answer,
            )

        # Runtime error (SyntaxError, NameError, etc.)
        return ExecutionResult(
            success=False,
            stdout=stdout,
            stderr=stderr,
            runtime_error=True,
            error_message=stderr[:500],
            return_code=result.returncode,
            assertions_found=assertions_found,
            answer_extracted=answer,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            timeout=True,
            error_message=f"Execution timed out after {timeout}s",
            assertions_found=assertions_found,
        )

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _extract_step_from_assertion_error(error_output: str) -> int:
    """Extract the 0-indexed step number from an assertion error.

    Looks for "FAIL:Step k:" pattern first, then falls back to
    counting which assertion in the traceback failed.
    """
    # Try FAIL:Step k: pattern in error message
    match = re.search(r"FAIL:Step\s*(\d+)", error_output)
    if match:
        return int(match.group(1)) - 1  # Convert to 0-indexed

    # Try to find line number and count assertions up to it
    match = re.search(r'File ".*?", line (\d+)', error_output)
    if match:
        line_no = int(match.group(1))
        # This gives us the line number — but we'd need the script
        # to count which assertion it is. Return -1 for now.
        return -1

    return -1


def _extract_assertion_message(error_output: str) -> str:
    """Extract the human-readable assertion error message."""
    # Look for the assertion message after AssertionError:
    match = re.search(r"AssertionError:\s*(.+?)(?:\n|$)", error_output)
    if match:
        return match.group(1).strip()

    # Look for FAIL: pattern
    match = re.search(r"FAIL:(.+?)(?:\n|$)", error_output)
    if match:
        return match.group(1).strip()

    return error_output[-200:]  # Last 200 chars as fallback


def execute_batch(
    scripts: List[str],
    timeout: int = 30,
    max_workers: int = 8,
) -> List[ExecutionResult]:
    """Execute multiple verification scripts in parallel.

    Uses a process pool to limit concurrent executions.
    """
    results = []
    # Execute sequentially to avoid fork issues with SymPy
    for script in scripts:
        results.append(execute_verification_script(script, timeout))
    return results


def analyze_execution_results(
    results: List[ExecutionResult],
) -> Dict[str, float]:
    """Compute aggregate statistics over execution results."""
    n = len(results)
    if n == 0:
        return {}

    n_success = sum(1 for r in results if r.success)
    n_assertion = sum(1 for r in results if r.assertion_error)
    n_runtime = sum(1 for r in results if r.runtime_error)
    n_timeout = sum(1 for r in results if r.timeout)

    return {
        "total": n,
        "success_rate": n_success / n,
        "assertion_fail_rate": n_assertion / n,
        "runtime_error_rate": n_runtime / n,
        "timeout_rate": n_timeout / n,
        "script_validity_rate": (n_success + n_assertion) / n,  # Scripts that ran (didn't crash)
    }


if __name__ == "__main__":
    # Test execution
    print("Testing executor...")

    # Test 1: Successful script
    script1 = """
from sympy import *
x = symbols('x')
eq = x**2 - 5*x + 6
assert expand((x-2)*(x-3)) == expand(eq), "FAIL:Step 1: factoring"
assert eq.subs(x, 2) == 0, "FAIL:Step 2: x=2 not root"
print("ANSWER: 5")
"""
    r1 = execute_verification_script(script1)
    print(f"Test 1 (should pass): {r1.verdict}, answer={r1.answer_extracted}")

    # Test 2: Assertion failure
    script2 = """
from sympy import *
x = symbols('x')
assert 2 + 2 == 5, "FAIL:Step 1: arithmetic"
"""
    r2 = execute_verification_script(script2)
    print(f"Test 2 (should fail at step 0): {r2.verdict}, step={r2.error_step}")

    # Test 3: Runtime error
    script3 = """
undefined_variable + 1
"""
    r3 = execute_verification_script(script3)
    print(f"Test 3 (should be runtime error): {r3.verdict}")

    # Test 4: Timeout
    script4 = """
import time
time.sleep(35)
"""
    r4 = execute_verification_script(script4, timeout=2)
    print(f"Test 4 (should timeout): {r4.verdict}")
