"""X-SGRV: Cross-family LLM-extracted symbolic verification.

Pipeline:
1. EXTRACTOR (different model family from solver): reads the problem statement,
   outputs an executable Python/SymPy script `verify(answer) -> (passes, info)`.
2. EXECUTOR (local sandbox): runs `verify(solver_answer)` and returns PASS/FAIL.

Independence property: the extractor never sees the solver's solution. Its only
input is the problem text. Verification is deterministic symbolic execution.
"""
import re
import sys
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


EXTRACTION_PROMPT = """You are a MATH VERIFICATION EXPERT. Your job is to read a math problem and produce a Python function that checks whether a candidate answer is correct, WITHOUT solving the problem yourself.

# Rules
1. You must output a Python script that defines a function `verify(answer) -> bool`.
2. The function takes the candidate answer (as a string or number) and returns True if the answer satisfies the problem's constraints.
3. Use SymPy for symbolic math. `from sympy import *` is pre-imported.
4. The function must be DETERMINISTIC: no LLM calls, no network, no randomness.
5. You may use bounded search, enumeration, or closed-form checking.
6. If the problem has a unique answer, verify the candidate equals the correct answer by re-deriving it from the constraints.
7. If the problem asks for a count or sum over a bounded set, enumerate and sum.
8. If you cannot write a verifier (e.g., the problem needs creativity you don't have), output exactly `UNVERIFIABLE` and nothing else.
9. Keep scripts under 80 lines. Use at most 5 seconds of compute.
10. DO NOT solve the problem and hardcode the answer. The verifier must work by re-checking constraints, not by comparing to a memorized answer.

# Good example

Problem: Find the number of ordered triples (a, b, c) of positive integers with a + b + c = 10 and a <= b <= c.

```python
def verify(answer):
    count = 0
    for a in range(1, 11):
        for b in range(a, 11):
            for c in range(b, 11):
                if a + b + c == 10:
                    count += 1
    try:
        return int(str(answer).strip()) == count
    except Exception:
        return False
```

# Bad example (hardcodes the answer - NEVER DO THIS)

```python
def verify(answer):
    return int(str(answer)) == 8  # BAD: this is solving, not verifying
```

# Another good example

Problem: Find the sum of all integer bases b > 9 such that 17_b divides 97_b.

```python
def verify(answer):
    valid = []
    for b in range(10, 1000):
        x = 1*b + 7  # 17 in base b
        y = 9*b + 7  # 97 in base b
        if y % x == 0:
            valid.append(b)
    total = sum(valid)
    try:
        return int(str(answer).strip()) == total
    except Exception:
        return False
```

# Now extract for this problem

Problem: {problem}

Output ONLY the Python code (starting with ```python and ending with ```), or the exact word UNVERIFIABLE if you cannot construct a verifier. Do not include any explanation before or after.
"""


@dataclass
class ExtractionResult:
    """Result of the extraction step."""
    script: Optional[str]
    unverifiable: bool
    raw_response: str
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of running the extracted verifier on a candidate answer."""
    script: Optional[str]
    candidate_answer: str
    executed: bool
    verdict: Optional[bool]  # True=pass, False=fail, None=error
    error: Optional[str] = None
    execution_time: float = 0.0


def extract_verifier(
    problem: str,
    llm_call: Callable[[str], str],
) -> ExtractionResult:
    """Extract a SymPy verifier script from the problem using an LLM.

    Args:
        problem: The math problem statement.
        llm_call: A function that takes a prompt and returns a response.

    Returns:
        ExtractionResult with the extracted script or UNVERIFIABLE verdict.
    """
    try:
        raw = llm_call(EXTRACTION_PROMPT.format(problem=problem))
    except Exception as e:
        return ExtractionResult(None, False, "", error=f"llm_call failed: {e}")

    raw = raw.strip()

    # Check for UNVERIFIABLE
    if re.search(r"^\s*UNVERIFIABLE\s*$", raw, re.MULTILINE) or raw == "UNVERIFIABLE":
        return ExtractionResult(None, True, raw)

    # Extract Python code block
    code_match = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
    if not code_match:
        # Sometimes LLMs forget the ```python marker
        code_match = re.search(r"```\n(.*?)```", raw, re.DOTALL)
    if not code_match:
        # Last resort: assume the whole response is code if it has `def verify`
        if "def verify" in raw:
            return ExtractionResult(raw, False, raw)
        return ExtractionResult(None, False, raw, error="no code block found")

    code = code_match.group(1).strip()

    # Basic sanity check: must define `verify`
    if "def verify" not in code:
        return ExtractionResult(None, False, raw, error="no verify function defined")

    return ExtractionResult(code, False, raw)


def execute_verifier(
    script: str,
    candidate_answer: str,
    timeout: float = 10.0,
) -> VerificationResult:
    """Execute the extracted verifier on a candidate answer in a sandbox.

    Args:
        script: The Python script containing `def verify(answer) -> bool`.
        candidate_answer: The solver's answer (as a string).
        timeout: Maximum execution time in seconds.

    Returns:
        VerificationResult with the verdict.
    """
    import time
    # Prepend SymPy import and append a driver that calls verify() and prints result
    full_script = textwrap.dedent(f"""
        import sys
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("script exceeded time budget")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm({int(timeout)})

        from sympy import *
        import sympy

        # === EXTRACTED VERIFIER ===
        {textwrap.indent(script, "        ").strip()}
        # === END EXTRACTED VERIFIER ===

        candidate = {repr(candidate_answer)}
        try:
            result = verify(candidate)
            if result is True:
                print("XSGRV_VERDICT_PASS")
            elif result is False:
                print("XSGRV_VERDICT_FAIL")
            else:
                print("XSGRV_VERDICT_UNKNOWN")
        except Exception as e:
            print(f"XSGRV_VERDICT_ERROR: {{type(e).__name__}}: {{e}}")

        signal.alarm(0)
    """)

    t0 = time.time()
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_script)
            script_path = f.name

        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        Path(script_path).unlink()

        dt = time.time() - t0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if "XSGRV_VERDICT_PASS" in stdout:
            return VerificationResult(script, candidate_answer, True, True, execution_time=dt)
        elif "XSGRV_VERDICT_FAIL" in stdout:
            return VerificationResult(script, candidate_answer, True, False, execution_time=dt)
        elif "XSGRV_VERDICT_ERROR" in stdout:
            err = stdout.split("XSGRV_VERDICT_ERROR:", 1)[1].strip()
            return VerificationResult(script, candidate_answer, True, None, error=err, execution_time=dt)
        else:
            return VerificationResult(
                script, candidate_answer, True, None,
                error=f"no verdict. stdout={stdout[:200]} stderr={stderr[:200]}",
                execution_time=dt,
            )
    except subprocess.TimeoutExpired:
        return VerificationResult(script, candidate_answer, False, None, error="subprocess timeout", execution_time=time.time() - t0)
    except Exception as e:
        return VerificationResult(script, candidate_answer, False, None, error=f"{type(e).__name__}: {e}", execution_time=time.time() - t0)


def run_xsgrv(
    problem: str,
    candidate_answer: str,
    extractor_fn: Callable[[str], str],
    timeout: float = 10.0,
):
    """Full X-SGRV pipeline: extract → execute.

    Returns a dict with fields: extraction, verification, verdict, cov_fires.
    """
    ext = extract_verifier(problem, extractor_fn)
    if ext.unverifiable or ext.script is None:
        return {
            "extraction": ext,
            "verification": None,
            "verdict": None,  # abstained
            "cov_fires": False,
        }
    ver = execute_verifier(ext.script, candidate_answer, timeout=timeout)
    if not ver.executed or ver.verdict is None:
        return {
            "extraction": ext,
            "verification": ver,
            "verdict": None,  # execution error = abstain
            "cov_fires": False,
        }
    return {
        "extraction": ext,
        "verification": ver,
        "verdict": ver.verdict,  # True or False
        "cov_fires": True,
    }
