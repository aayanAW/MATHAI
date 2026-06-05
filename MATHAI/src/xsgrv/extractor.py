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


EXTRACTION_PROMPT = """You are a MATH VERIFICATION EXPERT. Your job is to read a math problem and produce a Python function that checks whether a candidate answer is correct. You are NOT allowed to solve the problem; you can only enumerate or directly check constraints.

# STRICT ABSTENTION RULE
If writing a verifier would require you to reproduce the problem's mathematical insight (construct a proof, derive a formula, use a clever bijection, compute geometric areas from implicit constraints, etc.), you MUST output exactly:

UNVERIFIABLE

Do not attempt to half-solve the problem. Verifying is NOT solving. If you cannot enumerate or directly check constraints in < 80 lines, abstain.

# Allowed verification patterns (produce a script if ANY of these fit)
- Bounded enumeration: "for x in range(...)" over all candidates satisfying constraints, then check count/sum
- Polynomial evaluation: substitute the claimed answer into the given equation and check it satisfies
- Divisibility / modular check: directly compute a%b and compare
- Root check: substitute claimed root into polynomial and check = 0
- Base conversion / number-theoretic test: direct computation over bounded integers
- Counting with given constraints: enumerate a bounded set and count

# NOT allowed (output UNVERIFIABLE)
- Geometric problems where you'd have to compute areas/lengths from coordinate setups you invent
- Problems where the verifier would have to re-derive the formula the problem is asking for
- Problems where the answer is a function of another value the problem asks you to compute first
- Problems requiring calculus, limits, or analysis
- Problems where "verifying" would itself take > 5 seconds of compute

# Rules
1. Output a Python function `verify(answer) -> bool`.
2. Pre-imported: `from sympy import *`, `math`, `itertools.permutations/combinations/product`, `math.comb`, `Fraction`, `Counter`.
3. The function must be DETERMINISTIC: no network, no LLM, no randomness.
4. Parse the candidate answer defensively: `try: int(str(answer).strip())` and handle failure.
5. Keep scripts under 60 lines.
6. DO NOT hardcode the answer. The check must re-derive the result from problem constraints.
7. If you have ANY doubt about whether your verifier is correct, output UNVERIFIABLE.

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
        import math
        import itertools
        from itertools import permutations, combinations, product, combinations_with_replacement
        from math import gcd, lcm, factorial, floor, ceil, sqrt, log, log2, log10, sin, cos, tan, pi, e
        from fractions import Fraction
        from collections import Counter, defaultdict
        import re

        def timeout_handler(signum, frame):
            raise TimeoutError("script exceeded time budget")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm({int(timeout)})

        from sympy import *
        import sympy
        from sympy import Rational, Integer, Float

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


def _deployment_time_probes(candidate_answer: str) -> list[str]:
    """Generate wrong-answer probes relative to the candidate (not the gold).

    A well-behaved verifier must reject all of these. If it accepts any, the
    verifier is broken (or a tautology) and we must abstain. This runs at
    deployment time and therefore NEVER touches the gold answer.

    All probes are guaranteed distinct from the candidate. Non-integer
    candidates fall back to a fixed set that is extremely unlikely to match.
    """
    cand_str = str(candidate_answer).strip()
    try:
        c = int(cand_str)
        probes_int = {c - 1, c + 1, c + 7, c * 2 if c != 0 else 100, c + 100}
        probes_int.discard(c)
        # Return deterministic ordering for reproducibility
        return [str(p) for p in sorted(probes_int)]
    except (ValueError, TypeError):
        fallbacks = ["0", "1", "-1", "42", "100"]
        return [p for p in fallbacks if p != cand_str]


def deployment_time_sanity_check(
    script: str,
    candidate_answer: str,
    timeout: float = 10.0,
) -> tuple[bool, list[dict]]:
    """Check whether the verifier accepts wrong-answer probes near the candidate.

    Returns (broken, probe_results). `broken=True` means the verifier accepted
    at least one probe that is clearly not the candidate, so the verifier is
    unreliable and the caller should abstain.

    This function NEVER reads or uses the gold answer. Its only input is the
    candidate the solver proposed. This is what makes it deployment-time safe.
    """
    probes = _deployment_time_probes(candidate_answer)
    probe_results = []
    broken = False
    for probe in probes:
        # Defensive: probes should never equal the candidate, but double-check
        if probe == str(candidate_answer).strip():
            continue
        ver = execute_verifier(script, probe, timeout=timeout)
        probe_results.append({
            "probe": probe,
            "verdict": ver.verdict,
            "error": ver.error,
        })
        if ver.verdict is True:
            broken = True
    return broken, probe_results


def run_xsgrv(
    problem: str,
    candidate_answer: str,
    extractor_fn: Callable[[str], str],
    timeout: float = 10.0,
    deployment_filter: bool = True,
):
    """Full X-SGRV pipeline: extract → sanity-check → execute.

    If ``deployment_filter`` is True (default), a deployment-time adversarial
    filter runs before returning a non-None verdict. If any probe near the
    candidate is accepted, the verifier is classified BROKEN and we abstain.

    Returns a dict with fields:
        extraction, verification, verdict, cov_fires,
        filter_broken (bool), filter_probes (list)
    """
    ext = extract_verifier(problem, extractor_fn)
    if ext.unverifiable or ext.script is None:
        return {
            "extraction": ext,
            "verification": None,
            "verdict": None,  # abstained
            "cov_fires": False,
            "filter_broken": False,
            "filter_probes": [],
        }

    # Deployment-time sanity check runs BEFORE verifying the candidate.
    # It uses only the candidate answer, never the gold.
    filter_broken = False
    filter_probes: list[dict] = []
    if deployment_filter:
        filter_broken, filter_probes = deployment_time_sanity_check(
            ext.script, candidate_answer, timeout=timeout
        )
        if filter_broken:
            return {
                "extraction": ext,
                "verification": None,
                "verdict": None,  # abstain on broken verifier
                "cov_fires": False,
                "filter_broken": True,
                "filter_probes": filter_probes,
            }

    ver = execute_verifier(ext.script, candidate_answer, timeout=timeout)
    if not ver.executed or ver.verdict is None:
        return {
            "extraction": ext,
            "verification": ver,
            "verdict": None,  # execution error = abstain
            "cov_fires": False,
            "filter_broken": filter_broken,
            "filter_probes": filter_probes,
        }
    return {
        "extraction": ext,
        "verification": ver,
        "verdict": ver.verdict,  # True or False
        "cov_fires": True,
        "filter_broken": filter_broken,
        "filter_probes": filter_probes,
    }
