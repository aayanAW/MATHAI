"""Subprocess-based sandbox for executing extractor-emitted verifiers.

Ported from the X-SGRV reference implementation
(MATHAI/src/xsgrv/extractor.py) with minor cleanups. Behavior preserved
verbatim so existing cached extractor outputs remain reproducible.
"""
from __future__ import annotations

import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VerificationResult:
    """Result of running a verifier on a candidate answer."""
    script: Optional[str]
    candidate_answer: str
    executed: bool
    verdict: Optional[bool]  # True=pass, False=fail, None=error
    error: Optional[str] = None
    execution_time: float = 0.0


def execute_verifier(
    script: str,
    candidate_answer: str,
    timeout: float = 10.0,
    python_executable: str = "python3",
) -> VerificationResult:
    """Execute the extractor-emitted ``verify(answer) -> bool`` against a candidate.

    Args:
        script: Python source containing ``def verify(answer)``.
        candidate_answer: The solver's plurality answer, passed as a string.
        timeout: Wall-clock cap in seconds (enforced by SIGALRM + subprocess).
        python_executable: Python interpreter for the subprocess. The
            interpreter must have ``sympy`` installed.

    Returns:
        VerificationResult with verdict ``True`` / ``False`` / ``None``.
    """
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
                print("VE_VERDICT_PASS")
            elif result is False:
                print("VE_VERDICT_FAIL")
            else:
                print("VE_VERDICT_UNKNOWN")
        except Exception as e:
            print(f"VE_VERDICT_ERROR: {{type(e).__name__}}: {{e}}")

        signal.alarm(0)
    """)

    t0 = time.time()
    script_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_script)
            script_path = f.name

        proc = subprocess.run(
            [python_executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )

        dt = time.time() - t0
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if "VE_VERDICT_PASS" in stdout:
            return VerificationResult(script, candidate_answer, True, True, execution_time=dt)
        if "VE_VERDICT_FAIL" in stdout:
            return VerificationResult(script, candidate_answer, True, False, execution_time=dt)
        if "VE_VERDICT_ERROR" in stdout:
            err = stdout.split("VE_VERDICT_ERROR:", 1)[1].strip()
            return VerificationResult(script, candidate_answer, True, None,
                                      error=err, execution_time=dt)
        return VerificationResult(
            script, candidate_answer, True, None,
            error=f"no verdict; stdout={stdout[:200]} stderr={stderr[:200]}",
            execution_time=dt,
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(script, candidate_answer, False, None,
                                  error="subprocess timeout",
                                  execution_time=time.time() - t0)
    except Exception as e:
        return VerificationResult(script, candidate_answer, False, None,
                                  error=f"{type(e).__name__}: {e}",
                                  execution_time=time.time() - t0)
    finally:
        if script_path is not None:
            try:
                Path(script_path).unlink()
            except OSError:
                pass
