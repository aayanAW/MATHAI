"""SymPy-based answer checking for MATH problems.

Uses symbolic comparison, NOT string matching. This is critical —
string matching misses equivalent expressions like "1/2" vs "0.5" vs "\\frac{1}{2}".
"""
import re
import signal
from typing import Optional, Tuple

import sympy
from sympy import (
    Eq, Rational, S, oo, pi, sqrt, simplify, sympify, I, E,
    cos, sin, tan, log, exp, factorial, binomial,
)
from sympy.parsing.latex import parse_latex


# Timeout handler for SymPy operations that might hang
class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("SymPy operation timed out")


def normalize_latex(s: str) -> str:
    """Normalize a LaTeX answer string for comparison.

    Handles common MATH dataset formatting:
    - \\frac{a}{b} → a/b
    - \\sqrt{x} → sqrt(x)
    - \\text{...} → strip
    - \\left, \\right → strip
    - \\, → strip (thin space)
    - \\dfrac → \\frac
    - \\tfrac → \\frac
    """
    s = s.strip()
    # Remove text commands
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\operatorname\{([^}]*)\}", r"\1", s)
    # Normalize fraction commands
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    # Remove formatting commands
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = s.replace("\\;", "")
    s = s.replace("\\quad", " ")
    s = s.replace("\\qquad", " ")
    s = s.replace("\\displaystyle", "")
    # Handle degrees
    s = s.replace("^\\circ", "")
    s = s.replace("^{\\circ}", "")
    # Handle percent
    s = s.replace("\\%", "")
    # Strip whitespace
    s = s.strip()
    # Handle empty
    if not s:
        return ""
    return s


def _extract_braced(s: str, start: int) -> tuple:
    """Extract content inside balanced braces starting at s[start] = '{'.

    Returns (content, end_index) where end_index is past the closing '}'.
    """
    if start >= len(s) or s[start] != "{":
        return "", start
    depth = 1
    i = start + 1
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    return s[start + 1:i - 1], i


def _replace_frac(s: str) -> str:
    """Replace \\frac{num}{den} with ((num)/(den)), handling nested braces."""
    result = []
    i = 0
    while i < len(s):
        if s[i:].startswith("\\frac{"):
            i += 5  # skip \frac
            num, i = _extract_braced(s, i)
            den, i = _extract_braced(s, i)
            # Recursively process numerator and denominator
            num = _replace_frac(num)
            den = _replace_frac(den)
            result.append(f"(({num})/({den}))")
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def _replace_sqrt(s: str) -> str:
    """Replace \\sqrt{expr} with sqrt(expr), handling nested braces."""
    result = []
    i = 0
    while i < len(s):
        if s[i:].startswith("\\sqrt{"):
            i += 5  # skip \sqrt
            content, i = _extract_braced(s, i)
            content = _replace_sqrt(content)
            result.append(f"sqrt({content})")
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def parse_answer(answer_str: str) -> Optional[sympy.Expr]:
    """Parse a MATH answer string into a SymPy expression.

    Tries multiple parsing strategies in order of reliability.
    """
    s = normalize_latex(answer_str)
    if not s:
        return None

    # Strategy 1: Direct sympify for simple expressions
    # Replace common LaTeX with Python
    s_python = s
    # Handle nested braces in \frac using a proper parser
    s_python = _replace_frac(s_python)
    s_python = _replace_sqrt(s_python)
    s_python = s_python.replace("\\pi", "pi")
    s_python = s_python.replace("\\infty", "oo")
    s_python = s_python.replace("\\cdot", "*")
    s_python = s_python.replace("\\times", "*")
    s_python = s_python.replace("^", "**")

    try:
        result = sympify(s_python)
        if result is not None:
            return result
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        pass

    # Strategy 2: parse_latex
    try:
        result = parse_latex(s)
        if result is not None:
            return result
    except Exception:
        pass

    # Strategy 3: Handle negative fractions like -\frac{1}{3}
    s_neg = s_python.strip()
    neg = False
    if s_neg.startswith("-"):
        neg = True
        s_neg = s_neg[1:].strip()
    # Re-try sympify after stripping negative
    if neg:
        try:
            result = sympify(s_neg)
            if result is not None:
                return -result
        except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
            pass

    # Strategy 4: Try as a plain number
    try:
        val = float(s_python.replace(" ", ""))
        return sympify(val)
    except (ValueError, TypeError):
        pass

    # Strategy 5: Handle "a*\pi" or "a\pi" patterns
    pi_match = re.match(r"^(-?\d*\.?\d*)\s*\\?pi$", s.strip())
    if pi_match:
        coeff = pi_match.group(1)
        if coeff in ("", "+"):
            return pi
        if coeff == "-":
            return -pi
        return sympify(float(coeff)) * pi

    return None


def answers_equivalent(
    predicted: str,
    gold: str,
    timeout_seconds: int = 5,
) -> Tuple[bool, str]:
    """Check if two answer strings are mathematically equivalent.

    Uses SymPy symbolic comparison with timeout protection.

    Args:
        predicted: Model's predicted answer.
        gold: Gold standard answer.
        timeout_seconds: Max seconds for SymPy operations.

    Returns:
        (is_correct, method) where method describes how equivalence was determined.
    """
    # Quick string comparison first
    p_norm = normalize_latex(predicted).strip()
    g_norm = normalize_latex(gold).strip()

    if p_norm == g_norm:
        return True, "string_match"

    # Try case-insensitive for text answers
    if p_norm.lower() == g_norm.lower():
        return True, "case_insensitive"

    # Parse both to SymPy
    p_expr = parse_answer(predicted)
    g_expr = parse_answer(gold)

    if p_expr is None or g_expr is None:
        # Can't parse — fall back to string comparison
        return p_norm == g_norm, "unparseable"

    # Set timeout for SymPy operations
    # SIGALRM only works in main thread; skip timeout if in worker thread
    import threading
    use_alarm = threading.current_thread() is threading.main_thread()
    old_handler = None
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    try:
        # Strategy 1: Direct equality
        if p_expr == g_expr:
            return True, "sympy_equal"

        # Strategy 2: Difference simplifies to zero
        # Only works for numeric/algebraic types, not tuples/sets
        try:
            diff = sympy.expand(p_expr - g_expr)
            if diff == 0:
                return True, "expand_zero"

            # Strategy 3: simplify(diff) == 0 (slower but catches more)
            if simplify(diff) == 0:
                return True, "simplify_zero"
        except (TypeError, AttributeError):
            # p_expr or g_expr is a non-numeric type (tuple, set, Point2D, etc.)
            # Fall through to string comparison
            pass

        # Strategy 4: Numerical comparison (for transcendental expressions)
        # Use relaxed tolerance for truncated decimals (e.g., 0.333333 vs 1/3)
        try:
            p_float = complex(p_expr.evalf(15))
            g_float = complex(g_expr.evalf(15))
            # Absolute and relative tolerance
            abs_diff = abs(p_float - g_float)
            rel_diff = abs_diff / max(abs(g_float), 1e-15)
            if abs_diff < 1e-4 or rel_diff < 1e-4:
                return True, "numerical"
        except (TypeError, ValueError, AttributeError):
            pass

        # Strategy 5: String comparison as last resort for structured answers
        if str(p_expr) == str(g_expr):
            return True, "str_equal"

        return False, "different"

    except TimeoutError:
        # SymPy timed out — try numerical as last resort
        try:
            p_float = complex(p_expr.evalf(15))
            g_float = complex(g_expr.evalf(15))
            abs_diff = abs(p_float - g_float)
            rel_diff = abs_diff / max(abs(g_float), 1e-15)
            if abs_diff < 1e-4 or rel_diff < 1e-4:
                return True, "numerical_after_timeout"
        except (TypeError, ValueError, AttributeError):
            pass
        return False, "timeout"

    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def extract_model_answer(response: str) -> str:
    """Extract the final answer from a model's response.

    Looks for common answer patterns:
    - \\boxed{...}
    - "The answer is ..."
    - "ANSWER: ..."
    - Last number/expression in the response
    """
    # Try \\boxed{} first
    idx = response.rfind("\\boxed{")
    if idx != -1:
        start = idx + 7
        depth = 1
        i = start
        while i < len(response) and depth > 0:
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
            i += 1
        return response[start:i - 1]

    # Try "ANSWER:" pattern (from verification scripts)
    match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response)
    if match:
        return match.group(1).strip()

    # Try "the answer is" pattern
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)",
        response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Try "= " at end of last line
    lines = response.strip().split("\n")
    for line in reversed(lines):
        match = re.search(r"=\s*(.+?)$", line.strip())
        if match:
            return match.group(1).strip()

    return response.strip().split("\n")[-1].strip()


if __name__ == "__main__":
    # Test answer checking
    test_cases = [
        ("\\frac{1}{2}", "0.5", True),
        ("3", "3", True),
        ("\\sqrt{2}", "1.41421356", True),
        ("x^2 + 1", "1 + x^2", True),
        ("\\frac{3}{4}", "\\frac{6}{8}", True),
        ("5", "6", False),
        ("\\pi", "3.14159265", True),
        ("-\\frac{1}{3}", "-0.333333", True),
    ]

    print("Testing answer equivalence checker:")
    for pred, gold, expected in test_cases:
        result, method = answers_equivalent(pred, gold)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{pred}' vs '{gold}' → {result} ({method})")
