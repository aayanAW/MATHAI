"""Template-based property test generation for PBT.

Each template generates an executable Python/SymPy script that tests
a mathematical claim against the problem statement using random inputs.

Design principles:
1. Every test references the PROBLEM STATEMENT, not the model's solution
2. Templates validate preconditions before generating code
3. If validation fails -> return None (mapped to UNTESTABLE)
4. Templates track independence level (fully vs partially independent)
"""
import re
from typing import Optional, Tuple

import sympy
from sympy.parsing.latex import parse_latex

from .claim_classifier import ClaimType


class ValidationError(Exception):
    """Raised when extraction/validation fails."""
    pass


class TestResult:
    """Result of template test generation."""
    def __init__(
        self,
        script: Optional[str],
        claim_type: ClaimType,
        independence: str,  # "fully", "partially", "failed"
        validation_notes: str = "",
    ):
        self.script = script
        self.claim_type = claim_type
        self.independence = independence
        self.validation_notes = validation_notes


def _extract_polynomial_from_problem(problem_text: str) -> Optional[str]:
    """Try to extract a polynomial equation from the problem text.

    Returns a SymPy-parseable string, or None if extraction fails.
    """
    # Pattern: "x^2 - 5x + 6 = 0" or "$x^2 - 5x + 6$"
    patterns = [
        r"\$([^$]*[x]\^?\d*[^$]*=\s*0)\$",  # $expr = 0$
        r"([a-z]\^?\{?\d?\}?\s*[-+].*?=\s*0)",  # expr = 0
        r"\$([^$]*[x][^$]*)\$",  # any $expr with x$
    ]
    for pattern in patterns:
        m = re.search(pattern, problem_text)
        if m:
            expr_str = m.group(1).strip()
            # Remove = 0 if present
            expr_str = re.sub(r"\s*=\s*0\s*$", "", expr_str)
            # Convert LaTeX to Python
            expr_str = expr_str.replace("^", "**")
            expr_str = re.sub(r"(\d)([a-z])", r"\1*\2", expr_str)  # 5x -> 5*x
            return expr_str
    return None


def _parse_expr_safe(expr_str: str) -> Optional[sympy.Expr]:
    """Safely parse a string to SymPy expression."""
    try:
        return sympy.sympify(expr_str)
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        pass
    try:
        return parse_latex(expr_str)
    except Exception:
        pass
    return None


def _validate_parseable(value: str, name: str) -> None:
    """Validate that a value is parseable by SymPy."""
    if not value or not value.strip():
        raise ValidationError(f"{name} is empty")
    expr = _parse_expr_safe(value)
    if expr is None:
        raise ValidationError(f"{name} is not parseable: {value!r}")


def generate_root_claim_test(
    claimed_roots: list,
    problem_text: str,
) -> TestResult:
    """Generate property test for a root claim.

    Tests claimed roots against the polynomial from the PROBLEM STATEMENT.
    Fully independent: polynomial from problem, roots from model.
    """
    # PRECONDITION 1: Extract polynomial from problem
    poly_str = _extract_polynomial_from_problem(problem_text)
    if poly_str is None:
        return TestResult(None, ClaimType.ROOT_CLAIM, "failed",
                         "Cannot extract polynomial from problem text")

    # PRECONDITION 2: Polynomial must be parseable
    try:
        _validate_parseable(poly_str, "problem polynomial")
    except ValidationError as e:
        return TestResult(None, ClaimType.ROOT_CLAIM, "failed", str(e))

    # PRECONDITION 3: Claimed roots must be parseable
    parsed_roots = []
    for r in claimed_roots:
        try:
            _validate_parseable(str(r), f"claimed root {r}")
            parsed_roots.append(str(r).strip())
        except ValidationError:
            return TestResult(None, ClaimType.ROOT_CLAIM, "failed",
                             f"Cannot parse claimed root: {r}")

    if not parsed_roots:
        return TestResult(None, ClaimType.ROOT_CLAIM, "failed",
                         "No valid roots extracted")

    roots_str = ", ".join(parsed_roots)

    script = f"""from sympy import *
x = symbols('x')

# Polynomial from PROBLEM STATEMENT (specification-grounded)
f = {poly_str}

# Test claimed roots against problem polynomial
claimed_roots = [{roots_str}]
for root in claimed_roots:
    val = f.subs(x, root)
    val_expanded = expand(val)
    assert val_expanded == 0, f"FAIL: f({{root}}) = {{val_expanded}}, not a root"

print("PASS: All claimed roots verified against problem polynomial")
"""

    return TestResult(script, ClaimType.ROOT_CLAIM, "fully",
                     f"poly={poly_str}, roots={roots_str}")


def generate_algebraic_equiv_test(
    lhs: str,
    rhs: str,
    problem_text: str,
) -> TestResult:
    """Generate property test for an algebraic equivalence claim.

    Tests that LHS and RHS are equal by:
    1. Symbolic expansion check
    2. Random numerical evaluation at 200 points
    """
    # Clean LaTeX to SymPy
    lhs_clean = _latex_to_sympy(lhs)
    rhs_clean = _latex_to_sympy(rhs)
    if lhs_clean is None or rhs_clean is None:
        return TestResult(None, ClaimType.ALGEBRAIC_EQUIV, "failed",
                         f"Cannot convert LaTeX: lhs={lhs}, rhs={rhs}")

    lhs = lhs_clean
    rhs = rhs_clean

    # PRECONDITION: Both sides parseable
    try:
        _validate_parseable(lhs, "LHS")
        _validate_parseable(rhs, "RHS")
    except ValidationError as e:
        return TestResult(None, ClaimType.ALGEBRAIC_EQUIV, "failed", str(e))

    # Determine independence: is LHS or RHS from the problem?
    lhs_in_problem = any(token in problem_text for token in lhs.split() if len(token) > 2)
    independence = "fully" if lhs_in_problem else "partially"

    script = f"""from sympy import *
from sympy import Rational
import random

x = symbols('x')

lhs = {lhs}
rhs = {rhs}

# Test 1: Symbolic expansion
diff = expand(lhs - rhs)
assert diff == 0, f"FAIL: expand(lhs - rhs) = {{diff}}, not zero"

# Test 2: Random numerical evaluation (200 points)
for _ in range(200):
    t = Rational(random.randint(-50, 50), random.randint(1, 10))
    lhs_val = lhs.subs(x, t) if hasattr(lhs, 'subs') else lhs
    rhs_val = rhs.subs(x, t) if hasattr(rhs, 'subs') else rhs
    assert lhs_val == rhs_val, f"FAIL: at x={{t}}: lhs={{lhs_val}} != rhs={{rhs_val}}"

print("PASS: Algebraic equivalence verified")
"""

    return TestResult(script, ClaimType.ALGEBRAIC_EQUIV, independence,
                     f"lhs={lhs}, rhs={rhs}")


def generate_divisibility_test(
    number: str,
    divisor: str,
) -> TestResult:
    """Generate property test for a divisibility claim."""
    try:
        _validate_parseable(number, "number")
        _validate_parseable(divisor, "divisor")
    except ValidationError as e:
        return TestResult(None, ClaimType.DIVISIBILITY, "failed", str(e))

    script = f"""from sympy import *

number = {number}
divisor = {divisor}

# Direct divisibility check
assert number % divisor == 0, f"FAIL: {{number}} is NOT divisible by {{divisor}} (remainder={{number % divisor}})"

print("PASS: Divisibility verified")
"""

    return TestResult(script, ClaimType.DIVISIBILITY, "partially",
                     f"{number} % {divisor}")


def generate_modular_test(
    value: str,
    remainder: str,
    modulus: str,
) -> TestResult:
    """Generate property test for a modular arithmetic claim."""
    try:
        _validate_parseable(value, "value")
        _validate_parseable(remainder, "remainder")
        _validate_parseable(modulus, "modulus")
    except ValidationError as e:
        return TestResult(None, ClaimType.MODULAR, "failed", str(e))

    script = f"""
value = {value}
remainder = {remainder}
modulus = {modulus}

actual_remainder = value % modulus
assert actual_remainder == remainder, \\
    f"FAIL: {{value}} mod {{modulus}} = {{actual_remainder}}, not {{remainder}}"

print("PASS: Modular arithmetic verified")
"""

    return TestResult(script, ClaimType.MODULAR, "partially",
                     f"{value} mod {modulus} = {remainder}")


def _latex_to_sympy(expr_str: str) -> Optional[str]:
    """Convert a LaTeX expression to SymPy-parseable string."""
    s = expr_str.strip()
    if not s:
        return None
    # Remove display math delimiters
    s = s.strip("$").strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    # Handle \frac{a}{b} -> ((a)/(b)) with nested brace support
    while "\\frac" in s:
        idx = s.find("\\frac")
        # Skip \frac and find first {
        pos = idx + 5
        while pos < len(s) and s[pos] in " \t":
            pos += 1
        if pos >= len(s) or s[pos] != "{":
            break
        # Extract numerator
        depth = 1
        start_num = pos + 1
        pos += 1
        while pos < len(s) and depth > 0:
            if s[pos] == "{": depth += 1
            elif s[pos] == "}": depth -= 1
            pos += 1
        num = s[start_num:pos-1]
        # Extract denominator
        while pos < len(s) and s[pos] in " \t":
            pos += 1
        if pos >= len(s) or s[pos] != "{":
            break
        depth = 1
        start_den = pos + 1
        pos += 1
        while pos < len(s) and depth > 0:
            if s[pos] == "{": depth += 1
            elif s[pos] == "}": depth -= 1
            pos += 1
        den = s[start_den:pos-1]
        s = s[:idx] + f"(({num})/({den}))" + s[pos:]
    # Handle \sqrt{x} -> sqrt(x)
    while "\\sqrt" in s:
        idx = s.find("\\sqrt")
        pos = idx + 5
        while pos < len(s) and s[pos] in " \t":
            pos += 1
        if pos >= len(s) or s[pos] != "{":
            s = s[:idx] + "sqrt" + s[pos:]
            continue
        depth = 1
        start = pos + 1
        pos += 1
        while pos < len(s) and depth > 0:
            if s[pos] == "{": depth += 1
            elif s[pos] == "}": depth -= 1
            pos += 1
        content = s[start:pos-1]
        s = s[:idx] + f"sqrt({content})" + s[pos:]
    # Simple replacements
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\div", "/")
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^", "**")
    # Remove remaining curly braces (grouping)
    s = s.replace("{", "(").replace("}", ")")
    # Insert multiplication: 5x -> 5*x
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    # Remove remaining backslashes (unknown commands)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.strip()
    return s if s else None


def generate_numerical_eval_test(
    expression: str,
    claimed_value: str,
) -> TestResult:
    """Generate property test for a numerical evaluation claim."""
    # Try to make expression parseable
    clean_expr = _latex_to_sympy(expression)
    if clean_expr is None:
        return TestResult(None, ClaimType.NUMERICAL_EVAL, "failed",
                         "Cannot clean expression")

    try:
        _validate_parseable(claimed_value, "claimed value")
    except ValidationError as e:
        return TestResult(None, ClaimType.NUMERICAL_EVAL, "failed", str(e))

    # Try to validate the cleaned expression
    try:
        _validate_parseable(clean_expr, "expression")
    except ValidationError:
        return TestResult(None, ClaimType.NUMERICAL_EVAL, "failed",
                         f"Expression not parseable after cleaning: {clean_expr}")

    script = f"""from sympy import *

# Evaluate expression
try:
    expr = sympify('{clean_expr}')
    claimed = sympify('{claimed_value}')

    # Exact check first
    if expr == claimed:
        pass  # exact match
    elif expand(expr - claimed) == 0:
        pass  # algebraically equal
    else:
        # Numerical fallback
        computed = float(expr.evalf(15))
        expected = float(claimed.evalf(15))
        diff = abs(computed - expected)
        rel = diff / max(abs(expected), 1e-15)
        assert diff < 1e-4 or rel < 1e-4, \\
            f"FAIL: {{expr}} = {{computed}}, not {{expected}} (diff={{diff}})"
    print("PASS: Numerical evaluation verified")
except Exception as e:
    assert False, f"FAIL: evaluation error: {{e}}"
"""

    return TestResult(script, ClaimType.NUMERICAL_EVAL, "partially",
                     f"{clean_expr} = {claimed_value}")


def generate_final_answer_test(
    claimed_answer: str,
    gold_answer: str,
) -> TestResult:
    """Generate property test for the final answer.

    This test is fully independent — it compares the model's claimed
    answer against the gold answer from the problem.
    """
    script = f"""from sympy import *

claimed = sympify('{claimed_answer}')
gold = sympify('{gold_answer}')

# Exact comparison
if claimed == gold:
    print("PASS: Final answer matches gold")
elif expand(claimed - gold) == 0:
    print("PASS: Final answer equivalent to gold (expand)")
else:
    # Numerical fallback
    try:
        diff = abs(float(claimed.evalf(15) - gold.evalf(15)))
        assert diff < 1e-4, f"FAIL: claimed={{claimed}}, gold={{gold}}, diff={{diff}}"
        print("PASS: Final answer matches gold (numerical)")
    except (TypeError, ValueError):
        assert str(claimed) == str(gold), f"FAIL: claimed={{claimed}} != gold={{gold}}"
        print("PASS: Final answer matches gold (string)")
"""

    return TestResult(script, ClaimType.FINAL_ANSWER, "fully",
                     f"claimed={claimed_answer}, gold={gold_answer}")


def generate_factoring_test(
    original: str,
    factored: str,
    problem_text: str,
) -> TestResult:
    """Generate property test for a factoring claim.

    Tests that original and factored forms are equal by evaluating
    at 200 random rational points (Schwartz-Zippel).
    """
    # Clean LaTeX
    orig_clean = _latex_to_sympy(original)
    fact_clean = _latex_to_sympy(factored)

    if orig_clean is None or fact_clean is None:
        return TestResult(None, ClaimType.FACTORING, "failed",
                         "Cannot convert LaTeX to SymPy")

    # Check both are parseable
    try:
        _validate_parseable(orig_clean, "original")
        _validate_parseable(fact_clean, "factored")
    except ValidationError as e:
        return TestResult(None, ClaimType.FACTORING, "failed", str(e))

    # Determine independence
    orig_in_problem = any(
        token in problem_text for token in original.replace("\\", " ").split()
        if len(token) > 2
    )
    independence = "fully" if orig_in_problem else "partially"

    script = f"""from sympy import *
import random

x = symbols('x')
r = symbols('r')  # some problems use r instead of x

original = {orig_clean}
factored = {fact_clean}

# Test 1: Symbolic expansion
diff = expand(original - factored)
assert diff == 0, f"FAIL: expand(original - factored) = {{diff}}"

# Test 2: Random numerical evaluation (200 points)
# Detect which variable the expression actually uses
free_vars = original.free_symbols if hasattr(original, 'free_symbols') else set()
test_var = list(free_vars)[0] if free_vars else x
for _ in range(200):
    t = Rational(random.randint(-50, 50), random.randint(1, 10))
    o_val = original.subs(test_var, t)
    f_val = factored.subs(test_var, t)
    assert o_val == f_val, f"FAIL at {{test_var}}={{t}}: orig={{o_val}} != fact={{f_val}}"

print("PASS: Factoring verified")
"""

    return TestResult(script, ClaimType.FACTORING, independence,
                     f"orig={orig_clean}, fact={fact_clean}")


def generate_enumeration_test(
    claimed_count: str,
    problem_text: str,
) -> TestResult:
    """Generate property test for an enumeration/counting claim.

    For small claims (count < 1000), we can do basic sanity checks.
    Full enumeration requires problem-specific logic we can't template.
    """
    try:
        _validate_parseable(claimed_count, "claimed count")
    except ValidationError as e:
        return TestResult(None, ClaimType.ENUMERATION, "failed", str(e))

    # Basic sanity: count should be a non-negative integer
    script = f"""from sympy import *

claimed = sympify('{claimed_count}')

# Sanity checks
assert claimed == int(claimed), f"FAIL: count {{claimed}} is not an integer"
assert int(claimed) >= 0, f"FAIL: count {{int(claimed)}} is negative"

print("PASS: Enumeration count is a non-negative integer")
"""

    return TestResult(script, ClaimType.ENUMERATION, "partially",
                     f"count={claimed_count} (sanity check only)")


# Template dispatch table
TEMPLATE_MAP = {
    ClaimType.ROOT_CLAIM: "root",
    ClaimType.FACTORING: "factoring",
    ClaimType.ALGEBRAIC_EQUIV: "equiv",
    ClaimType.DIVISIBILITY: "divisible",
    ClaimType.MODULAR: "modular",
    ClaimType.NUMERICAL_EVAL: "numerical",
    ClaimType.FINAL_ANSWER: "answer",
    ClaimType.ENUMERATION: "enumerate",
    ClaimType.COORDINATE: "coordinate",
}
