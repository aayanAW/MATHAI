"""Claim type classification for PBT.

Classifies each reasoning step into a testable claim type via regex
pattern matching. Falls back to UNTESTABLE for unrecognized patterns.

Extraction risk: misclassification can cause false passes/fails.
The extraction-validation pipeline (in test_generator.py) catches
most errors, but this module is the primary coverage bottleneck.
"""
import re
from enum import Enum
from typing import List, Optional, Tuple


class ClaimType(Enum):
    """Types of mathematical claims that can be property-tested."""
    ROOT_CLAIM = "root"
    FACTORING = "factoring"
    ALGEBRAIC_EQUIV = "equiv"
    DIVISIBILITY = "divisible"
    NUMERICAL_EVAL = "numerical"
    ENUMERATION = "enumerate"
    COORDINATE = "coordinate"
    MODULAR = "modular"
    FINAL_ANSWER = "answer"
    UNTESTABLE = "untestable"


# Each pattern returns (ClaimType, dict of extracted operands).
# Patterns are tried in order; first match wins.
# IMPORTANT: patterns are designed to be CONSERVATIVE — a false match
# that extracts wrong operands is worse than a miss that falls to UNTESTABLE.

# Common math expression pattern (captures balanced parens/braces)
_EXPR = r"[-+]?\s*[\d\w\^\*\/\(\)\{\}\\\s\.π]+"
_NUM = r"[-+]?\d+(?:\.\d+)?(?:/\d+)?"


def classify_claim(
    step_text: str,
    problem_text: str,
    step_index: int = 0,
) -> Tuple[ClaimType, dict]:
    """Classify a reasoning step's claim type.

    Args:
        step_text: The text of the reasoning step.
        problem_text: The original problem text.
        step_index: 0-indexed step number.

    Returns:
        (ClaimType, extracted_operands) where operands is a dict
        of values needed by the template. Empty dict for UNTESTABLE.
    """
    text = step_text.strip()
    text_lower = text.lower()

    # === FINAL_ANSWER (highest priority — check boxed answer) ===
    # Handle nested braces (e.g., \boxed{\frac{1}{2}})
    boxed_idx = text.rfind("\\boxed{")
    if boxed_idx != -1:
        start = boxed_idx + 7
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        claimed = text[start:i-1].strip()
        if claimed:
            return ClaimType.FINAL_ANSWER, {
                "claimed_answer": claimed,
            }

    # Also check "the answer is" pattern
    answer_match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)",
        text, re.IGNORECASE,
    )
    if answer_match and step_index > 0:  # Don't trigger on problem restatement
        return ClaimType.FINAL_ANSWER, {
            "claimed_answer": answer_match.group(1).strip(),
        }

    # === ROOT_CLAIM ===
    # Patterns: "x = 3 is a root", "roots are 2 and 3", "solutions are x = ..."
    root_patterns = [
        r"(?:root|solution|zero)s?\s+(?:are|is)\s+(?:x\s*=\s*)?({num})(?:\s*(?:and|,)\s*(?:x\s*=\s*)?({num}))*".format(num=_NUM),
        r"x\s*=\s*({num})\s+(?:is\s+a\s+)?(?:root|solution|zero)".format(num=_NUM),
        r"(?:solving|setting).*?(?:gives?|yields?|we get)\s+x\s*=\s*({num})".format(num=_NUM),
    ]
    for pattern in root_patterns:
        m = re.search(pattern, text_lower)
        if m:
            # Extract all claimed roots
            roots = re.findall(r"x\s*=\s*(" + _NUM + r")|(?:^|\s)(" + _NUM + r")(?:\s|$)", text_lower)
            claimed_roots = [r[0] or r[1] for r in roots if r[0] or r[1]]
            if not claimed_roots:
                claimed_roots = [m.group(1)]
            return ClaimType.ROOT_CLAIM, {
                "claimed_roots": claimed_roots,
            }

    # === FACTORING ===
    # Extract original expression and factored form from LaTeX
    if re.search(r"factor(?:s|ed|ing|ize)", text_lower):
        # Try to extract "A factors as B" or "A = B" from LaTeX blocks
        latex_exprs = re.findall(r"\\\((.+?)\\\)|\$(.+?)\$", text)
        latex_clean = [m[0] or m[1] for m in latex_exprs]
        # Look for an equation with = sign among extracted LaTeX
        for expr in latex_clean:
            if "=" in expr:
                parts = expr.split("=", 1)
                return ClaimType.FACTORING, {
                    "original": parts[0].strip(),
                    "factored": parts[1].strip(),
                }
        # If we found at least 2 expressions, assume first is original, last is factored
        if len(latex_clean) >= 2:
            return ClaimType.FACTORING, {
                "original": latex_clean[0].strip(),
                "factored": latex_clean[-1].strip(),
            }
        return ClaimType.FACTORING, {}

    # === ALGEBRAIC_EQUIV ===
    # Extract equations from LaTeX \( ... \) or $ ... $ blocks
    latex_blocks = re.findall(r"\\\((.+?)\\\)|\$(.+?)\$", text)
    for block in latex_blocks:
        content = (block[0] or block[1]).strip()
        if "=" in content and len(content) < 100:
            parts = content.split("=", 1)
            lhs_raw = parts[0].strip()
            rhs_raw = parts[1].strip()
            has_var = lambda s: bool(re.search(r"[a-zA-Z]", s))
            if has_var(lhs_raw) and len(lhs_raw) > 1 and len(rhs_raw) > 0:
                return ClaimType.ALGEBRAIC_EQUIV, {
                    "lhs": lhs_raw,
                    "rhs": rhs_raw,
                }

    # === NUMERICAL_COMPUTATION ===
    # "expr = number" — the most common testable pattern in LLM solutions
    # Extract from LaTeX: \( expr = number \) or $expr = number$
    latex_comp = re.search(
        r"(?:\$|\\\()\s*(.+?)\s*=\s*(" + _NUM + r")\s*(?:\$|\\\))",
        text,
    )
    if latex_comp:
        expr = latex_comp.group(1).strip()
        val = latex_comp.group(2).strip()
        # Only if expression is short and parseable
        if len(expr) < 50 and re.search(r"[a-zA-Z0-9]", expr):
            return ClaimType.NUMERICAL_EVAL, {
                "expression": expr,
                "claimed_value": val,
            }

    # === DIVISIBILITY ===
    div_match = re.search(
        r"({num})\s+is\s+(?:divisible|a\s+multiple)\s+(?:by|of)\s+({num})".format(num=_NUM),
        text_lower,
    )
    if div_match:
        return ClaimType.DIVISIBILITY, {
            "number": div_match.group(1),
            "divisor": div_match.group(2),
        }

    # === MODULAR ===
    mod_match = re.search(
        r"({num})\s*(?:≡|\\equiv|=)\s*({num})\s*\(?(?:mod|%)\s*({num})".format(num=_NUM),
        text_lower,
    )
    if mod_match:
        return ClaimType.MODULAR, {
            "value": mod_match.group(1),
            "remainder": mod_match.group(2),
            "modulus": mod_match.group(3),
        }

    # === NUMERICAL_EVAL ===
    # Patterns: "sin(pi/6) = 1/2", "log(100) = 2", explicit numerical evaluation
    num_eval_match = re.search(
        r"((?:sin|cos|tan|log|ln|exp|sqrt)\s*\([^)]+\))\s*=\s*(" + _NUM + r")",
        text_lower,
    )
    if num_eval_match:
        return ClaimType.NUMERICAL_EVAL, {
            "expression": num_eval_match.group(1),
            "claimed_value": num_eval_match.group(2),
        }

    # === ENUMERATION ===
    enum_match = re.search(
        r"(?:there\s+are|total\s+(?:of\s+)?|count\s+is|number\s+of\s+\w+\s+is)\s*({num})".format(num=_NUM),
        text_lower,
    )
    if enum_match:
        return ClaimType.ENUMERATION, {
            "claimed_count": enum_match.group(1),
        }

    # === COORDINATE ===
    coord_match = re.search(
        r"(?:area|distance|length|perimeter|slope)\s+(?:is|=|equals)\s*({num})".format(num=_NUM),
        text_lower,
    )
    if coord_match:
        return ClaimType.COORDINATE, {
            "claimed_value": coord_match.group(1),
            "metric_type": re.search(r"(area|distance|length|perimeter|slope)", text_lower).group(1),
        }

    # === UNTESTABLE (default) ===
    return ClaimType.UNTESTABLE, {}


def classify_all_steps(
    steps: List[str],
    problem_text: str,
) -> List[Tuple[ClaimType, dict]]:
    """Classify all steps in a solution.

    Args:
        steps: List of step texts (from parse_nl_steps).
        problem_text: The original problem text.

    Returns:
        List of (ClaimType, operands) tuples.
    """
    return [
        classify_claim(step, problem_text, i)
        for i, step in enumerate(steps)
    ]
