"""Parse chain-of-thought solutions into individual steps.

Handles multiple formatting conventions:
- "## Step k: ..." (our standard format)
- "Step k: ..." or "Step k." or "Step k)"
- Numbered lists: "1. ...", "1) ..."
"""
import re
from typing import List, Tuple


def parse_nl_steps(solution: str) -> List[str]:
    """Parse a CoT solution into individual reasoning steps.

    Returns list of step strings. Each step includes its content
    but not the step header itself.
    """
    # Try "## Step k:" format first (our standard)
    pattern = r"##\s*Step\s+\d+[:.]\s*"
    parts = re.split(pattern, solution)
    if len(parts) > 1:
        return [p.strip() for p in parts[1:] if p.strip()]

    # Try "Step k:" or "Step k." format
    pattern = r"(?:^|\n)\s*Step\s+\d+[:.)\s]"
    parts = re.split(pattern, solution)
    if len(parts) > 1:
        return [p.strip() for p in parts[1:] if p.strip()]

    # Try numbered list: "1. " or "1) "
    pattern = r"(?:^|\n)\s*\d+[.)]\s+"
    parts = re.split(pattern, solution)
    if len(parts) > 1:
        return [p.strip() for p in parts[1:] if p.strip()]

    # Fallback: split on double newlines
    parts = solution.strip().split("\n\n")
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]

    # Last resort: return the whole solution as one step
    return [solution.strip()]


def parse_verification_blocks(script: str) -> List[str]:
    """Parse a cumulative verification script into per-step code blocks.

    Splits on '# === STEP k ===' delimiters.
    Returns list of code strings, one per step.
    """
    # Split on step delimiters
    pattern = r"#\s*===\s*STEP\s+\d+\s*==="
    parts = re.split(pattern, script)

    if len(parts) <= 1:
        # Try alternative delimiter: "# Step k"
        pattern = r"#\s*Step\s+\d+[:\s]"
        parts = re.split(pattern, script)

    if len(parts) <= 1:
        # No delimiters found — treat whole script as one block
        return [script.strip()]

    # First part is usually imports/setup
    blocks = []
    setup = parts[0].strip()
    for i, part in enumerate(parts[1:]):
        if part.strip():
            blocks.append(part.strip())

    return blocks


def extract_assertions(script: str) -> List[str]:
    """Extract all assert statements from a Python script.

    Returns list of assertion strings (the full `assert ...` line).
    """
    assertions = []
    for line in script.split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert "):
            assertions.append(stripped)
    return assertions


def count_steps_alignment(nl_steps: List[str], code_blocks: List[str]) -> bool:
    """Check if NL steps and code blocks are 1:1 aligned.

    Returns True if counts match.
    """
    return len(nl_steps) == len(code_blocks)


def extract_step_from_error(error_msg: str) -> int:
    """Extract step number from an assertion error message.

    Looks for "FAIL:Step k:" pattern in the error message.
    Falls back to assertion position if pattern not found.

    Returns 0-indexed step number, or -1 if unparseable.
    """
    # Try FAIL:Step k: pattern
    match = re.search(r"FAIL:Step\s*(\d+)", error_msg)
    if match:
        return int(match.group(1)) - 1  # Convert to 0-indexed

    # Try Step k pattern
    match = re.search(r"Step\s*(\d+)", error_msg)
    if match:
        return int(match.group(1)) - 1

    return -1


def extract_python_code(response: str) -> str:
    """Extract Python code from a model response.

    Handles ```python...``` code blocks and bare code.
    """
    # Try to find ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return "\n\n".join(matches)

    # Try ``` ... ``` blocks (without python tag)
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Filter to blocks that look like Python
        py_blocks = [m for m in matches if any(
            kw in m for kw in ["import", "assert", "def ", "print(", "from "]
        )]
        if py_blocks:
            return "\n\n".join(py_blocks)

    # No code blocks — try to find code by looking for import/assert lines
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "assert ", "print(")) or in_code:
            code_lines.append(line)
            in_code = True
        elif in_code and stripped and not stripped.startswith("#"):
            # Check if this looks like continuation
            if any(stripped.startswith(c) for c in [
                "x", "y", "z", "eq", "result", "answer", "claimed", "for ", "if ",
            ]):
                code_lines.append(line)
            else:
                in_code = False

    return "\n".join(code_lines) if code_lines else ""


if __name__ == "__main__":
    # Test step parsing
    test_solution = """## Step 1: Identify the equation
We need to solve x^2 - 5x + 6 = 0.

## Step 2: Factor the quadratic
x^2 - 5x + 6 = (x - 2)(x - 3)

## Step 3: Find the solutions
Setting each factor to 0: x = 2 or x = 3.

## Step 4: Compute the sum
The sum of solutions is 2 + 3 = 5.

The answer is \\boxed{5}."""

    steps = parse_nl_steps(test_solution)
    print(f"Parsed {len(steps)} steps:")
    for i, s in enumerate(steps):
        print(f"  Step {i+1}: {s[:60]}...")

    # Test assertion extraction
    test_script = """from sympy import *
x = symbols('x')
# === STEP 1 ===
eq = x**2 - 5*x + 6
# === STEP 2 ===
assert expand((x-2)*(x-3)) == expand(eq), "FAIL:Step 2: factoring"
# === STEP 3 ===
assert eq.subs(x, 2) == 0, "FAIL:Step 3: x=2 not root"
assert eq.subs(x, 3) == 0, "FAIL:Step 3: x=3 not root"
"""
    assertions = extract_assertions(test_script)
    print(f"\nExtracted {len(assertions)} assertions:")
    for a in assertions:
        print(f"  {a}")
