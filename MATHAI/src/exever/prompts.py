"""Prompt templates for ExeVer pipeline.

All prompts are designed for the two-pass architecture:
- Pass 1: Standard CoT solution generation
- Pass 2: Verification code generation conditioned on NL solution
- Repair: Targeted fix of a specific failing step
"""

# === PASS 1: Solution Generation (Standard CoT) ===

SOLVE_PROMPT = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

## Step 2: [brief title]
[reasoning and computation for this step]

...continue for all steps...

At the end, state your final answer as: The answer is \\boxed{{answer}}.

Problem: {problem}"""


# === PASS 2: Verification Code Generation ===

VERIFY_PROMPT = """Below is a step-by-step math solution. Write a CUMULATIVE Python/SymPy script that verifies each step via transition assertions.

RULES (follow exactly):
1. The script is CUMULATIVE — each step's code builds on variables from prior steps.
2. Start with: from sympy import *
3. For EACH step in the solution, write a code section with delimiter:
   # === STEP k ===
4. In each section:
   - Assign key variables/expressions from this step
   - Write assert statements that check the step's CLAIM is consistent with prior state
   - Use transition assertions: verify the CLAIMED result, do NOT re-derive independently
   - Label: assert condition, "FAIL:Step k: description"
5. Generate EXACTLY one code section per solution step. Do NOT skip or merge steps.
6. Use expand() for algebraic comparison (NOT simplify(), which can hang).
7. Use .subs() for substitution checks.
8. At the end: print("ANSWER:", final_answer)

GOOD assertion (checks transition):
  assert expand(claimed_factored) == expand(original_eq), "FAIL:Step 2: factoring"

BAD assertion (re-derives — this is just solving in code):
  assert solve(eq) == [2, 3], "FAIL:Step 2: solving"

BAD assertion (trivial — tests nothing):
  assert True, "FAIL:Step 1: setup"

Solution to verify:
{solution}

Write the complete cumulative Python/SymPy verification script:"""


# === REPAIR PROMPT ===

REPAIR_PROMPT = """You are solving the following math problem. Your previous solution had an error at Step {step_num}.

Problem: {problem}

Verified correct steps (keep these exactly):
{verified_prefix}

The error was at Step {step_num}:
{failed_step}

Error message from verification:
{error_message}

Please provide:
1. A corrected Step {step_num} and all subsequent steps
2. Format each step as "## Step k: [title]" followed by reasoning
3. State your final answer as: The answer is \\boxed{{answer}}.

Continue from Step {step_num}:"""


# === INTERLEAVED PROMPT (for ablation — single-pass) ===

INTERLEAVED_PROMPT = """Solve the following math problem step by step, AND write Python/SymPy verification code for each step.

Format:
## Step k: [title]
[mathematical reasoning]

```python
# === STEP k ===
[Python/SymPy code that verifies this step's claims]
assert condition, "FAIL:Step k: description"
```

The code is CUMULATIVE — each step builds on prior variables.
Use transition assertions (check claims, don't re-derive).
At the end: print("ANSWER:", final_answer)

Problem: {problem}"""


# === SPOT-CHECK TEMPLATE ===
# Appended to verification scripts for model-free cross-checks

SPOT_CHECK_TEMPLATE = """
# === SPOT CHECKS (model-free) ===
try:
    # Sanity bound checks
    _answer = {answer_var}
    if isinstance(_answer, (int, float, sympy.Rational, sympy.Integer, sympy.Float)):
        _answer_float = float(_answer)
        # Check for NaN/Inf
        import math
        assert not math.isnan(_answer_float), "SPOT:Answer is NaN"
        assert not math.isinf(_answer_float), "SPOT:Answer is Inf"
except Exception:
    pass  # Spot checks are best-effort, don't crash the script
"""


def format_solve_prompt(problem: str) -> str:
    """Format the solution generation prompt."""
    return SOLVE_PROMPT.format(problem=problem)


def format_verify_prompt(solution: str) -> str:
    """Format the verification code generation prompt."""
    return VERIFY_PROMPT.format(solution=solution)


def format_repair_prompt(
    problem: str,
    verified_prefix: str,
    failed_step: str,
    step_num: int,
    error_message: str,
) -> str:
    """Format the repair prompt."""
    return REPAIR_PROMPT.format(
        problem=problem,
        verified_prefix=verified_prefix,
        failed_step=failed_step,
        step_num=step_num,
        error_message=error_message,
    )


def format_interleaved_prompt(problem: str) -> str:
    """Format the single-pass interleaved prompt (for ablation)."""
    return INTERLEAVED_PROMPT.format(problem=problem)
