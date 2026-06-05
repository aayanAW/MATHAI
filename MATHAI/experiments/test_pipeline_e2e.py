"""End-to-end test of the ExeVer pipeline using mock responses.

Simulates the full two-pass pipeline with realistic mock LLM outputs
to verify the pipeline logic works correctly before GPU deployment.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exever.pipeline import run_exever, run_multi_sample_exever
from src.exever.executor import execute_verification_script
from src.exever.step_parser import parse_nl_steps, extract_python_code, extract_assertions
from src.exever.prompts import format_solve_prompt, format_verify_prompt
from src.eval.answer_check import answers_equivalent, extract_model_answer
from src.eval.metrics import assertion_quality, verification_coverage
from src.inference.model_wrapper import MockModel


# === Mock Responses ===

MOCK_SOLUTION_CORRECT = """## Step 1: Set up the equation
We need to find the sum of the solutions of x^2 - 5x + 6 = 0.

## Step 2: Factor the quadratic
We can factor this as x^2 - 5x + 6 = (x - 2)(x - 3).

## Step 3: Find the solutions
Setting each factor to zero: x - 2 = 0 gives x = 2, and x - 3 = 0 gives x = 3.

## Step 4: Compute the sum
The sum of the solutions is 2 + 3 = 5.

The answer is \\boxed{5}."""

MOCK_VERIFICATION_CORRECT = """```python
from sympy import *

x = symbols('x')

# === STEP 1 ===
eq = x**2 - 5*x + 6

# === STEP 2 ===
claimed_factored = (x - 2) * (x - 3)
assert expand(claimed_factored) == expand(eq), "FAIL:Step 2: factoring incorrect"

# === STEP 3 ===
assert eq.subs(x, 2) == 0, "FAIL:Step 3: x=2 is not a root"
assert eq.subs(x, 3) == 0, "FAIL:Step 3: x=3 is not a root"

# === STEP 4 ===
claimed_sum = 2 + 3
assert claimed_sum == 5, "FAIL:Step 4: sum incorrect"

print("ANSWER:", claimed_sum)
```"""

MOCK_SOLUTION_WITH_ERROR = """## Step 1: Set up the equation
We need to find the sum of the solutions of x^2 - 5x + 6 = 0.

## Step 2: Factor the quadratic
We can factor this as x^2 - 5x + 6 = (x - 1)(x - 6).

## Step 3: Find the solutions
Setting each factor to zero: x = 1 and x = 6.

## Step 4: Compute the sum
The sum of the solutions is 1 + 6 = 7.

The answer is \\boxed{7}."""

MOCK_VERIFICATION_WITH_ERROR = """```python
from sympy import *

x = symbols('x')

# === STEP 1 ===
eq = x**2 - 5*x + 6

# === STEP 2 ===
claimed_factored = (x - 1) * (x - 6)
assert expand(claimed_factored) == expand(eq), "FAIL:Step 2: factoring incorrect"

# === STEP 3 ===
assert eq.subs(x, 1) == 0, "FAIL:Step 3: x=1 is not a root"

# === STEP 4 ===
claimed_sum = 1 + 6
assert claimed_sum == 7, "FAIL:Step 4: sum incorrect"

print("ANSWER:", claimed_sum)
```"""


def test_correct_solution():
    """Test pipeline with a correct solution."""
    print("=" * 60)
    print("TEST 1: Correct solution")
    print("=" * 60)

    # Create mock model that returns the correct solution
    model = MockModel(responses={
        "Solve the following": MOCK_SOLUTION_CORRECT,
        "Write a CUMULATIVE": MOCK_VERIFICATION_CORRECT,
    })

    result = run_exever(
        problem="What is the sum of the solutions of x^2 - 5x + 6 = 0?",
        gold_answer="5",
        solver_fn=model.generate,
        verifier_fn=model.generate,
        problem_id="test_correct",
    )

    print(f"  Answer: {result.answer_predicted}")
    print(f"  Correct: {result.answer_correct}")
    print(f"  Verification score: {result.verification_score}")
    print(f"  Steps: {result.n_steps}")
    print(f"  Assertions: {result.n_assertions}")
    print(f"  Repair rounds: {result.n_repair_rounds}")
    print(f"  Verdict: {result.execution_result.verdict if result.execution_result else 'N/A'}")

    assert result.answer_correct, "Answer should be correct"
    assert result.verification_score == 1.0, "Verification score should be 1.0"
    assert result.n_repair_rounds == 0, "No repairs needed"
    print("  PASSED!")


def test_error_detection():
    """Test pipeline detects errors in wrong solutions."""
    print("\n" + "=" * 60)
    print("TEST 2: Error detection")
    print("=" * 60)

    model = MockModel(responses={
        "Solve the following": MOCK_SOLUTION_WITH_ERROR,
        "Write a CUMULATIVE": MOCK_VERIFICATION_WITH_ERROR,
    })

    result = run_exever(
        problem="What is the sum of the solutions of x^2 - 5x + 6 = 0?",
        gold_answer="5",
        solver_fn=model.generate,
        verifier_fn=model.generate,
        problem_id="test_error",
        max_repairs=0,  # No repair, just detect
    )

    print(f"  Answer: {result.answer_predicted}")
    print(f"  Correct: {result.answer_correct}")
    print(f"  Verification score: {result.verification_score}")
    print(f"  Verdict: {result.execution_result.verdict if result.execution_result else 'N/A'}")

    assert not result.answer_correct, "Answer should be wrong"
    assert result.verification_score == 0.0, "Score should be 0 (assertion failed)"
    if result.execution_result:
        assert result.execution_result.assertion_error, "Should detect assertion error"
        print(f"  Failed at step: {result.execution_result.error_step}")
        print(f"  Error: {result.execution_result.error_message}")
    print("  PASSED!")


def test_verification_execution():
    """Test direct execution of verification scripts."""
    print("\n" + "=" * 60)
    print("TEST 3: Verification script execution")
    print("=" * 60)

    # Extract and execute the correct verification
    code = extract_python_code(MOCK_VERIFICATION_CORRECT)
    result = execute_verification_script(code)
    print(f"  Correct script: verdict={result.verdict}, answer={result.answer_extracted}")
    assert result.success, "Correct script should pass"

    # Extract and execute the wrong verification
    code_wrong = extract_python_code(MOCK_VERIFICATION_WITH_ERROR)
    result_wrong = execute_verification_script(code_wrong)
    print(f"  Wrong script: verdict={result_wrong.verdict}")
    assert result_wrong.assertion_error, "Wrong script should fail assertion"

    print("  PASSED!")


def test_assertion_quality_metric():
    """Test assertion quality classification."""
    print("\n" + "=" * 60)
    print("TEST 4: Assertion quality")
    print("=" * 60)

    assertions = [
        'assert expand(claimed) == expand(eq), "FAIL:Step 1"',  # Non-trivial
        'assert eq.subs(x, 2) == 0, "FAIL:Step 2"',  # Non-trivial
        'assert True',  # Trivial
        'assert isinstance(x, Symbol)',  # Trivial
        'assert x == x',  # Trivial
    ]

    aq = assertion_quality(assertions)
    print(f"  Total: {aq['total']}")
    print(f"  Trivial: {aq['trivial']}")
    print(f"  Non-trivial: {aq['nontrivial']}")
    print(f"  Quality rate: {aq['quality_rate']:.0%}")

    assert aq["nontrivial"] == 2, f"Expected 2 non-trivial, got {aq['nontrivial']}"
    assert aq["trivial"] == 3, f"Expected 3 trivial, got {aq['trivial']}"
    print("  PASSED!")


def test_answer_checker_on_math():
    """Test answer checker on actual MATH dataset answers."""
    print("\n" + "=" * 60)
    print("TEST 5: Answer checker on MATH dataset")
    print("=" * 60)

    with open("results/math_test_sample_300.json") as f:
        sample = json.load(f)

    # Check that self-comparison works for all gold answers
    n_self_match = 0
    n_fail = 0
    for p in sample[:50]:
        result, method = answers_equivalent(p["answer"], p["answer"])
        if result:
            n_self_match += 1
        else:
            n_fail += 1
            if n_fail <= 3:
                print(f"  Self-match failed: '{p['answer']}' ({method})")

    print(f"  Self-match: {n_self_match}/50")
    assert n_self_match >= 45, f"Too many self-match failures: {n_fail}"
    print("  PASSED!")


def main():
    print("ExeVer End-to-End Pipeline Tests")
    print("=" * 60)

    test_correct_solution()
    test_error_detection()
    test_verification_execution()
    test_assertion_quality_metric()
    test_answer_checker_on_math()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
