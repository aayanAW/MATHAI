"""Test generator: dispatches to templates and validates extraction.

The six-stage extraction-validation pipeline runs BEFORE any test
is generated. If any stage fails, the step falls to UNTESTABLE.
"""
from typing import List, Optional

from .claim_classifier import ClaimType
from .test_templates import (
    TestResult,
    generate_algebraic_equiv_test,
    generate_divisibility_test,
    generate_enumeration_test,
    generate_factoring_test,
    generate_final_answer_test,
    generate_modular_test,
    generate_numerical_eval_test,
    generate_root_claim_test,
)


def generate_test(
    step_text: str,
    claim_type: ClaimType,
    operands: dict,
    problem_text: str,
    gold_answer: str = "",
) -> TestResult:
    """Generate a property test for a classified step.

    Dispatches to the appropriate template based on claim type.
    Returns TestResult with script=None if validation fails.

    Args:
        step_text: The reasoning step text.
        claim_type: The classified claim type.
        operands: Extracted operands from the classifier.
        problem_text: Original problem text.
        gold_answer: Gold answer (for FINAL_ANSWER only).

    Returns:
        TestResult with script, independence level, and validation notes.
    """
    if claim_type == ClaimType.UNTESTABLE:
        return TestResult(None, ClaimType.UNTESTABLE, "failed", "UNTESTABLE claim type")

    if claim_type == ClaimType.ROOT_CLAIM:
        roots = operands.get("claimed_roots", [])
        if not roots:
            return TestResult(None, claim_type, "failed", "No roots extracted")
        return generate_root_claim_test(roots, problem_text)

    if claim_type == ClaimType.ALGEBRAIC_EQUIV:
        lhs = operands.get("lhs", "")
        rhs = operands.get("rhs", "")
        if not lhs or not rhs:
            return TestResult(None, claim_type, "failed", "Missing LHS or RHS")
        return generate_algebraic_equiv_test(lhs, rhs, problem_text)

    if claim_type == ClaimType.DIVISIBILITY:
        number = operands.get("number", "")
        divisor = operands.get("divisor", "")
        if not number or not divisor:
            return TestResult(None, claim_type, "failed", "Missing number or divisor")
        return generate_divisibility_test(number, divisor)

    if claim_type == ClaimType.MODULAR:
        value = operands.get("value", "")
        remainder = operands.get("remainder", "")
        modulus = operands.get("modulus", "")
        if not all([value, remainder, modulus]):
            return TestResult(None, claim_type, "failed", "Missing modular operands")
        return generate_modular_test(value, remainder, modulus)

    if claim_type == ClaimType.NUMERICAL_EVAL:
        expression = operands.get("expression", "")
        claimed = operands.get("claimed_value", "")
        if not expression or not claimed:
            return TestResult(None, claim_type, "failed", "Missing expression or value")
        return generate_numerical_eval_test(expression, claimed)

    if claim_type == ClaimType.FINAL_ANSWER:
        claimed = operands.get("claimed_answer", "")
        if not claimed:
            return TestResult(None, claim_type, "failed", "No answer extracted")
        if not gold_answer:
            return TestResult(None, claim_type, "failed", "No gold answer for comparison")
        return generate_final_answer_test(claimed, gold_answer)

    if claim_type == ClaimType.FACTORING:
        original = operands.get("original", "")
        factored = operands.get("factored", "")
        if not original or not factored:
            return TestResult(None, claim_type, "failed", "Missing original or factored form")
        return generate_factoring_test(original, factored, problem_text)

    if claim_type == ClaimType.ENUMERATION:
        count = operands.get("claimed_count", "")
        if not count:
            return TestResult(None, claim_type, "failed", "No count extracted")
        return generate_enumeration_test(count, problem_text)

    # COORDINATE — template not yet implemented
    return TestResult(None, claim_type, "failed",
                     f"Template not yet implemented for {claim_type.value}")


def generate_tests_for_solution(
    steps: List[str],
    classifications: list,
    problem_text: str,
    gold_answer: str = "",
) -> List[TestResult]:
    """Generate tests for all steps in a solution.

    Args:
        steps: List of step texts.
        classifications: List of (ClaimType, operands) from classify_all_steps.
        problem_text: Original problem text.
        gold_answer: Gold answer for final answer test.

    Returns:
        List of TestResult objects (one per step).
    """
    results = []
    for step, (ctype, operands) in zip(steps, classifications):
        result = generate_test(step, ctype, operands, problem_text, gold_answer)
        results.append(result)
    return results
