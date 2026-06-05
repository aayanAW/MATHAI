"""PBT pipeline: orchestrates claim classification, test generation, execution.

This is the main entry point for running specification-grounded
randomized checking on a single problem's solution.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..eval.answer_check import answers_equivalent, extract_model_answer
from ..exever.executor import execute_verification_script
from ..exever.step_parser import parse_nl_steps
from .claim_classifier import ClaimType, classify_all_steps
from .test_generator import generate_tests_for_solution
from .test_templates import TestResult

logger = logging.getLogger(__name__)


@dataclass
class PBTStepResult:
    """Result for a single step."""
    step_index: int
    step_text: str
    claim_type: ClaimType
    label: str  # "PASS", "FAIL", "UNTESTABLE"
    independence: str  # "fully", "partially", "failed"
    validation_notes: str = ""
    execution_stdout: str = ""
    execution_stderr: str = ""


@dataclass
class PBTResult:
    """Result from running PBT on a single problem's solution."""
    problem_id: str
    problem_text: str
    solution_text: str
    gold_answer: str

    # Step-level
    n_steps: int = 0
    step_results: List[PBTStepResult] = field(default_factory=list)

    # Aggregate
    n_testable: int = 0  # Steps where a test was generated
    n_fully_independent: int = 0
    n_partially_independent: int = 0
    n_untestable: int = 0
    n_passed: int = 0
    n_failed: int = 0

    # Answer
    answer_predicted: str = ""
    answer_correct: bool = False

    # FPVR (post-hoc, requires gold answer)
    all_tested_pass: bool = False  # All testable steps passed
    fpvr: Optional[bool] = None  # True if all pass but answer wrong

    def to_dict(self) -> Dict:
        """Serialize for JSON output."""
        return {
            "problem_id": self.problem_id,
            "n_steps": self.n_steps,
            "n_testable": self.n_testable,
            "n_fully_independent": self.n_fully_independent,
            "n_partially_independent": self.n_partially_independent,
            "n_untestable": self.n_untestable,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "answer_predicted": self.answer_predicted,
            "answer_correct": self.answer_correct,
            "all_tested_pass": self.all_tested_pass,
            "fpvr": self.fpvr,
            "step_results": [
                {
                    "step_index": sr.step_index,
                    "claim_type": sr.claim_type.value,
                    "label": sr.label,
                    "independence": sr.independence,
                    "validation_notes": sr.validation_notes,
                }
                for sr in self.step_results
            ],
        }


def run_pbt(
    problem: str,
    solution: str,
    gold_answer: str,
    problem_id: str = "",
    execution_timeout: int = 30,
) -> PBTResult:
    """Run specification-grounded randomized checking on a solution.

    Args:
        problem: The math problem text.
        solution: The model's NL solution.
        gold_answer: The gold standard answer.
        problem_id: Identifier for this problem.
        execution_timeout: Timeout for each test execution.

    Returns:
        PBTResult with all step-level and aggregate metrics.
    """
    result = PBTResult(
        problem_id=problem_id,
        problem_text=problem,
        solution_text=solution,
        gold_answer=gold_answer,
    )

    # Extract answer
    result.answer_predicted = extract_model_answer(solution)
    result.answer_correct, _ = answers_equivalent(result.answer_predicted, gold_answer)

    # Parse steps
    steps = parse_nl_steps(solution)
    result.n_steps = len(steps)

    if not steps:
        return result

    # Classify all steps
    classifications = classify_all_steps(steps, problem)

    # Generate tests
    test_results = generate_tests_for_solution(
        steps, classifications, problem, gold_answer
    )

    # Execute each test
    for i, (step, (ctype, _), test_result) in enumerate(
        zip(steps, classifications, test_results)
    ):
        step_result = PBTStepResult(
            step_index=i,
            step_text=step[:200],  # Truncate for storage
            claim_type=ctype,
            label="UNTESTABLE",
            independence=test_result.independence,
            validation_notes=test_result.validation_notes,
        )

        if test_result.script is not None:
            # Execute the test
            exec_result = execute_verification_script(
                test_result.script, timeout=execution_timeout
            )
            step_result.execution_stdout = exec_result.stdout[:500]
            step_result.execution_stderr = exec_result.stderr[:500]

            if exec_result.success:
                step_result.label = "PASS"
                result.n_passed += 1
            else:
                step_result.label = "FAIL"
                result.n_failed += 1

            result.n_testable += 1
            if test_result.independence == "fully":
                result.n_fully_independent += 1
            elif test_result.independence == "partially":
                result.n_partially_independent += 1
        else:
            result.n_untestable += 1

        result.step_results.append(step_result)

    # Compute aggregate metrics
    tested_labels = [sr.label for sr in result.step_results if sr.label != "UNTESTABLE"]
    result.all_tested_pass = all(l == "PASS" for l in tested_labels) if tested_labels else False

    # FPVR: false positive = all tests pass but answer is wrong
    if result.all_tested_pass and tested_labels:
        result.fpvr = not result.answer_correct
    else:
        result.fpvr = None  # Not applicable (some test failed or no tests)

    return result
