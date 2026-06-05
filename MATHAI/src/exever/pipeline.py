"""Full ExeVer pipeline: two-pass verification + repair + backtracking.

Implements Algorithm 1 (ExeVer) and Algorithm 2 (Multi-Sample ExeVer)
from the proposal.
"""
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..eval.answer_check import answers_equivalent, extract_model_answer
from ..eval.metrics import verification_coverage
from .executor import ExecutionResult, execute_verification_script
from .prompts import format_repair_prompt, format_solve_prompt, format_verify_prompt
from .step_parser import (
    count_steps_alignment,
    extract_assertions,
    extract_python_code,
    extract_step_from_error,
    parse_nl_steps,
    parse_verification_blocks,
)

logger = logging.getLogger(__name__)


@dataclass
class ExeVerResult:
    """Result from running ExeVer on a single problem."""
    problem: str
    problem_id: str = ""
    # Solution
    nl_solution: str = ""
    verification_script: str = ""
    # Execution
    execution_result: Optional[ExecutionResult] = None
    # Metrics
    answer_predicted: str = ""
    answer_correct: bool = False
    verification_score: float = 0.0
    coverage: float = 0.0
    # Step-level details
    n_steps: int = 0
    n_assertions: int = 0
    n_trivial_assertions: int = 0
    step_verdicts: List[str] = field(default_factory=list)
    # Repair info
    n_repair_rounds: int = 0
    n_backtracks: int = 0
    repair_history: List[Dict] = field(default_factory=list)
    # Meta — FPVR (False Positive Verification Rate) is computed post-hoc
    # in analysis scripts, NOT here. The pipeline does not use the gold answer
    # for any decision-making during inference.
    # Legacy field retained for backward compatibility with result JSONs:
    echo_chamber: Optional[bool] = None  # DEPRECATED: was P(wrong | ALL_PASS), tautological

    def to_dict(self) -> Dict:
        return {
            "problem_id": self.problem_id,
            "answer_predicted": self.answer_predicted,
            "answer_correct": self.answer_correct,
            "verification_score": self.verification_score,
            "coverage": self.coverage,
            "n_steps": self.n_steps,
            "n_assertions": self.n_assertions,
            "n_trivial_assertions": self.n_trivial_assertions,
            "step_verdicts": self.step_verdicts,
            "n_repair_rounds": self.n_repair_rounds,
            "n_backtracks": self.n_backtracks,
            "echo_chamber": self.echo_chamber,
            "verdict": self.execution_result.verdict if self.execution_result else "NO_EXEC",
        }


# Type alias for the LLM generation function
GenerateFn = Callable[[str], str]


def run_exever(
    problem: str,
    gold_answer: str,
    solver_fn: GenerateFn,
    verifier_fn: GenerateFn,
    problem_id: str = "",
    max_repairs: int = 2,
    max_backtrack: int = 1,
    execution_timeout: int = 30,
) -> ExeVerResult:
    """Run ExeVer on a single problem (Algorithm 1).

    Args:
        problem: The math problem text.
        gold_answer: The gold standard answer.
        solver_fn: Function(prompt) -> response for Pass 1 (NL solution).
        verifier_fn: Function(prompt) -> response for Pass 2 (verification code).
            May be same or different from solver_fn (cross-model).
        problem_id: Identifier for this problem.
        max_repairs: Maximum repair attempts per step.
        max_backtrack: Maximum backtracking depth.
        execution_timeout: Timeout for script execution.

    Returns:
        ExeVerResult with all details.
    """
    result = ExeVerResult(problem=problem, problem_id=problem_id)

    # === Pass 1: Generate NL solution ===
    solve_prompt = format_solve_prompt(problem)
    nl_solution = solver_fn(solve_prompt)
    result.nl_solution = nl_solution
    result.answer_predicted = extract_model_answer(nl_solution)
    result.answer_correct, _ = answers_equivalent(result.answer_predicted, gold_answer)

    # Parse steps
    nl_steps = parse_nl_steps(nl_solution)
    result.n_steps = len(nl_steps)

    # === Pass 2: Generate verification code ===
    verify_prompt = format_verify_prompt(nl_solution)
    verify_response = verifier_fn(verify_prompt)
    verification_script = extract_python_code(verify_response)
    result.verification_script = verification_script

    if not verification_script.strip():
        logger.warning(f"Empty verification script for {problem_id}")
        return result

    # Count assertions
    assertions = extract_assertions(verification_script)
    result.n_assertions = len(assertions)
    result.n_trivial_assertions = sum(1 for a in assertions if _is_trivial(a))

    # === Execute and repair loop ===
    repairs_at: Dict[int, int] = defaultdict(int)
    backtrack_count = 0
    current_script = verification_script
    current_solution = nl_solution

    for round_num in range(max_repairs * (max_backtrack + 1)):
        # Execute
        exec_result = execute_verification_script(current_script, timeout=execution_timeout)
        result.execution_result = exec_result
        result.verification_script = current_script

        if exec_result.success:
            # All assertions passed
            result.verification_score = 1.0
            result.coverage = 1.0 if exec_result.assertions_found > 0 else 0.0
            # NOTE: echo_chamber field is DEPRECATED (tautological: = not answer_correct).
            # FPVR is computed post-hoc in analysis scripts from gold answers.
            # The pipeline itself should NOT use gold answers for inference decisions.
            result.echo_chamber = not result.answer_correct  # legacy, do not use in paper
            # Re-extract answer from script output if available
            if exec_result.answer_extracted:
                script_answer = exec_result.answer_extracted
                script_correct, _ = answers_equivalent(script_answer, gold_answer)
                if script_correct and not result.answer_correct:
                    # Script got right answer but NL didn't — interesting case
                    result.answer_predicted = script_answer
                    result.answer_correct = True
            break

        if exec_result.runtime_error or exec_result.timeout:
            # Script crashed or timed out — can't repair
            result.verification_score = 0.0
            break

        if exec_result.assertion_error:
            j = exec_result.error_step
            if j < 0:
                # Can't identify which step failed
                result.verification_score = 0.0
                break

            # Record repair attempt
            result.repair_history.append({
                "round": round_num,
                "failed_step": j,
                "error_message": exec_result.error_message,
            })

            # Check if we've exhausted repairs at this step
            repairs_at[j] += 1
            if repairs_at[j] > max_repairs:
                if backtrack_count < max_backtrack:
                    # Backtrack: try repairing step j-1 instead
                    j = max(0, j - 1)
                    backtrack_count += 1
                    result.n_backtracks += 1
                    repairs_at[j] += 1
                else:
                    # Exhausted all options
                    result.verification_score = 0.0
                    break

            result.n_repair_rounds += 1

            # Generate repair
            current_nl_steps = parse_nl_steps(current_solution)
            verified_prefix = "\n\n".join(
                f"## Step {i+1}: {s}" for i, s in enumerate(current_nl_steps[:j])
            )
            failed_step_text = current_nl_steps[j] if j < len(current_nl_steps) else ""

            repair_prompt = format_repair_prompt(
                problem=problem,
                verified_prefix=verified_prefix,
                failed_step=failed_step_text,
                step_num=j + 1,
                error_message=exec_result.error_message,
            )
            repaired = solver_fn(repair_prompt)

            # Reconstruct full solution
            prefix_text = "\n\n".join(
                f"## Step {i+1}: {s}" for i, s in enumerate(current_nl_steps[:j])
            )
            current_solution = prefix_text + "\n\n" + repaired
            result.nl_solution = current_solution
            result.answer_predicted = extract_model_answer(current_solution)
            result.answer_correct, _ = answers_equivalent(
                result.answer_predicted, gold_answer
            )

            # Re-generate verification code for repaired solution
            verify_prompt = format_verify_prompt(current_solution)
            verify_response = verifier_fn(verify_prompt)
            current_script = extract_python_code(verify_response)

            if not current_script.strip():
                result.verification_score = 0.0
                break

    return result


def run_multi_sample_exever(
    problem: str,
    gold_answer: str,
    solver_fn: GenerateFn,
    verifier_fn: GenerateFn,
    problem_id: str = "",
    n_samples: int = 8,
    max_repairs: int = 2,
    max_backtrack: int = 1,
) -> Tuple[ExeVerResult, List[ExeVerResult]]:
    """Run Multi-Sample ExeVer (Algorithm 2).

    Generate N solutions, run ExeVer on each, select best by
    verification score + majority vote.

    Returns:
        (best_result, all_results)
    """
    all_results = []
    for i in range(n_samples):
        r = run_exever(
            problem=problem,
            gold_answer=gold_answer,
            solver_fn=solver_fn,
            verifier_fn=verifier_fn,
            problem_id=f"{problem_id}_sample{i}",
            max_repairs=max_repairs,
            max_backtrack=max_backtrack,
        )
        all_results.append(r)

    # Select best solution
    # Step 1: Filter by verification score ≥ 0.5
    verified = [r for r in all_results if r.verification_score >= 0.5]

    if verified:
        # Step 2: Weighted majority vote with symbolic equivalence grouping
        # Group by symbolic equivalence to avoid vote-splitting
        groups: List[tuple] = []  # (canonical_answer, total_weight, results)
        for r in verified:
            merged = False
            for i, (canonical, weight, members) in enumerate(groups):
                eq, _ = answers_equivalent(r.answer_predicted, canonical)
                if eq:
                    groups[i] = (canonical, weight + r.verification_score, members + [r])
                    merged = True
                    break
            if not merged:
                groups.append((r.answer_predicted, r.verification_score, [r]))

        # Pick group with highest total weight
        groups.sort(key=lambda x: x[1], reverse=True)
        best_group = groups[0]

        # Step 3: Among group members, pick highest V(S)
        best = max(best_group[2], key=lambda r: r.verification_score)
    else:
        # Fallback: standard majority voting with symbolic equivalence
        groups: List[tuple] = []
        for r in all_results:
            merged = False
            for i, (canonical, count, members) in enumerate(groups):
                eq, _ = answers_equivalent(r.answer_predicted, canonical)
                if eq:
                    groups[i] = (canonical, count + 1, members + [r])
                    merged = True
                    break
            if not merged:
                groups.append((r.answer_predicted, 1, [r]))

        groups.sort(key=lambda x: x[1], reverse=True)
        best = groups[0][2][0] if groups else all_results[0]

    return best, all_results


def _is_trivial(assertion: str) -> bool:
    """Quick check if an assertion is trivial."""
    body = assertion.replace("assert ", "", 1).strip()
    if body.startswith("True"):
        return True
    if "==" in body:
        parts = body.split("==", 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip().split(",")[0].strip()
        if lhs == rhs:
            return True
    if body.startswith("isinstance("):
        return True
    return False
