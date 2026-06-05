"""Run ExeVer feasibility experiments (Part C of the plan).

This script orchestrates all 5 feasibility experiments:
- Exp 0: Environment setup and sanity checks
- Exp 1: Baseline establishment (CoT pass@1, majority@8)
- Exp 2: Two-pass feasibility (make-or-break)
- Exp 3: Coverage + echo chamber measurement
- Exp 4: Repair + backtracking

Usage:
    python -m experiments.run_feasibility --exp 0  # Setup only
    python -m experiments.run_feasibility --exp 1  # Baselines
    python -m experiments.run_feasibility --exp 2  # Two-pass
    python -m experiments.run_feasibility --exp 3  # Coverage
    python -m experiments.run_feasibility --exp 4  # Repair
    python -m experiments.run_feasibility --exp all  # Everything
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_math import load_math_test, save_problems, load_problems, stratified_sample
from src.eval.answer_check import answers_equivalent, extract_model_answer
from src.eval.metrics import (
    compute_pass_at_k_for_problems,
    aggregate_coverage_by_subject,
    aggregate_coverage_by_level,
    assertion_quality,
)
from src.exever.executor import execute_verification_script, analyze_execution_results
from src.exever.step_parser import extract_assertions, extract_python_code, parse_nl_steps
from src.exever.prompts import format_solve_prompt, format_verify_prompt
from src.inference.model_wrapper import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def exp0_setup():
    """Experiment 0: Environment setup and sanity checks."""
    logger.info("=== Experiment 0: Setup ===")

    # Test SymPy
    logger.info("Testing SymPy...")
    from sympy import symbols, expand, Eq, solve
    x = symbols('x')
    assert expand((x-2)*(x-3)) == x**2 - 5*x + 6
    logger.info("  SymPy OK")

    # Test answer checking
    logger.info("Testing answer checker...")
    assert answers_equivalent("\\frac{1}{2}", "0.5")[0]
    assert answers_equivalent("3", "3")[0]
    assert not answers_equivalent("5", "6")[0]
    logger.info("  Answer checker OK")

    # Test executor
    logger.info("Testing sandboxed executor...")
    test_script = 'from sympy import *\nassert 2 + 2 == 4, "FAIL:Step 1: arithmetic"\nprint("ANSWER: 4")'
    result = execute_verification_script(test_script, timeout=10)
    assert result.success, f"Execution failed: {result.stderr}"
    assert result.answer_extracted == "4"
    logger.info("  Executor OK")

    # Test timeout
    timeout_script = "import time; time.sleep(5)"
    result = execute_verification_script(timeout_script, timeout=2)
    assert result.timeout, "Timeout not working"
    logger.info("  Timeout handling OK")

    # Load MATH dataset
    logger.info("Loading MATH test set...")
    problems = load_math_test()
    logger.info(f"  Loaded {len(problems)} problems")

    # Create stratified sample
    sample = stratified_sample(problems, n_per_level=60)
    logger.info(f"  Stratified sample: {len(sample)} problems")

    # Save
    save_problems(problems, str(RESULTS_DIR / "math_test_full.json"))
    save_problems(sample, str(RESULTS_DIR / "math_test_sample_300.json"))
    logger.info("  Saved to results/")

    # Verify boxed answer extraction
    n_with_answer = sum(1 for p in problems if p["answer"])
    logger.info(f"  Problems with extracted answers: {n_with_answer}/{len(problems)}")

    logger.info("=== Experiment 0 PASSED ===")


def exp1_baseline(backend: str = "mock", model_id: str = ""):
    """Experiment 1: Baseline establishment."""
    logger.info("=== Experiment 1: Baselines ===")

    # Load sample
    sample_path = RESULTS_DIR / "math_test_sample_300.json"
    if not sample_path.exists():
        exp0_setup()
    problems = load_problems(str(sample_path))

    # Create model
    model = create_model(model_id or "Qwen/Qwen2.5-Math-7B-Instruct", backend=backend)

    # Run CoT on all problems
    results = []
    for i, p in enumerate(problems):
        prompt = format_solve_prompt(p["problem"])
        response = model.generate(prompt)
        predicted = extract_model_answer(response)
        correct, method = answers_equivalent(predicted, p["answer"])

        results.append({
            "id": p["id"],
            "level": p["level"],
            "type": p["type"],
            "answer_predicted": predicted,
            "answer_correct": correct,
            "check_method": method,
        })

        if (i + 1) % 50 == 0:
            n_correct = sum(1 for r in results if r["answer_correct"])
            logger.info(f"  Progress: {i+1}/{len(problems)}, accuracy: {n_correct/(i+1):.1%}")

    # Compute metrics
    n_correct = sum(1 for r in results if r["answer_correct"])
    pass_at_1 = n_correct / len(results)
    logger.info(f"\n  CoT pass@1: {pass_at_1:.1%} ({n_correct}/{len(results)})")

    # By level
    from collections import defaultdict
    by_level = defaultdict(list)
    for r in results:
        by_level[r["level"]].append(r["answer_correct"])
    for lv in sorted(by_level):
        acc = sum(by_level[lv]) / len(by_level[lv])
        logger.info(f"  Level {lv}: {acc:.1%} ({sum(by_level[lv])}/{len(by_level[lv])})")

    # Save results
    output = {
        "experiment": "exp1_baseline",
        "model": model_id or "mock",
        "n_problems": len(results),
        "pass_at_1": pass_at_1,
        "results": results,
    }
    with open(RESULTS_DIR / "exp1_baseline.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=== Experiment 1 DONE ===")
    return output


def exp2_two_pass(backend: str = "mock", model_id: str = ""):
    """Experiment 2: Two-pass feasibility (MAKE-OR-BREAK)."""
    logger.info("=== Experiment 2: Two-Pass Feasibility ===")

    sample_path = RESULTS_DIR / "math_test_sample_300.json"
    if not sample_path.exists():
        exp0_setup()
    problems = load_problems(str(sample_path))

    model = create_model(model_id or "Qwen/Qwen2.5-Math-7B-Instruct", backend=backend)

    results = []
    for i, p in enumerate(problems):
        # Pass 1: Generate NL solution
        solve_prompt = format_solve_prompt(p["problem"])
        nl_solution = model.generate(solve_prompt)

        # Parse steps
        nl_steps = parse_nl_steps(nl_solution)

        # Pass 2: Generate verification code
        verify_prompt = format_verify_prompt(nl_solution)
        verify_response = model.generate(verify_prompt)
        verification_code = extract_python_code(verify_response)

        # Analyze verification code quality
        is_valid_python = bool(verification_code.strip())
        assertions = extract_assertions(verification_code)
        aq = assertion_quality(assertions)

        # Execute
        exec_result = None
        if is_valid_python:
            exec_result = execute_verification_script(verification_code, timeout=30)

        # Extract answer from NL solution
        predicted = extract_model_answer(nl_solution)
        correct, _ = answers_equivalent(predicted, p["answer"])

        results.append({
            "id": p["id"],
            "level": p["level"],
            "type": p["type"],
            "n_nl_steps": len(nl_steps),
            "has_valid_python": is_valid_python,
            "n_assertions": len(assertions),
            "assertion_quality": aq,
            "execution_verdict": exec_result.verdict if exec_result else "NO_CODE",
            "execution_success": exec_result.success if exec_result else False,
            "answer_correct": correct,
        })

        if (i + 1) % 50 == 0:
            n_valid = sum(1 for r in results if r["has_valid_python"])
            n_exec = sum(1 for r in results if r["execution_success"])
            logger.info(
                f"  Progress: {i+1}/{len(problems)}, "
                f"valid Python: {n_valid/(i+1):.0%}, "
                f"execution success: {n_exec/(i+1):.0%}"
            )

    # Compute feasibility metrics
    n = len(results)
    script_validity = sum(1 for r in results if r["has_valid_python"]) / n
    execution_rate = sum(
        1 for r in results
        if r["execution_verdict"] in ("ALL_PASS", "FAIL_STEP_0", "FAIL_STEP_1", "FAIL_STEP_2")
        or (r["execution_verdict"] and r["execution_verdict"].startswith("FAIL_STEP"))
    ) / n
    # Scripts that ran (success or assertion fail, not crash)
    ran_scripts = [r for r in results if r["execution_verdict"] not in ("NO_CODE", "ERROR", "TIMEOUT")]
    execution_rate = len(ran_scripts) / n if n > 0 else 0

    avg_quality = sum(r["assertion_quality"]["quality_rate"] for r in results if r["has_valid_python"]) / max(1, sum(1 for r in results if r["has_valid_python"]))

    logger.info(f"\n=== FEASIBILITY GATES ===")
    logger.info(f"  G1: Script validity rate: {script_validity:.0%} (threshold: ≥60%)")
    logger.info(f"  G2: Execution rate: {execution_rate:.0%} (threshold: ≥50%)")
    logger.info(f"  G3: Assertion quality: {avg_quality:.0%} (threshold: ≥50%)")

    # Save
    output = {
        "experiment": "exp2_two_pass",
        "script_validity_rate": script_validity,
        "execution_rate": execution_rate,
        "avg_assertion_quality": avg_quality,
        "results": results,
    }
    with open(RESULTS_DIR / "exp2_two_pass.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=== Experiment 2 DONE ===")
    return output


def exp3_coverage(backend: str = "mock", model_id: str = ""):
    """Experiment 3: Coverage + Echo Chamber measurement."""
    logger.info("=== Experiment 3: Coverage + Echo Chamber ===")

    # Load exp2 results (reuse verification scripts)
    exp2_path = RESULTS_DIR / "exp2_two_pass.json"
    if not exp2_path.exists():
        logger.info("Running exp2 first...")
        exp2_two_pass(backend, model_id)

    with open(exp2_path) as f:
        exp2_data = json.load(f)

    results = exp2_data["results"]

    # Compute coverage by subject
    by_subject = {}
    for r in results:
        subj = r["type"]
        by_subject.setdefault(subj, []).append({
            "success": r["execution_success"],
            "verdict": r["execution_verdict"],
            "correct": r["answer_correct"],
            "n_assertions": r["n_assertions"],
        })

    logger.info("\n  Coverage by subject:")
    for subj in sorted(by_subject):
        items = by_subject[subj]
        n_ran = sum(1 for it in items if it["verdict"] not in ("NO_CODE", "ERROR", "TIMEOUT"))
        coverage = n_ran / len(items) if items else 0
        logger.info(f"    {subj}: {coverage:.0%} ({n_ran}/{len(items)})")

    # Compute echo chamber rate
    n_pass = sum(1 for r in results if r["execution_success"])
    n_echo = sum(1 for r in results if r["execution_success"] and not r["answer_correct"])
    ecr = n_echo / n_pass if n_pass > 0 else 0

    logger.info(f"\n  Echo chamber rate: {ecr:.0%} ({n_echo}/{n_pass})")
    logger.info(f"  G5 threshold: ≤20%, actual: {ecr:.0%}")

    # Coverage by level
    by_level = {}
    for r in results:
        lv = r["level"]
        by_level.setdefault(lv, []).append(r)

    logger.info("\n  Coverage by level:")
    for lv in sorted(by_level):
        items = by_level[lv]
        n_ran = sum(1 for r in items if r["execution_verdict"] not in ("NO_CODE", "ERROR", "TIMEOUT"))
        coverage = n_ran / len(items) if items else 0
        logger.info(f"    Level {lv}: {coverage:.0%} ({n_ran}/{len(items)})")

    output = {
        "experiment": "exp3_coverage",
        "echo_chamber_rate": ecr,
        "coverage_by_subject": {
            subj: sum(1 for it in items if it["verdict"] not in ("NO_CODE", "ERROR", "TIMEOUT")) / len(items)
            for subj, items in by_subject.items()
        },
    }
    with open(RESULTS_DIR / "exp3_coverage.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=== Experiment 3 DONE ===")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run ExeVer feasibility experiments")
    parser.add_argument("--exp", default="0", help="Which experiment to run (0-4 or 'all')")
    parser.add_argument("--backend", default="mock", choices=["mock", "vllm", "hf"])
    parser.add_argument("--model", default="", help="Model ID for non-mock backends")
    args = parser.parse_args()

    if args.exp == "0" or args.exp == "all":
        exp0_setup()
    if args.exp == "1" or args.exp == "all":
        exp1_baseline(args.backend, args.model)
    if args.exp == "2" or args.exp == "all":
        exp2_two_pass(args.backend, args.model)
    if args.exp == "3" or args.exp == "all":
        exp3_coverage(args.backend, args.model)


if __name__ == "__main__":
    main()
