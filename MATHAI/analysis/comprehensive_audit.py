"""Comprehensive audit of ExeVer exp5 results.

Addresses all Tier 1 issues:
1. FPVR computation (replaces echo chamber)
2. Repair audit (54 cases: difficulty distribution, false negatives)
3. Structured-answer exclusion count
4. Assertion type categorization
5. Coverage controlled for model accuracy

Outputs: analysis/audit_results.json
"""
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_exp5() -> dict:
    """Load exp5 MATH-500 results."""
    with open(RESULTS_DIR / "exp5_math500_full.json") as f:
        return json.load(f)


def load_problems() -> list:
    """Load MATH-500 problem set."""
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        return json.load(f)


# =========================================================================
# 1. FPVR Analysis (replaces echo chamber)
# =========================================================================

def compute_fpvr(results: list) -> dict:
    """Compute False Positive Verification Rate.

    FPVR = P(answer wrong | ALL_PASS)
    This is a post-hoc measurement, not a test-time signal.
    """
    all_pass = [r for r in results if r.get("verdict") == "ALL_PASS"]
    if not all_pass:
        return {"fpvr": None, "n_all_pass": 0}

    n_wrong_given_all_pass = sum(1 for r in all_pass if not r["answer_correct"])
    fpvr = n_wrong_given_all_pass / len(all_pass)

    # FPVR by level
    by_level = {}
    for lv in [1, 2, 3, 4, 5]:
        lv_pass = [r for r in all_pass if r["level"] == lv]
        if lv_pass:
            lv_wrong = sum(1 for r in lv_pass if not r["answer_correct"])
            by_level[lv] = {
                "n": len(lv_pass),
                "n_wrong": lv_wrong,
                "fpvr": round(lv_wrong / len(lv_pass), 4),
            }

    # FPVR by subject
    by_subject = {}
    subjects = sorted(set(r["type"] for r in all_pass))
    for subj in subjects:
        s_pass = [r for r in all_pass if r["type"] == subj]
        if s_pass:
            s_wrong = sum(1 for r in s_pass if not r["answer_correct"])
            by_subject[subj] = {
                "n": len(s_pass),
                "n_wrong": s_wrong,
                "fpvr": round(s_wrong / len(s_pass), 4),
            }

    # Calibration: ALL_PASS accuracy vs fallback accuracy
    fallback = [r for r in results if r.get("verdict") != "ALL_PASS"]
    all_pass_acc = sum(1 for r in all_pass if r["answer_correct"]) / len(all_pass)
    fallback_acc = (
        sum(1 for r in fallback if r["answer_correct"]) / len(fallback)
        if fallback else 0
    )

    return {
        "fpvr": round(fpvr, 4),
        "n_all_pass": len(all_pass),
        "n_wrong_given_all_pass": n_wrong_given_all_pass,
        "fpvr_by_level": by_level,
        "fpvr_by_subject": by_subject,
        "calibration": {
            "all_pass_accuracy": round(all_pass_acc, 4),
            "fallback_accuracy": round(fallback_acc, 4),
            "gap_pp": round((all_pass_acc - fallback_acc) * 100, 1),
        },
    }


# =========================================================================
# 2. Repair Audit
# =========================================================================

def audit_repairs(results: list) -> dict:
    """Audit repaired cases for bias and false negatives."""
    repaired = [r for r in results if r.get("repaired", False)]
    all_results = results

    # Difficulty distribution of repaired cases
    repair_by_level = Counter(r["level"] for r in repaired)
    repair_by_subject = Counter(r["type"] for r in repaired)

    # False negatives: wrong answers that passed verification (ALL_PASS)
    all_pass_wrong = [
        r for r in all_results
        if r.get("verdict") == "ALL_PASS" and not r["answer_correct"]
    ]

    # False negatives: wrong answers that got no assertion failure
    # (fallback verdicts: SYNTAX_ERROR, RUNTIME_ERROR, TIMEOUT, NO_SCRIPT)
    fallback_wrong = [
        r for r in all_results
        if r.get("verdict") not in ("ALL_PASS", "REPAIRED", "REPAIRED_UNVERIFIED")
        and "FAIL_STEP" not in r.get("verdict", "")
        and not r["answer_correct"]
    ]

    # Assertion failures that triggered repair
    assertion_failures = [
        r for r in all_results
        if "FAIL_STEP" in r.get("verdict", "")
    ]

    return {
        "n_repaired": len(repaired),
        "repair_accuracy": (
            sum(1 for r in repaired if r["answer_correct"]) / len(repaired)
            if repaired else 0
        ),
        "repair_by_level": dict(sorted(repair_by_level.items())),
        "repair_by_subject": dict(sorted(repair_by_subject.items())),
        "false_positives_all_pass": {
            "count": len(all_pass_wrong),
            "description": "Wrong answers that passed ALL assertions (FPVR cases)",
            "by_level": dict(Counter(r["level"] for r in all_pass_wrong)),
            "by_subject": dict(Counter(r["type"] for r in all_pass_wrong)),
        },
        "false_negatives_no_verdict": {
            "count": len(fallback_wrong),
            "description": "Wrong answers where verification crashed/failed (no useful signal)",
            "by_level": dict(Counter(r["level"] for r in fallback_wrong)),
        },
        "assertion_failures": {
            "count": len(assertion_failures),
            "description": "Cases where assertion failed (triggered repair attempt)",
        },
        "trigger_rate": round(
            len(assertion_failures) / len(all_results), 4
        ) if all_results else 0,
        "interpretation": (
            "The verifier catches a narrow class of shallow computational errors "
            f"(trigger rate: {len(assertion_failures)}/{len(all_results)} = "
            f"{len(assertion_failures)/len(all_results)*100:.1f}%). "
            f"Repair is effective on this class ({len(repaired)} repaired, "
            f"{sum(1 for r in repaired if r['answer_correct'])}/{len(repaired)} correct). "
            f"The verifier does NOT catch conceptual errors — these pass as "
            f"false positives (FPVR = {len(all_pass_wrong)}/{len([r for r in all_results if r.get('verdict')=='ALL_PASS'])} "
            f"= {len(all_pass_wrong)/max(len([r for r in all_results if r.get('verdict')=='ALL_PASS']),1)*100:.1f}%)."
        ),
    }


# =========================================================================
# 3. Structured-Answer Exclusion Count
# =========================================================================

STRUCTURED_PATTERNS = [
    r"\{.*,.*\}",           # Sets: {1, 2, 3}
    r"\(.*,.*\)",           # Tuples/intervals: (1, 3) or (-1, 3]
    r"\\begin\{[pm]matrix", # Matrices
    r"\s+or\s+",            # Disjunctive: "2 or -2"
    r"\\text\{",            # Text answers: \text{yes}
    r"\\begin\{array\}",    # Array environments
]


def count_structured_answers(problems: list) -> dict:
    """Count how many MATH-500 gold answers have structured formats."""
    structured = []
    scalar = []

    for p in problems:
        answer = p.get("answer", "")
        is_structured = False
        matched_pattern = None

        for pattern in STRUCTURED_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                is_structured = True
                matched_pattern = pattern
                break

        if is_structured:
            structured.append({
                "id": p["id"],
                "answer": answer,
                "level": p["level"],
                "type": p["type"],
                "matched_pattern": matched_pattern,
            })
        else:
            scalar.append(p)

    by_subject = Counter(s["type"] for s in structured)
    by_level = Counter(s["level"] for s in structured)

    return {
        "total_problems": len(problems),
        "n_structured": len(structured),
        "n_scalar": len(scalar),
        "pct_structured": round(len(structured) / len(problems) * 100, 1),
        "structured_by_subject": dict(sorted(by_subject.items())),
        "structured_by_level": dict(sorted(by_level.items())),
        "examples": structured[:10],  # First 10 examples
        "scope_statement": (
            f"ExeVer measures SymPy consistency-checkability for scalar algebraic "
            f"and numerical outputs. {len(structured)} of {len(problems)} problems "
            f"({len(structured)/len(problems)*100:.1f}%) have structured answers "
            f"(sets, intervals, tuples, matrices, text) that fall outside our "
            f"comparator's scope."
        ),
    }


# =========================================================================
# 4. Assertion Type Categorization
# =========================================================================

def categorize_assertion(assertion: str) -> str:
    """Categorize a single assertion by type.

    Categories:
    - algebraic_equivalence: assert expand(A) == expand(B)
    - substitution_check: assert expr.subs(x, val) == expected
    - numerical_evaluation: assert abs(...) < tol, or float comparisons
    - bound_type_check: assert x > 0, isinstance, type checks
    - trivial: assert True, identity, len > 0
    - other: anything else
    """
    body = assertion.replace("assert ", "", 1).strip()

    # Trivial
    if body.startswith("True"):
        return "trivial"
    if "==" in body:
        parts = body.split("==", 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip().split(",")[0].strip()
        if lhs == rhs:
            return "trivial"
    if body.startswith("isinstance("):
        return "trivial"
    if "len(" in body and "> 0" in body:
        return "trivial"

    # Algebraic equivalence: expand(...) == expand(...), Eq(..., ...)
    if "expand(" in body and "==" in body:
        return "algebraic_equivalence"
    if body.startswith("Eq("):
        return "algebraic_equivalence"

    # Substitution: .subs(
    if ".subs(" in body:
        return "substitution_check"

    # Numerical: abs(...) < , float(...), round(...)
    if "abs(" in body and ("<" in body or ">" in body):
        return "numerical_evaluation"
    if "float(" in body or "round(" in body:
        return "numerical_evaluation"
    if "evalf(" in body:
        return "numerical_evaluation"

    # Bound/type checks: > 0, < 100, isinstance, is not None
    if re.search(r"[<>]=?\s*\d", body):
        return "bound_type_check"
    if "is not None" in body or "is None" in body:
        return "bound_type_check"
    if "isinstance(" in body:
        return "bound_type_check"

    # General equality checks (most common)
    if "==" in body:
        return "algebraic_equivalence"  # Default for == comparisons

    return "other"


def categorize_all_assertions(results: list) -> dict:
    """Categorize all assertions from exp5 verification scripts."""
    all_assertions = []
    for r in results:
        # Extract assertions from the result if available
        n_asserts = r.get("assertions", 0)
        if n_asserts > 0:
            all_assertions.append({
                "id": r["id"],
                "n_assertions": n_asserts,
                "verdict": r["verdict"],
            })

    # We don't have individual assertion text in the results JSON.
    # We can only report assertion COUNTS and quality from the stored data.
    # Full categorization requires re-parsing verification scripts.

    total_assertions = sum(r.get("assertions", 0) for r in results)
    total_with_assertions = sum(1 for r in results if r.get("assertions", 0) > 0)
    avg_assertions = total_assertions / max(total_with_assertions, 1)

    return {
        "total_assertions": total_assertions,
        "problems_with_assertions": total_with_assertions,
        "avg_assertions_per_problem": round(avg_assertions, 2),
        "note": (
            "Full assertion-type categorization requires re-parsing verification "
            "scripts from raw responses (not stored in exp5 results JSON). "
            "The categorize_assertion() function is implemented and ready for "
            "use when scripts are available."
        ),
    }


# =========================================================================
# 5. Coverage Controlled for Model Accuracy
# =========================================================================

def coverage_controlled_for_accuracy(results: list) -> dict:
    """Compute coverage after controlling for greedy accuracy per cell.

    Shows that coverage isn't JUST about difficulty — some cells
    have similar accuracy but different coverage.
    """
    cells = defaultdict(lambda: {"n": 0, "greedy_correct": 0, "has_verdict": 0})

    for r in results:
        key = (r["type"], r["level"])
        cells[key]["n"] += 1
        if r["answer_correct"]:
            cells[key]["greedy_correct"] += 1
        if r["verdict"] in ("ALL_PASS",) or "FAIL_STEP" in r.get("verdict", ""):
            cells[key]["has_verdict"] += 1

    cell_data = []
    for (subj, level), data in sorted(cells.items()):
        n = data["n"]
        if n < 5:  # Skip tiny cells
            continue
        greedy_acc = data["greedy_correct"] / n
        coverage = data["has_verdict"] / n
        cell_data.append({
            "subject": subj,
            "level": level,
            "n": n,
            "greedy_accuracy": round(greedy_acc, 3),
            "problem_coverage": round(coverage, 3),
            "residual": round(coverage - greedy_acc, 3),  # Coverage beyond accuracy
        })

    # Compute correlation between greedy accuracy and coverage
    if cell_data:
        accs = [c["greedy_accuracy"] for c in cell_data]
        covs = [c["problem_coverage"] for c in cell_data]
        n = len(accs)
        mean_a = sum(accs) / n
        mean_c = sum(covs) / n
        cov = sum((a - mean_a) * (c - mean_c) for a, c in zip(accs, covs)) / n
        std_a = (sum((a - mean_a) ** 2 for a in accs) / n) ** 0.5
        std_c = (sum((c - mean_c) ** 2 for c in covs) / n) ** 0.5
        correlation = cov / (std_a * std_c) if std_a * std_c > 0 else 0
    else:
        correlation = 0

    return {
        "cells": cell_data,
        "accuracy_coverage_correlation": round(correlation, 3),
        "interpretation": (
            f"Pearson correlation between greedy accuracy and problem coverage "
            f"across subject×level cells: r = {correlation:.3f}. "
            f"{'This confirms coverage is heavily confounded with difficulty.' if abs(correlation) > 0.5 else 'Coverage shows some independence from difficulty.'}"
        ),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("ExeVer Comprehensive Audit")
    print("=" * 70)

    # Load data
    exp5 = load_exp5()
    results = exp5["exever_results"]
    problems = load_problems()
    print(f"Loaded {len(results)} exp5 results, {len(problems)} problems")

    output = {}

    # 1. FPVR
    print("\n--- 1. FPVR Analysis ---")
    fpvr_data = compute_fpvr(results)
    output["fpvr"] = fpvr_data
    print(f"  FPVR: {fpvr_data['fpvr']:.4f} ({fpvr_data['n_wrong_given_all_pass']}/{fpvr_data['n_all_pass']})")
    print(f"  Calibration gap: +{fpvr_data['calibration']['gap_pp']}pp")
    level_strs = []
    for k, v in sorted(fpvr_data["fpvr_by_level"].items()):
        level_strs.append(f"L{k}={v['fpvr']:.3f}")
    print(f"  FPVR by level: {', '.join(level_strs)}")

    # 2. Repair Audit
    print("\n--- 2. Repair Audit ---")
    repair_data = audit_repairs(results)
    output["repair_audit"] = repair_data
    print(f"  Repaired: {repair_data['n_repaired']}")
    print(f"  Repair accuracy: {repair_data['repair_accuracy']:.3f}")
    print(f"  Trigger rate: {repair_data['trigger_rate']:.3f}")
    print(f"  By level: {repair_data['repair_by_level']}")
    print(f"  False positives (ALL_PASS wrong): {repair_data['false_positives_all_pass']['count']}")
    print(f"  False negatives (fallback wrong): {repair_data['false_negatives_no_verdict']['count']}")

    # 3. Structured Answers
    print("\n--- 3. Structured-Answer Exclusions ---")
    struct_data = count_structured_answers(problems)
    output["structured_answers"] = struct_data
    print(f"  Structured: {struct_data['n_structured']}/{struct_data['total_problems']} ({struct_data['pct_structured']}%)")
    print(f"  By subject: {struct_data['structured_by_subject']}")

    # 4. Assertion Categorization
    print("\n--- 4. Assertion Categorization ---")
    assert_data = categorize_all_assertions(results)
    output["assertion_categorization"] = assert_data
    print(f"  Total assertions: {assert_data['total_assertions']}")
    print(f"  Avg per problem: {assert_data['avg_assertions_per_problem']}")

    # 5. Coverage vs Accuracy
    print("\n--- 5. Coverage Controlled for Accuracy ---")
    cov_data = coverage_controlled_for_accuracy(results)
    output["coverage_vs_accuracy"] = cov_data
    print(f"  Correlation: r = {cov_data['accuracy_coverage_correlation']}")
    print(f"  {cov_data['interpretation']}")

    # Save
    out_path = RESULTS_DIR / "audit_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
