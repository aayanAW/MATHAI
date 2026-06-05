"""Metrics for ExeVer evaluation.

Standardized metric definitions:
- Problem Coverage (PC): P(verdict in {ALL_PASS, FAIL_STEP}) — PRIMARY
- False Positive Verification Rate (FPVR): P(answer wrong | ALL_PASS) — PRIMARY
- Script Validity Rate (SVR): P(script compiles)
- Assertion Quality (AQ): P(assertion non-trivial)
- Repair Rate (RR): P(correct | repair triggered)

Legacy metrics (retained for backward compat):
- pass@k (unbiased estimator, Chen et al. 2021)
- majority@k
- Verification Precision/Recall (VP/VR) — requires gold step labels
- Reasoning Validity Rate (RVR)
"""
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: k for pass@k.

    Returns:
        Estimated pass@k probability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def majority_at_k(answers: List[str], gold: str, check_fn) -> bool:
    """Majority voting accuracy from k samples.

    Args:
        answers: List of predicted answers (strings).
        gold: Gold answer string.
        check_fn: Function(pred, gold) -> (bool, str) for answer comparison.

    Returns:
        Whether the majority-voted answer is correct.
    """
    if not answers:
        return False
    # Count answer occurrences (use normalized form)
    counts: Counter = Counter(answers)
    majority_answer = counts.most_common(1)[0][0]
    is_correct, _ = check_fn(majority_answer, gold)
    return is_correct


def verification_coverage(verdicts: List[str]) -> float:
    """Fraction of steps that are checkable (PASS or FAIL, not ERROR).

    VC = |{steps with PASS or FAIL}| / |all steps|
    """
    if not verdicts:
        return 0.0
    checkable = sum(1 for v in verdicts if v in ("PASS", "FAIL"))
    return checkable / len(verdicts)


def verification_precision(
    verdicts: List[str],
    gold_labels: List[bool],
) -> Optional[float]:
    """Among FAIL verdicts, fraction that are truly incorrect.

    VP = TP_fail / (TP_fail + FP_fail)

    Args:
        verdicts: List of ExeVer verdicts per step.
        gold_labels: List of ground-truth correctness per step (True=correct).

    Returns:
        Precision, or None if no FAIL verdicts.
    """
    tp = 0  # Correctly identified as FAIL (truly incorrect)
    fp = 0  # Incorrectly identified as FAIL (actually correct)
    for v, g in zip(verdicts, gold_labels):
        if v == "FAIL":
            if not g:  # Truly incorrect
                tp += 1
            else:  # Actually correct
                fp += 1
    if tp + fp == 0:
        return None
    return tp / (tp + fp)


def verification_recall(
    verdicts: List[str],
    gold_labels: List[bool],
) -> Optional[float]:
    """Among truly incorrect steps, fraction caught as FAIL.

    VR = TP_fail / (TP_fail + FN_fail)
    """
    tp = 0  # Correctly identified as FAIL
    fn = 0  # Missed (incorrect step but not caught)
    for v, g in zip(verdicts, gold_labels):
        if not g:  # Truly incorrect step
            if v == "FAIL":
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return None
    return tp / (tp + fn)


def reasoning_validity_rate(
    answer_correct: bool,
    verdicts: List[str],
) -> Optional[bool]:
    """Whether a correct-answer solution has all checkable steps passing.

    RVR is computed as the fraction of correct-answer solutions where
    all checkable steps are PASS (no FAIL).

    Returns None if answer is wrong (not applicable).
    Returns True if answer correct AND no FAIL verdicts.
    Returns False if answer correct BUT some FAIL verdicts.
    """
    if not answer_correct:
        return None
    return all(v != "FAIL" for v in verdicts)


def false_positive_verification_rate(
    assertions_pass: bool,
    answer_correct: bool,
) -> Optional[bool]:
    """Whether this is a false positive verification case.

    FPVR = P(answer wrong | ALL_PASS)

    This is a post-hoc measurement requiring gold answers. It is NOT
    a test-time signal. It measures how often verification "passes"
    on incorrect solutions — the fundamental ceiling on same-model
    consistency checking.

    Returns True if false positive (assertions pass but answer wrong).
    Returns False if true positive (assertions pass and answer correct).
    Returns None if assertions didn't all pass.
    """
    if not assertions_pass:
        return None
    return not answer_correct


# Backward compatibility alias
echo_chamber_rate = false_positive_verification_rate


def assertion_quality(assertions: List[str]) -> Dict[str, int]:
    """Classify assertions as trivial or non-trivial.

    Trivial assertions:
    - assert True
    - assert x == x (identity)
    - assert isinstance(...) (type check)
    - assert len(...) > 0 (existence only)

    Returns dict with counts: {'trivial': n, 'nontrivial': m, 'total': n+m}
    """
    trivial = 0
    nontrivial = 0
    for a in assertions:
        a_stripped = a.strip()
        if _is_trivial_assertion(a_stripped):
            trivial += 1
        else:
            nontrivial += 1
    return {
        "trivial": trivial,
        "nontrivial": nontrivial,
        "total": trivial + nontrivial,
        "quality_rate": nontrivial / (trivial + nontrivial) if (trivial + nontrivial) > 0 else 0.0,
    }


def _is_trivial_assertion(assertion: str) -> bool:
    """Check if an assertion is trivial (doesn't test math)."""
    # Remove the assert keyword
    body = assertion.replace("assert ", "", 1).strip()

    # Direct True
    if body.startswith("True"):
        return True

    # Identity check: x == x
    if "==" in body:
        parts = body.split("==", 1)
        if parts[0].strip() == parts[1].strip().split(",")[0].strip():
            return True

    # Type check
    if body.startswith("isinstance("):
        return True

    # Length existence check
    if "len(" in body and "> 0" in body:
        return True

    return False


def compute_pass_at_k_for_problems(
    results: List[Dict],
    k_values: List[int] = [1, 4, 8, 16],
) -> Dict[int, float]:
    """Compute pass@k across a set of problems.

    Args:
        results: List of dicts with 'n_samples' and 'n_correct' per problem.
        k_values: Which k values to compute.

    Returns:
        Dict mapping k -> average pass@k.
    """
    output = {}
    for k in k_values:
        scores = []
        for r in results:
            n = r["n_samples"]
            c = r["n_correct"]
            if n >= k:
                scores.append(pass_at_k(n, c, k))
        output[k] = np.mean(scores) if scores else 0.0
    return output


def aggregate_coverage_by_subject(
    results: List[Dict],
) -> Dict[str, float]:
    """Aggregate verification coverage by MATH subject.

    Args:
        results: List of dicts with 'type' (subject) and 'coverage' per problem.

    Returns:
        Dict mapping subject -> average coverage.
    """
    by_subject: Dict[str, List[float]] = {}
    for r in results:
        subj = r.get("type", "unknown")
        cov = r.get("coverage", 0.0)
        by_subject.setdefault(subj, []).append(cov)

    return {subj: np.mean(covs) for subj, covs in sorted(by_subject.items())}


def aggregate_coverage_by_level(
    results: List[Dict],
) -> Dict[int, float]:
    """Aggregate verification coverage by MATH difficulty level."""
    by_level: Dict[int, List[float]] = {}
    for r in results:
        level = r.get("level", 0)
        cov = r.get("coverage", 0.0)
        by_level.setdefault(level, []).append(cov)

    return {lv: np.mean(covs) for lv, covs in sorted(by_level.items())}
