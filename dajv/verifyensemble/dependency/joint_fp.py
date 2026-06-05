"""Joint-FP estimator and independence bound.

A 'false positive' for verifier i on problem j means: candidate is WRONG
(problem_correct[j] is False) but verifier i ACCEPTS it.

The joint-FP rate of verifiers i, j is the proportion of wrong candidates
that BOTH accept. Under independence, this equals pi_i * pi_j where
pi_k = P(verifier_k accepts | candidate is wrong).
"""
from __future__ import annotations

from typing import Sequence


def joint_fp_rate(
    accept_i: Sequence[bool],
    accept_j: Sequence[bool],
    problem_correct: Sequence[bool],
) -> dict:
    """Compute empirical joint FP rate and independence prediction.

    Args:
        accept_i, accept_j: binary acceptance sequences (same length).
        problem_correct: True iff the candidate matches the gold answer.

    Returns:
        dict:
            n_wrong         number of wrong candidates
            pi_i            P(i accepts | wrong)
            pi_j            P(j accepts | wrong)
            indep_bound     pi_i * pi_j
            joint_observed  P(both accept | wrong)
            ratio           joint_observed / indep_bound (or +inf if bound==0)
            excess          joint_observed - indep_bound
    """
    if not (len(accept_i) == len(accept_j) == len(problem_correct)):
        raise ValueError("inputs must be same length")

    wrong = [k for k, c in enumerate(problem_correct) if not c]
    n_wrong = len(wrong)
    if n_wrong == 0:
        return {
            "n_wrong": 0,
            "pi_i": 0.0, "pi_j": 0.0,
            "indep_bound": 0.0, "joint_observed": 0.0,
            "ratio": 1.0, "excess": 0.0,
        }

    a_fp = sum(1 for k in wrong if accept_i[k]) / n_wrong
    b_fp = sum(1 for k in wrong if accept_j[k]) / n_wrong
    joint_fp = sum(1 for k in wrong if accept_i[k] and accept_j[k]) / n_wrong
    indep = a_fp * b_fp
    if indep == 0:
        ratio = float("inf") if joint_fp > 0 else 1.0
    else:
        ratio = joint_fp / indep
    return {
        "n_wrong": n_wrong,
        "pi_i": a_fp,
        "pi_j": b_fp,
        "indep_bound": indep,
        "joint_observed": joint_fp,
        "ratio": ratio,
        "excess": joint_fp - indep,
    }


def independence_bound_fp(marginals: Sequence[float]) -> float:
    """Naive product-of-marginals bound on joint FP across k verifiers."""
    p = 1.0
    for m in marginals:
        p *= m
    return p
