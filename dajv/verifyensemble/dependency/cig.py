"""Cumulative Information Gain (CIG), adapted from Kuai et al. 2026.

Operational definition: the mutual information (in nats) between the
acceptance variables of two verifiers, restricted to the wrong-candidate
subset. Higher CIG means stronger directional dependency between the
two verifiers conditional on the input being wrong.

We use the joint entropy decomposition

    CIG(i, j | wrong) = H(A_i | wrong) + H(A_j | wrong) - H(A_i, A_j | wrong)

with the convention 0 log 0 = 0.
"""
from __future__ import annotations

import math
from typing import Sequence


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def cig(
    accept_i: Sequence[bool],
    accept_j: Sequence[bool],
    problem_correct: Sequence[bool],
) -> float:
    """Cumulative Information Gain restricted to wrong candidates."""
    wrong = [k for k, c in enumerate(problem_correct) if not c]
    n = len(wrong)
    if n == 0:
        return 0.0

    p_i = sum(1 for k in wrong if accept_i[k]) / n
    p_j = sum(1 for k in wrong if accept_j[k]) / n

    # Joint distribution over (A_i, A_j)
    counts = {(False, False): 0, (False, True): 0, (True, False): 0, (True, True): 0}
    for k in wrong:
        counts[(bool(accept_i[k]), bool(accept_j[k]))] += 1
    joint = {k: v / n for k, v in counts.items()}

    H_ij = 0.0
    for p in joint.values():
        if p > 0:
            H_ij -= p * math.log(p)
    H_i = _binary_entropy(p_i)
    H_j = _binary_entropy(p_j)
    return H_i + H_j - H_ij
