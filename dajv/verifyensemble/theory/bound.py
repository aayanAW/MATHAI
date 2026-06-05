"""Numerical implementation of Theorem 1 (joint-FP upper bound).

Theorem 1 (DAJV joint-FP bound).
Let V_1, ..., V_k be Bernoulli indicators with marginals
    pi_i = P(V_i = 1),
and pairwise correlations
    rho_ij = Corr(V_i, V_j) in [-1, 1].
Then

    P( and_i V_i = 1 ) <= B(pi, rho) :=
        prod_i pi_i  +  sum_{i<j}  max(rho_ij, 0) * sqrt(pi_i(1-pi_i) pi_j(1-pi_j)).

Recovers the independence bound prod_i pi_i when all rho_ij <= 0, and
is bounded above by the union bound 1 - prod_i (1 - pi_i).

Proof sketch (see paper Appendix A.1).
"""
from __future__ import annotations

import math
from typing import Sequence


def independence_lower_bound(pi: Sequence[float]) -> float:
    """Product-of-marginals bound: trivial under independence."""
    p = 1.0
    for m in pi:
        p *= m
    return p


def union_upper_bound(pi: Sequence[float]) -> float:
    """Generic union upper bound on joint FP: 1 - prod(1 - pi_i)."""
    p = 1.0
    for m in pi:
        p *= (1.0 - m)
    return 1.0 - p


def dajv_upper_bound(
    pi: Sequence[float],
    rho: Sequence[Sequence[float]],
) -> float:
    """Theorem 1 upper bound on joint-FP using measured pairwise rho.

    Args:
        pi:  length k, marginal acceptance probabilities on wrong candidates
        rho: k x k matrix of pairwise correlations in [-1, 1]; only upper
             triangle is read (rho is assumed symmetric).
    """
    k = len(pi)
    base = independence_lower_bound(pi)
    correction = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            r = rho[i][j]
            if r > 0:
                correction += r * math.sqrt(
                    max(pi[i] * (1 - pi[i]), 0.0)
                    * max(pi[j] * (1 - pi[j]), 0.0)
                )
    # Clip to the union-bound ceiling so the bound is always tight enough
    return min(base + correction, union_upper_bound(pi))
