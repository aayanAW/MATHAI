"""Numerical implementation of Theorem 2 (calibration sample complexity).

Theorem 2 (DAJV sample complexity).
To estimate every off-diagonal entry of the dependency matrix
D (k x k) to within entrywise L-infinity error eps with probability
at least 1 - delta, it suffices that the calibration set size satisfy

    n  >=  (1 / (2 eps^2)) * log(  k * (k - 1) / delta  ).

Proof: Hoeffding's inequality on each pairwise correlation estimator
plus a union bound over k(k-1)/2 distinct pairs.
"""
from __future__ import annotations

import math


def required_n(k: int, eps: float, delta: float = 0.05) -> int:
    """Theorem 2 lower bound on required calibration sample size."""
    if k < 2:
        return 0
    if eps <= 0 or eps >= 1:
        raise ValueError("eps must be in (0, 1)")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    n_pairs = k * (k - 1) / 2
    return math.ceil(math.log(2 * n_pairs / delta) / (2 * eps ** 2))
