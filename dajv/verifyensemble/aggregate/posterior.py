"""Clopper-Pearson exact binomial confidence intervals."""
from __future__ import annotations

import math


def _beta_cdf_inverse(alpha: float, a: float, b: float) -> float:
    """Inverse regularized incomplete beta (Beta-distribution CDF inverse).

    Pure-Python bisection. Sufficient for the small alpha grids used in
    Clopper-Pearson. Returns x in (0, 1) with I_x(a, b) = alpha.
    """
    # Bisection over [0, 1]
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _beta_cdf(mid, a, b) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) via continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use the standard relation I_x(a, b) = 1 - I_{1-x}(b, a) for stability
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(x, a, b) / a
    return 1.0 - bt * _betacf(1 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Lentz's continued-fraction for the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def clopper_pearson(k_success: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Clopper-Pearson confidence interval for a binomial proportion.

    Args:
        k_success: number of successes.
        n: number of trials.
        alpha: 1 - confidence level (default 0.05 -> 95% CI).

    Returns:
        (lower, upper) on the proportion.
    """
    if n == 0:
        return 0.0, 1.0
    if k_success == 0:
        lower = 0.0
    else:
        lower = _beta_cdf_inverse(alpha / 2, k_success, n - k_success + 1)
    if k_success == n:
        upper = 1.0
    else:
        upper = _beta_cdf_inverse(1 - alpha / 2, k_success + 1, n - k_success)
    return lower, upper
