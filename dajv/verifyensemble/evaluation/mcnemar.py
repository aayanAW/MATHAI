"""McNemar mid-p exact test for matched binary outcomes.

Reference: Fagerland, Lydersen, Laake (2013). The McNemar test for
binary matched-pairs data: mid-p and asymptotic are better than exact
conditional. BMC Med Res Methodol, 13:91.
"""
from __future__ import annotations

import math
from typing import Sequence


def _binomial_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    log_pmf = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
               + k * math.log(p) + (n - k) * math.log(1 - p))
    return math.exp(log_pmf)


def mcnemar_mid_p(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
) -> dict:
    """Mid-p exact McNemar test.

    Counts discordant pairs: b01 = (A wrong, B correct), b10 = (A correct, B wrong).
    Test H0: b01 = b10. Mid-p halves the probability of the observed count.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("length mismatch")
    b01 = sum(1 for a, b in zip(correct_a, correct_b) if (not a) and b)
    b10 = sum(1 for a, b in zip(correct_a, correct_b) if a and (not b))
    nd = b01 + b10
    if nd == 0:
        return {"b01": 0, "b10": 0, "n_discordant": 0, "mid_p": 1.0}
    k = min(b01, b10)
    p_obs = _binomial_pmf(k, nd, 0.5)
    p_extreme = sum(_binomial_pmf(i, nd, 0.5) for i in range(k))
    p_two_sided = 2 * (p_extreme + 0.5 * p_obs)
    p_two_sided = min(p_two_sided, 1.0)
    return {"b01": b01, "b10": b10, "n_discordant": nd, "mid_p": p_two_sided}
