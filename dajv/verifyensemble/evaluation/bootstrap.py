"""Bootstrap confidence intervals for arbitrary metrics."""
from __future__ import annotations

import random
from typing import Callable, Sequence


def bootstrap_metric(
    metric_fn: Callable,
    *arrays: Sequence,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap a metric and return point estimate + CI.

    Args:
        metric_fn: callable taking the *arrays as inputs and returning a float.
        *arrays: one or more parallel sequences (resampled together).
        n_bootstrap: number of resamples.
        alpha: 1 - confidence level (default 0.05 -> 95% CI).
        seed: RNG seed.
    """
    rng = random.Random(seed)
    arrays = tuple(list(a) for a in arrays)
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("arrays must be same length")

    point = metric_fn(*arrays)
    boots = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        sampled = tuple([a[i] for i in idx] for a in arrays)
        boots.append(metric_fn(*sampled))
    boots.sort()
    lo = boots[max(0, int((alpha / 2) * len(boots)) - 1)]
    hi = boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)))]
    return {"point": point, "lower": lo, "upper": hi, "n_bootstrap": n_bootstrap}
