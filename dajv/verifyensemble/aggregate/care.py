"""CARE-style aggregation baseline (Zhao et al. 2026, paraphrased).

CARE assumes each verifier's accept signal is the sum of a latent
true-quality signal and a shared confounder. We use a simplified
implementation suitable for binary acceptance:
  1. Fit per-verifier reliability w_i on calibration data via
     logistic regression of correctness onto each verifier's vote.
  2. At deployment, compute weighted vote score s = sum(w_i * vote_i)
     and threshold against a calibrated cutoff.

This is intentionally lighter than the full CARE method but matches its
core insight: down-weight verifiers whose votes are correlated with a
known confounder (here, base-model architecture family).
"""
from __future__ import annotations

from typing import Sequence

from verifyensemble.aggregate.posterior import clopper_pearson


def fit_care_weights(
    accept_calibration: Sequence[Sequence[bool]],
    problem_correct_calibration: Sequence[bool],
    ridge: float = 1e-3,
) -> list[float]:
    """Fit per-verifier weights via ridge-regularized logistic.

    Closed-form via Newton-Raphson on a binary logistic model:
        P(correct) = sigmoid(sum_i w_i * accept_i)

    Args:
        accept_calibration: shape (k, n)
        problem_correct_calibration: length n
        ridge: L2 regularization on weights

    Returns:
        weights w of length k
    """
    import math
    k = len(accept_calibration)
    n = len(problem_correct_calibration)
    w = [0.0] * k
    X = [[1.0 if accept_calibration[i][j] else 0.0 for i in range(k)]
         for j in range(n)]
    y = [1.0 if c else 0.0 for c in problem_correct_calibration]

    def _sigmoid(zi: float) -> float:
        # numerically stable
        if zi >= 0:
            ez = math.exp(-zi)
            return 1.0 / (1.0 + ez)
        ez = math.exp(zi)
        return ez / (1.0 + ez)

    for _ in range(50):
        # logits
        z = [sum(w[i] * X[j][i] for i in range(k)) for j in range(n)]
        p = [_sigmoid(zi) for zi in z]
        # gradient
        g = [0.0] * k
        for j in range(n):
            r = (p[j] - y[j])
            for i in range(k):
                g[i] += X[j][i] * r
        for i in range(k):
            g[i] += ridge * w[i]
        # Hessian diagonal (approx)
        h = [ridge] * k
        for j in range(n):
            pj = p[j] * (1 - p[j])
            for i in range(k):
                h[i] += X[j][i] * X[j][i] * pj
        # Newton step
        step = [g[i] / max(h[i], 1e-6) for i in range(k)]
        max_step = max(abs(s) for s in step) if step else 0.0
        for i in range(k):
            w[i] -= step[i]
        if max_step < 1e-6:
            break
    return w


def care_aggregate(
    votes: Sequence,
    weights: Sequence[float],
    threshold: float = 0.5,
) -> dict:
    """CARE-style weighted aggregation. Calibrated against a calibration set.

    Args:
        votes: length k; values in {True, False, None}
        weights: length k; fitted via fit_care_weights
        threshold: P(correct) above which to COMMIT
    """
    import math
    working = [(w, v) for w, v in zip(weights, votes) if v is not None]
    if not working:
        return {"P_correct": None, "lower": None, "upper": None,
                "recommendation": "ABSTAIN_NO_VERIFIERS"}
    z = sum(w * (1.0 if v is True else 0.0) for w, v in working)
    if z >= 0:
        P = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        P = ez / (1.0 + ez)
    n_working = len(working)
    n_accept = sum(1 for _, v in working if v is True)
    lo, hi = clopper_pearson(n_accept, n_working)
    rec = "COMMIT" if P >= threshold else "ABSTAIN"
    return {"P_correct": P, "lower": lo, "upper": hi,
            "recommendation": rec,
            "n_working": n_working, "n_accept": n_accept}
