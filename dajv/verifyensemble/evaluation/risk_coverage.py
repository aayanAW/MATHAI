"""Risk-coverage curves and AUC.

Risk = 1 - precision among committed items.
Coverage = fraction of items committed.

Curve is parameterized by the confidence-score threshold: at threshold
``tau``, commit items with score >= tau.
"""
from __future__ import annotations

from typing import Sequence


def risk_coverage_curve(
    scores: Sequence[float],
    correct: Sequence[bool],
) -> dict:
    """Compute risk and coverage at every distinct threshold.

    Args:
        scores: confidence score per item (higher = more confident).
        correct: True iff the item's answer matches gold.

    Returns:
        dict with sorted thresholds, coverage, risk, precision arrays.
    """
    if len(scores) != len(correct):
        raise ValueError("scores and correct must match in length")
    n = len(scores)
    if n == 0:
        return {"threshold": [], "coverage": [], "risk": [], "precision": []}

    pairs = sorted(zip(scores, correct), key=lambda t: -t[0])
    thresholds = []
    coverages = []
    risks = []
    precisions = []
    n_committed = 0
    n_correct = 0
    last_score = None
    for s, c in pairs:
        n_committed += 1
        if c:
            n_correct += 1
        if s != last_score:
            thresholds.append(s)
            cov = n_committed / n
            prec = n_correct / n_committed if n_committed > 0 else 0.0
            coverages.append(cov)
            precisions.append(prec)
            risks.append(1 - prec)
            last_score = s
    return {
        "threshold": thresholds,
        "coverage": coverages,
        "risk": risks,
        "precision": precisions,
    }


def risk_coverage_auc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    curve = risk_coverage_curve(scores, correct)
    if not curve["coverage"]:
        return 0.0
    cov = [0.0] + curve["coverage"]
    risk = [0.0] + curve["risk"]
    auc = 0.0
    for i in range(1, len(cov)):
        auc += 0.5 * (cov[i] - cov[i - 1]) * (risk[i] + risk[i - 1])
    return auc
