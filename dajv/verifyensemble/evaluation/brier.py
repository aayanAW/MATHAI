"""Brier score for probabilistic binary predictions."""
from __future__ import annotations

from typing import Sequence


def brier_score(
    confidences: Sequence[float],
    correct: Sequence[bool],
) -> float:
    """Mean squared error between predicted probability and outcome."""
    if len(confidences) != len(correct):
        raise ValueError("length mismatch")
    if not confidences:
        return 0.0
    return sum((c - (1.0 if y else 0.0)) ** 2 for c, y in zip(confidences, correct)) / len(confidences)
