"""Expected Calibration Error (ECE) and reliability diagrams.

ECE = sum over bins of (|bin| / N) * |acc(bin) - conf(bin)|

Default: 15 equal-frequency bins (matches common LLM-calibration
literature). Equal-width bins also supported.
"""
from __future__ import annotations

from typing import Sequence


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 15,
    strategy: str = "equal_frequency",
) -> float:
    """ECE on the items where a confidence prediction was made."""
    if len(confidences) != len(correct):
        raise ValueError("length mismatch")
    n = len(confidences)
    if n == 0:
        return 0.0

    pairs = sorted(zip(confidences, correct), key=lambda t: t[0])
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    if strategy == "equal_frequency":
        for idx, (c, y) in enumerate(pairs):
            bin_idx = min(n_bins - 1, int(idx * n_bins / n))
            bins[bin_idx].append((c, y))
    elif strategy == "equal_width":
        for c, y in pairs:
            bin_idx = min(n_bins - 1, int(c * n_bins))
            bins[bin_idx].append((c, y))
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    total = 0.0
    for bin_items in bins:
        if not bin_items:
            continue
        bin_conf = sum(c for c, _ in bin_items) / len(bin_items)
        bin_acc = sum(1 for _, y in bin_items if y) / len(bin_items)
        total += (len(bin_items) / n) * abs(bin_acc - bin_conf)
    return total


def reliability_diagram(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 15,
    strategy: str = "equal_frequency",
) -> dict:
    """Per-bin (mean_confidence, accuracy, count) for plotting."""
    if len(confidences) != len(correct):
        raise ValueError("length mismatch")
    n = len(confidences)
    pairs = sorted(zip(confidences, correct), key=lambda t: t[0])
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    if strategy == "equal_frequency":
        for idx, (c, y) in enumerate(pairs):
            bin_idx = min(n_bins - 1, int(idx * n_bins / n))
            bins[bin_idx].append((c, y))
    else:
        for c, y in pairs:
            bin_idx = min(n_bins - 1, int(c * n_bins))
            bins[bin_idx].append((c, y))

    mean_conf: list[float | None] = []
    accuracy: list[float | None] = []
    counts: list[int] = []
    for bin_items in bins:
        if not bin_items:
            mean_conf.append(None)
            accuracy.append(None)
            counts.append(0)
            continue
        mean_conf.append(sum(c for c, _ in bin_items) / len(bin_items))
        accuracy.append(sum(1 for _, y in bin_items if y) / len(bin_items))
        counts.append(len(bin_items))
    return {"mean_confidence": mean_conf, "accuracy": accuracy, "count": counts}
