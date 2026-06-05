"""Cohen's kappa for inter-verifier agreement on binary acceptance."""
from __future__ import annotations

from typing import Sequence


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa coefficient for two binary raters.

    Args:
        a, b: equal-length binary sequences (verifier-i acceptance vs
            verifier-j acceptance over the same problem set).

    Returns:
        kappa in [-1, 1]. Returns 0.0 when both raters are constant.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be same length")
    n = len(a)
    if n == 0:
        return 0.0

    a_pos = sum(1 for x in a if x)
    b_pos = sum(1 for x in b if x)
    agree = sum(1 for x, y in zip(a, b) if bool(x) == bool(y))

    po = agree / n
    pa_pos = a_pos / n
    pb_pos = b_pos / n
    pe = pa_pos * pb_pos + (1 - pa_pos) * (1 - pb_pos)

    if pe == 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)
