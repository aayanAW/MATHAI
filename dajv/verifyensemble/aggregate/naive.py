"""Naive consensus baselines: unanimous and majority.

Each aggregator returns
    (P_correct_estimate, lower, upper, recommendation)
where the interval is a Clopper-Pearson interval whose width depends on
the number of contributing working verifiers.
"""
from __future__ import annotations

from typing import Sequence

from verifyensemble.aggregate.posterior import clopper_pearson


def _coerce(votes: Sequence) -> list[str]:
    """Coerce a vote sequence to canonical labels: 'accept' / 'reject' / 'abstain'."""
    out = []
    for v in votes:
        if v is None or v == "abstain" or v == "broken":
            out.append("abstain")
        elif v is True or v == "accept" or v == 1 or v == "1":
            out.append("accept")
        else:
            out.append("reject")
    return out


def naive_unanimous(votes: Sequence) -> dict:
    """Strict unanimous consensus: commit iff every working verifier accepts."""
    labels = _coerce(votes)
    working = [l for l in labels if l != "abstain"]
    accepts = [l for l in working if l == "accept"]
    if not working:
        return {"P_correct": None, "lower": None, "upper": None,
                "recommendation": "ABSTAIN_NO_VERIFIERS",
                "n_working": 0, "n_accept": 0}
    if len(accepts) == len(working):
        # All working verifiers accepted; CI on success-rate naively at 1/1
        lo, hi = clopper_pearson(len(working), len(working))
        return {"P_correct": 1.0, "lower": lo, "upper": hi,
                "recommendation": "COMMIT",
                "n_working": len(working), "n_accept": len(accepts)}
    lo, hi = clopper_pearson(len(accepts), len(working))
    return {"P_correct": len(accepts) / len(working), "lower": lo, "upper": hi,
            "recommendation": "ABSTAIN",
            "n_working": len(working), "n_accept": len(accepts)}


def naive_majority(votes: Sequence) -> dict:
    """Majority consensus: commit iff > k/2 working verifiers accept."""
    labels = _coerce(votes)
    working = [l for l in labels if l != "abstain"]
    accepts = [l for l in working if l == "accept"]
    if not working:
        return {"P_correct": None, "lower": None, "upper": None,
                "recommendation": "ABSTAIN_NO_VERIFIERS",
                "n_working": 0, "n_accept": 0}
    lo, hi = clopper_pearson(len(accepts), len(working))
    rec = "COMMIT" if len(accepts) > len(working) / 2 else "ABSTAIN"
    return {"P_correct": len(accepts) / len(working), "lower": lo, "upper": hi,
            "recommendation": rec,
            "n_working": len(working), "n_accept": len(accepts)}
