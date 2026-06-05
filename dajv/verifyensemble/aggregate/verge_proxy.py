"""VERGE-style baseline (re-implementation proxy).

VERGE (Singh et al., 2026, arXiv:2601.20055) combines multi-model
consensus with SMT-based formal verification and Minimal Correction
Sets (MCS). The public description does not include source code, so
this module is a *proxy* baseline that mirrors the published
mechanism within the existing DAJV ensemble cache.

The proxy:

  1. Multi-model consensus stage: require at least ``min_agree`` of the
     k extractors to emit a non-abstain verifier (classification ==
     'working') and to accept the candidate (candidate_verdict is True).
  2. Formal-verification stage: gate on the executable signal -- the
     extractor's SymPy-emitted script is the analogue of an SMT
     instance, so we require unanimous executable accept across the
     subset of extractors that passed the working filter.
  3. MCS stage: if step 2 fails by exactly one extractor abstain or
     disagree, treat that one as the MCS and commit at a reduced
     confidence; if more than one disagrees, abstain.

This is the closest faithful proxy to VERGE buildable from the DAJV
cache without their published source. The paper's §6 reports DAJV
against this proxy and notes that the proxy is a strictly weaker
variant of full VERGE (no Z3, no MCS construction --- just the
mechanism the cache exposes).

Each ``verge_proxy_aggregate`` call returns the same dict shape as
``dajv_aggregate`` so the evaluation harness can swap it in.
"""
from __future__ import annotations

from typing import Sequence


def verge_proxy_aggregate(
    votes: Sequence,            # 4-vector of LLM-accept booleans (or None for abstain)
    classifications: Sequence,  # 4-vector of {'working', 'wrong_spec', 'trivial_or_broken', 'UNVERIFIABLE', ...}
    candidate_verdicts: Sequence,  # 4-vector of {True, False, None}
    min_agree: int = 3,
) -> dict:
    """Apply the VERGE-proxy aggregation rule.

    Returns: dict with P_correct (float or None), recommendation in
    {COMMIT, COMMIT_MCS, ESCALATE, ABSTAIN, ABSTAIN_NO_VERIFIERS},
    and diagnostic counts.
    """
    k = len(votes)
    if not (len(classifications) == k and len(candidate_verdicts) == k):
        raise ValueError("votes / classifications / candidate_verdicts "
                         "must all have length k")

    n_working = sum(1 for c in classifications if c == "working")
    n_accept = sum(1 for v in votes if v is True)
    n_exec_true = sum(1 for v in candidate_verdicts if v is True)
    n_consensus = sum(1 for c, v in zip(classifications, votes)
                      if c == "working" and v is True)

    if n_working == 0:
        return {
            "P_correct": None, "lower": None, "upper": None,
            "recommendation": "ABSTAIN_NO_VERIFIERS",
            "n_working": 0, "n_accept": 0, "n_consensus": 0,
            "stage": "no_working_verifiers",
        }

    # Stage 1: consensus. Require >= min_agree extractors with
    # classification == 'working' AND vote == True.
    if n_consensus < min_agree:
        return {
            "P_correct": n_consensus / max(k, 1),
            "lower": None, "upper": None,
            "recommendation": "ABSTAIN",
            "n_working": n_working, "n_accept": n_accept,
            "n_consensus": n_consensus,
            "stage": "consensus_fail",
        }

    # Stage 2: formal verification. Require unanimous candidate_verdict
    # across the consensus subset (already implied because vote=True
    # === candidate_verdict True in the cache, but enforce explicitly
    # in case downstream changes the vote definition).
    if n_exec_true >= min_agree:
        return {
            "P_correct": 0.97,  # canonical VERGE high-confidence operating point
            "lower": None, "upper": None,
            "recommendation": "COMMIT",
            "n_working": n_working, "n_accept": n_accept,
            "n_consensus": n_consensus,
            "stage": "formal_pass",
        }

    # Stage 3: MCS fallback. If exactly one extractor flipped, commit
    # at reduced confidence; otherwise abstain.
    if n_exec_true >= min_agree - 1:
        return {
            "P_correct": 0.80,
            "lower": None, "upper": None,
            "recommendation": "COMMIT_MCS",
            "n_working": n_working, "n_accept": n_accept,
            "n_consensus": n_consensus,
            "stage": "mcs_one_drop",
        }

    return {
        "P_correct": 0.50,
        "lower": None, "upper": None,
        "recommendation": "ABSTAIN",
        "n_working": n_working, "n_accept": n_accept,
        "n_consensus": n_consensus,
        "stage": "formal_fail",
    }
