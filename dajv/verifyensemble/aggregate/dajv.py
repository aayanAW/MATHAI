"""DAJV aggregation: dependency-aware copula over binary verifier votes.

The likelihood model:

    P(v | correct) = prod_i (1 - pi_i^pos)^{1-v_i} (pi_i^pos)^{v_i}
                     * exp( sum_{i<j} rho_ij * phi(v_i, v_j) )

where
    pi_i^pos = P(verifier_i accepts | candidate is correct)
    pi_i^neg = P(verifier_i accepts | candidate is wrong)
    rho_ij   = empirical second-order interaction term (the dependency)
    phi(v_i, v_j) = (v_i - mu_i)(v_j - mu_j) with mu_i = pi_i^cond

The posterior P(correct | v) follows by Bayes with a prior P(correct).
Per-verifier marginals and the dependency tensor are estimated on a
calibration set.

This module exposes:
    DajvCalibration: fit on calibration data
    dajv_aggregate(votes, calibration): apply at deployment
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from verifyensemble.aggregate.posterior import clopper_pearson


@dataclass
class DajvCalibration:
    extractor_ids: list[str]
    pi_pos: list[float]                # k: P(accept | correct)
    pi_neg: list[float]                # k: P(accept | wrong)
    rho_pos: list[list[float]]         # k x k: interaction on correct subset
    rho_neg: list[list[float]]         # k x k: interaction on wrong subset
    prior_correct: float               # P(correct) on the calibration set

    @classmethod
    def fit(
        cls,
        accept_cal: Sequence[Sequence[bool]],
        problem_correct_cal: Sequence[bool],
        extractor_ids: list[str],
    ) -> "DajvCalibration":
        """Estimate marginals + 2-way interactions on calibration data."""
        k = len(accept_cal)
        n = len(problem_correct_cal)
        if k != len(extractor_ids):
            raise ValueError(
                f"k mismatch: accept_cal has {k} rows, "
                f"extractor_ids has {len(extractor_ids)}"
            )
        for i in range(k):
            if len(accept_cal[i]) != n:
                raise ValueError(
                    f"accept_cal[{i}] has length {len(accept_cal[i])}, "
                    f"expected {n} (== len(problem_correct_cal))"
                )

        correct_idx = [j for j, c in enumerate(problem_correct_cal) if c]
        wrong_idx = [j for j, c in enumerate(problem_correct_cal) if not c]

        prior = len(correct_idx) / max(n, 1)

        def _marg(idx: list[int]) -> list[float]:
            if not idx:
                return [0.5] * k
            return [sum(1 for j in idx if accept_cal[i][j]) / len(idx)
                    for i in range(k)]

        pi_pos = _marg(correct_idx)
        pi_neg = _marg(wrong_idx)

        def _interaction(idx: list[int], pi: list[float]) -> list[list[float]]:
            """Pearson-style centered interaction in {0, 1}^k restricted to idx."""
            R = [[0.0] * k for _ in range(k)]
            if not idx:
                return R
            for i in range(k):
                for j in range(k):
                    if i == j:
                        R[i][j] = 0.0
                        continue
                    mu_i, mu_j = pi[i], pi[j]
                    s = 0.0
                    for t in idx:
                        ai = 1.0 if accept_cal[i][t] else 0.0
                        aj = 1.0 if accept_cal[j][t] else 0.0
                        s += (ai - mu_i) * (aj - mu_j)
                    cov = s / len(idx)
                    var_i = max(mu_i * (1 - mu_i), 1e-6)
                    var_j = max(mu_j * (1 - mu_j), 1e-6)
                    # Standardize to [-1, 1]-ish range
                    R[i][j] = cov / math.sqrt(var_i * var_j)
            return R

        rho_pos = _interaction(correct_idx, pi_pos)
        rho_neg = _interaction(wrong_idx, pi_neg)

        return cls(
            extractor_ids=list(extractor_ids),
            pi_pos=pi_pos,
            pi_neg=pi_neg,
            rho_pos=rho_pos,
            rho_neg=rho_neg,
            prior_correct=prior,
        )


def _log_likelihood(
    votes: Sequence,
    pi: list[float],
    rho: list[list[float]],
) -> float:
    """Log P(votes | hypothesis) with second-order interaction term.

    votes may include None entries (abstain); those are marginalized
    out by skipping them entirely.
    """
    active_indices = [i for i, v in enumerate(votes) if v is not None]
    if not active_indices:
        return 0.0

    log_p = 0.0
    # Marginals
    for i in active_indices:
        v = 1 if votes[i] is True else 0
        m = pi[i]
        m = min(max(m, 1e-6), 1 - 1e-6)
        log_p += v * math.log(m) + (1 - v) * math.log(1 - m)

    # Pairwise interactions among active indices
    for ii, i in enumerate(active_indices):
        for j in active_indices[ii + 1:]:
            mu_i = pi[i]
            mu_j = pi[j]
            vi = 1.0 if votes[i] is True else 0.0
            vj = 1.0 if votes[j] is True else 0.0
            phi_ij = (vi - mu_i) * (vj - mu_j)
            log_p += rho[i][j] * phi_ij
    return log_p


def dajv_aggregate(
    votes: Sequence,
    calibration: DajvCalibration,
    accept_threshold: float = 0.95,
    abstain_threshold: float = 0.50,
) -> dict:
    """Apply the DAJV aggregation rule at deployment time.

    Returns posterior P(correct | votes), Clopper-Pearson interval
    derived from the effective sample size, and a recommendation in
    {COMMIT, ESCALATE, ABSTAIN, ABSTAIN_NO_VERIFIERS}.
    """
    n_working = sum(1 for v in votes if v is not None)
    n_accept = sum(1 for v in votes if v is True)
    if n_working == 0:
        return {"P_correct": None, "lower": None, "upper": None,
                "recommendation": "ABSTAIN_NO_VERIFIERS",
                "n_working": 0, "n_accept": 0}

    log_p_correct = _log_likelihood(votes, calibration.pi_pos, calibration.rho_pos)
    log_p_wrong = _log_likelihood(votes, calibration.pi_neg, calibration.rho_neg)

    log_prior_correct = math.log(max(calibration.prior_correct, 1e-9))
    log_prior_wrong = math.log(max(1 - calibration.prior_correct, 1e-9))

    log_post_correct = log_p_correct + log_prior_correct
    log_post_wrong = log_p_wrong + log_prior_wrong

    # Softmax to posterior probability
    mx = max(log_post_correct, log_post_wrong)
    P_correct = math.exp(log_post_correct - mx) / (
        math.exp(log_post_correct - mx) + math.exp(log_post_wrong - mx)
    )

    # CI: treat the n_accept among n_working votes as a sufficient
    # statistic for the binomial confidence band.
    lo, hi = clopper_pearson(n_accept, n_working)

    if P_correct >= accept_threshold:
        rec = "COMMIT"
    elif P_correct < abstain_threshold:
        rec = "ABSTAIN_LIKELY_WRONG"
    else:
        rec = "ESCALATE"
    return {"P_correct": P_correct, "lower": lo, "upper": hi,
            "recommendation": rec,
            "n_working": n_working, "n_accept": n_accept}
