"""Cross-modality DAJV (H7' attempt).

Uses the empirical finding that the structural (script-classifies-as-working)
and executable (script-accepts-candidate) modalities are nearly independent
across LLMs to factorize the joint posterior.

Standard DAJV models the joint distribution of the $k$ ``accept'' votes
(accept_i = struct_i AND exec_i). This module fits TWO DAJV calibrations
in parallel:

  - DAJV_s: on the $k$ structural votes (struct_i).
  - DAJV_x: on the $k$ executable votes (exec_i).

Under the cross-modality independence approximation,

    P(C=1 | s, x) \\propto P(C=1) * P(s | C=1) * P(x | C=1)
                       / P(C=1) ** {1}       # avoid double-counting prior

The two DAJV calibrations each give P(C=1 | s) and P(C=1 | x); combining
under cross-modality independence yields

    P_xmod(C=1) = (p_s * p_x / pi) / (p_s * p_x / pi + (1 - p_s) * (1 - p_x) / (1 - pi))

where pi is the prior P(C=1) on the calibration set.

Empirical question: does the cross-modality combination improve the
calibrated operating point at the default $\tau = 0.95$ threshold?
This module implements the rule; ``scripts/run_xmod_dajv.py`` evaluates it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate


@dataclass
class XmodJointCalibration:
    """Per-LLM joint (struct, exec) calibration.

    For each LLM $i$, stores the 2x2 empirical distribution
    $P(s_i, x_i | C=c)$ for $c \\in \\{0,1\\}$.

    Naive Bayes across LLMs (treating per-LLM joint as independent
    across $i$): operationally exploits cross-modality independence
    via the richer 4-cell signal instead of the binarized accept.
    """
    extractor_ids: list[str]
    # joint_pos[i][s][x] = P(s_i = s, x_i = x | C = 1)
    joint_pos: list[list[list[float]]]
    joint_neg: list[list[list[float]]]
    prior_correct: float

    @classmethod
    def fit(
        cls,
        struct_cal: Sequence[Sequence[bool]],
        exec_cal: Sequence[Sequence[bool]],
        problem_correct_cal: Sequence[bool],
        extractor_ids: list[str],
        smooth: float = 0.5,
    ) -> "XmodJointCalibration":
        k = len(extractor_ids)
        n = len(problem_correct_cal)
        correct_idx = [j for j, c in enumerate(problem_correct_cal) if c]
        wrong_idx = [j for j, c in enumerate(problem_correct_cal) if not c]

        def _joint(idx: list[int]) -> list[list[list[float]]]:
            J = [[[smooth, smooth], [smooth, smooth]] for _ in range(k)]
            for i in range(k):
                for t in idx:
                    s = 1 if struct_cal[i][t] else 0
                    x = 1 if exec_cal[i][t] else 0
                    J[i][s][x] += 1.0
                # Normalize
                total = sum(J[i][s][x] for s in (0, 1) for x in (0, 1))
                for s in (0, 1):
                    for x in (0, 1):
                        J[i][s][x] /= total
            return J

        prior = len(correct_idx) / max(n, 1)
        return cls(
            extractor_ids=list(extractor_ids),
            joint_pos=_joint(correct_idx),
            joint_neg=_joint(wrong_idx),
            prior_correct=prior,
        )


def xmod_joint_aggregate(
    struct_votes: Sequence,
    exec_votes: Sequence,
    calibration: XmodJointCalibration,
    accept_threshold: float = 0.95,
    abstain_threshold: float = 0.50,
) -> dict:
    """Naive-Bayes across LLMs with per-LLM joint (struct, exec) cells.

    Aggregates the 4-cell empirical likelihoods across LLMs assuming
    independence across $i$ given $C$. Within each LLM, the 2x2 cell
    captures all of (struct=T or F) x (exec=T or F).
    """
    k = len(calibration.extractor_ids)
    if not struct_votes or not exec_votes:
        return {"P_correct": None, "recommendation": "ABSTAIN_NO_VERIFIERS",
                "stage": "xmod_joint_no_inputs"}

    log_p1 = math.log(max(calibration.prior_correct, 1e-12))
    log_p0 = math.log(max(1 - calibration.prior_correct, 1e-12))
    for i in range(k):
        s = struct_votes[i]
        x = exec_votes[i]
        if s is None or x is None:
            continue
        si = 1 if s else 0
        xi = 1 if x else 0
        log_p1 += math.log(max(calibration.joint_pos[i][si][xi], 1e-12))
        log_p0 += math.log(max(calibration.joint_neg[i][si][xi], 1e-12))

    mx = max(log_p1, log_p0)
    p_correct = math.exp(log_p1 - mx) / (
        math.exp(log_p1 - mx) + math.exp(log_p0 - mx)
    )

    if p_correct >= accept_threshold:
        rec = "COMMIT"
    elif p_correct < abstain_threshold:
        rec = "ABSTAIN_LIKELY_WRONG"
    else:
        rec = "ESCALATE"
    return {
        "P_correct": p_correct,
        "lower": None, "upper": None,
        "recommendation": rec,
        "stage": "xmod_joint",
    }


@dataclass
class XmodCalibration:
    """Pair of DAJV calibrations on the two modalities."""
    struct: DajvCalibration
    exec_: DajvCalibration
    prior_correct: float

    @classmethod
    def fit(
        cls,
        struct_cal: Sequence[Sequence[bool]],
        exec_cal: Sequence[Sequence[bool]],
        problem_correct_cal: Sequence[bool],
        extractor_ids: list[str],
    ) -> "XmodCalibration":
        struct_dajv = DajvCalibration.fit(
            struct_cal, problem_correct_cal,
            [f"s_{e}" for e in extractor_ids],
        )
        exec_dajv = DajvCalibration.fit(
            exec_cal, problem_correct_cal,
            [f"x_{e}" for e in extractor_ids],
        )
        n = len(problem_correct_cal)
        prior = sum(1 for c in problem_correct_cal if c) / max(n, 1)
        return cls(struct=struct_dajv, exec_=exec_dajv, prior_correct=prior)


def xmod_aggregate(
    struct_votes: Sequence,
    exec_votes: Sequence,
    calibration: XmodCalibration,
    accept_threshold: float = 0.95,
    abstain_threshold: float = 0.50,
) -> dict:
    """Cross-modality DAJV aggregation.

    Returns dict with the same keys as ``dajv_aggregate`` plus
    ``stage`` in {xmod_factorized, xmod_struct_only, xmod_exec_only,
    xmod_abstain_no_inputs}.
    """
    if not struct_votes and not exec_votes:
        return {
            "P_correct": None, "lower": None, "upper": None,
            "recommendation": "ABSTAIN_NO_VERIFIERS",
            "stage": "xmod_abstain_no_inputs",
        }

    out_s = dajv_aggregate(struct_votes, calibration.struct,
                           accept_threshold, abstain_threshold)
    out_x = dajv_aggregate(exec_votes, calibration.exec_,
                           accept_threshold, abstain_threshold)
    p_s = out_s.get("P_correct")
    p_x = out_x.get("P_correct")

    # If a modality is entirely absent (all None votes), fall back to
    # the other modality alone.
    if p_s is None and p_x is None:
        return {
            "P_correct": None, "lower": None, "upper": None,
            "recommendation": "ABSTAIN_NO_VERIFIERS",
            "stage": "xmod_abstain_no_inputs",
        }
    if p_s is None:
        return {
            "P_correct": p_x,
            "lower": out_x.get("lower"), "upper": out_x.get("upper"),
            "recommendation": out_x["recommendation"],
            "stage": "xmod_exec_only",
        }
    if p_x is None:
        return {
            "P_correct": p_s,
            "lower": out_s.get("lower"), "upper": out_s.get("upper"),
            "recommendation": out_s["recommendation"],
            "stage": "xmod_struct_only",
        }

    pi = max(min(calibration.prior_correct, 1.0 - 1e-9), 1e-9)
    # P(C=1 | s, x) under cross-modality independence given C
    log_num = math.log(p_s) + math.log(p_x) - math.log(pi)
    log_den_correct = log_num
    log_den_wrong = (
        math.log(max(1 - p_s, 1e-12))
        + math.log(max(1 - p_x, 1e-12))
        - math.log(max(1 - pi, 1e-12))
    )
    mx = max(log_den_correct, log_den_wrong)
    p_xmod = math.exp(log_den_correct - mx) / (
        math.exp(log_den_correct - mx) + math.exp(log_den_wrong - mx)
    )

    if p_xmod >= accept_threshold:
        rec = "COMMIT"
    elif p_xmod < abstain_threshold:
        rec = "ABSTAIN_LIKELY_WRONG"
    else:
        rec = "ESCALATE"

    return {
        "P_correct": p_xmod,
        "lower": None, "upper": None,
        "recommendation": rec,
        "stage": "xmod_factorized",
        "p_struct": p_s,
        "p_exec": p_x,
    }


def xmod_agreement_aggregate(
    struct_votes: Sequence,
    exec_votes: Sequence,
    calibration: XmodCalibration,
    accept_threshold: float = 0.95,
    abstain_threshold: float = 0.50,
    accept_threshold_per_modality: float = 0.85,
) -> dict:
    """Cross-modality AGREEMENT gating (alternative H7' aggregator).

    Commit only when both per-modality DAJVs cross
    ``accept_threshold_per_modality`` independently. This uses the
    cross-modality independence ($\\kappa \\approx 0.02$ across $42$
    pairs) as an evidence-amplification signal: requiring agreement
    across two near-independent modalities is much stronger than
    requiring agreement within one.

    Reports the geometric mean of the two posteriors as $P_{\\rm correct}$.
    """
    if not struct_votes and not exec_votes:
        return {"P_correct": None, "recommendation": "ABSTAIN_NO_VERIFIERS",
                "stage": "xmod_agree_no_inputs"}

    out_s = dajv_aggregate(struct_votes, calibration.struct)
    out_x = dajv_aggregate(exec_votes, calibration.exec_)
    p_s = out_s.get("P_correct")
    p_x = out_x.get("P_correct")

    if p_s is None and p_x is None:
        return {"P_correct": None, "recommendation": "ABSTAIN_NO_VERIFIERS",
                "stage": "xmod_agree_no_inputs"}

    p_combined = (
        math.sqrt(p_s * p_x) if (p_s is not None and p_x is not None)
        else (p_s if p_s is not None else p_x)
    )

    s_ok = p_s is not None and p_s >= accept_threshold_per_modality
    x_ok = p_x is not None and p_x >= accept_threshold_per_modality

    if s_ok and x_ok and p_combined is not None and p_combined >= accept_threshold:
        rec = "COMMIT"
    elif (s_ok and not x_ok) or (x_ok and not s_ok):
        # Cross-modality disagreement -> ESCALATE explicitly
        rec = "ESCALATE_XMOD_DISAGREE"
    elif p_combined is not None and p_combined < abstain_threshold:
        rec = "ABSTAIN_LIKELY_WRONG"
    else:
        rec = "ESCALATE"

    return {
        "P_correct": p_combined,
        "lower": None, "upper": None,
        "recommendation": rec,
        "stage": "xmod_agreement",
        "p_struct": p_s,
        "p_exec": p_x,
        "s_ok": s_ok,
        "x_ok": x_ok,
    }


def xmod_struct_gated_exec_dajv_aggregate(
    struct_votes: Sequence,
    exec_votes: Sequence,
    exec_dajv_calibration: DajvCalibration,
    accept_threshold: float = 0.95,
    abstain_threshold: float = 0.50,
    min_struct_agree: int = 3,
) -> dict:
    """Final H7' variant: exec-only DAJV gated by structural-agreement.

    Compute default DAJV on the executable signal alone. COMMIT only if
    the DAJV exec posterior crosses ``accept_threshold`` AND at least
    ``min_struct_agree`` LLMs emit a script that classifies as
    structurally working.

    Uses the cross-modality independence by treating the structural
    signal as an additional confidence gate on top of the executable
    DAJV --- without trying to model their joint distribution.
    """
    k = len(struct_votes)
    if k == 0 or len(exec_votes) != k:
        return {"P_correct": None, "recommendation": "ABSTAIN_NO_VERIFIERS",
                "stage": "xmod_struct_gated_no_inputs"}

    out = dajv_aggregate(exec_votes, exec_dajv_calibration,
                         accept_threshold, abstain_threshold)
    p = out.get("P_correct")
    n_struct_ok = sum(1 for s in struct_votes if s is True)

    if p is None:
        return {**out, "stage": "xmod_struct_gated_no_exec"}

    if p >= accept_threshold and n_struct_ok >= min_struct_agree:
        rec = "COMMIT"
    elif p >= accept_threshold and n_struct_ok < min_struct_agree:
        # exec DAJV says high P_correct but struct disagrees -> escalate
        rec = "ESCALATE_STRUCT_DISAGREE"
    elif p < abstain_threshold:
        rec = "ABSTAIN_LIKELY_WRONG"
    else:
        rec = "ESCALATE"
    return {
        "P_correct": p,
        "lower": out.get("lower"), "upper": out.get("upper"),
        "recommendation": rec,
        "stage": "xmod_struct_gated_exec",
        "n_struct_ok": n_struct_ok,
    }


# ---------------------------------------------------------------------------
# Block-Sparse Ising Aggregator (BSIA) -- H7' deeper attempt.
#
# Empirical motivation. The cross-modality measurement in
# Section 5.3 of the paper reports a block-sparse pairwise dependency
# structure on the 2k-dimensional verifier vector (s, x):
#
#   - within-modality cross-LLM kappa ~ 0.72 (DENSE),
#   - within-LLM cross-modality kappa ~ 0.04--0.13 (small),
#   - cross-LLM cross-modality kappa ~ 0.02--0.06 (NEAR ZERO).
#
# A natural model is an Ising/Bahadur factorization of P(s, x | C)
# with structural zeros on the cross-LLM cross-modality pairs.
# Concretely, the leading term is a per-LLM 2x2 cell P_i(s_i, x_i | C)
# (which captures both per-LLM marginals AND within-LLM s-x
# dependence), and the correction terms are within-modality cross-LLM
# centered Pearson interactions on (s_i, s_j) and (x_i, x_j) for
# i < j.  The cross-LLM cross-modality interactions are PINNED TO
# ZERO -- the empirically motivated sparsity prior.
#
# Compared to the four pre-registered H7' variants:
#
#   xmod_factorized:        ignores within-LLM s-x coupling (uses
#                           per-modality DAJV posteriors as independent),
#                           and the prior-divide hack double-counts.
#   xmod_joint NB:          captures per-LLM 2x2 cell, but treats LLMs
#                           as independent (misses kappa = 0.72).
#   xmod_agreement, gated:  hard gates -- too restrictive on coverage.
#
# BSIA is the unique combination of (i) per-LLM 2x2 cell, (ii)
# within-modality cross-LLM Pearson correction, (iii) zero
# cross-LLM cross-modality, and is the structural Bahadur expansion
# the empirical kappa block-sparsity directly implies.  Parameter
# count: 4 cells * k * 2 classes + k*(k-1) cross-LLM rho's * 2 classes
# = 8k + 2k(k-1).  For k = 7 that is 56 + 84 = 140 (vs full Ising on
# 14 binary variables = 14 + 91 = 105 per class -> 210 total),
# slightly fewer than full Ising and structurally motivated.
# ---------------------------------------------------------------------------


@dataclass
class BlockSparseIsingCalibration:
    """Block-sparse Ising calibration for the (s, x) verifier vector.

    Stores per-LLM joint cells and within-modality cross-LLM
    second-order Pearson interactions.  Cross-LLM cross-modality
    interactions are pinned to zero by construction.
    """
    extractor_ids: list[str]
    # k x 2 x 2: P(s_i = ss, x_i = xx | C = c)
    cells_pos: list[list[list[float]]]
    cells_neg: list[list[list[float]]]
    # k x k: standardized cov of (s_i, s_j) and (x_i, x_j) conditional on C.
    rho_pos_struct: list[list[float]]
    rho_neg_struct: list[list[float]]
    rho_pos_exec: list[list[float]]
    rho_neg_exec: list[list[float]]
    # Marginals (derived from cells), used for centering.
    mu_pos_struct: list[float]
    mu_neg_struct: list[float]
    mu_pos_exec: list[float]
    mu_neg_exec: list[float]
    prior_correct: float
    # Optional shrinkage toward zero for rho terms (regularization).
    rho_shrinkage: float
    # Optional temperature for posterior softening; defaults to 1.0
    # (no scaling). Fit via ``fit_bsia_temperature``.
    temperature: float = 1.0

    @classmethod
    def fit(
        cls,
        struct_cal: Sequence[Sequence[bool]],
        exec_cal: Sequence[Sequence[bool]],
        problem_correct_cal: Sequence[bool],
        extractor_ids: list[str],
        smooth: float = 0.5,
        rho_shrinkage: float = 0.5,
    ) -> "BlockSparseIsingCalibration":
        """Fit the block-sparse Ising calibration.

        Args:
            struct_cal: k x n_cal structural votes (bool).
            exec_cal:   k x n_cal executable votes (bool).
            problem_correct_cal: n_cal solver-correctness labels.
            extractor_ids: k LLM identifiers.
            smooth: Laplace smoothing for the 2x2 cells.
            rho_shrinkage: linear shrinkage of rho toward zero, in
                [0, 1].  0 = no shrinkage, 1 = collapse to per-LLM
                naive Bayes.  Default 0.5 matches the DAJV default's
                empirical sweet spot.
        """
        k = len(extractor_ids)
        n = len(problem_correct_cal)
        if k != len(struct_cal) or k != len(exec_cal):
            raise ValueError(
                f"k mismatch: extractor_ids has {k}, "
                f"struct_cal has {len(struct_cal)}, "
                f"exec_cal has {len(exec_cal)}"
            )
        for i in range(k):
            if len(struct_cal[i]) != n or len(exec_cal[i]) != n:
                raise ValueError(
                    f"row {i} length mismatch with problem_correct_cal"
                )

        correct_idx = [j for j, c in enumerate(problem_correct_cal) if c]
        wrong_idx = [j for j, c in enumerate(problem_correct_cal) if not c]
        prior = len(correct_idx) / max(n, 1)

        def _cells(idx: list[int]) -> list[list[list[float]]]:
            """Per-LLM 2x2 cell P_i(s_i, x_i | C) with Laplace smoothing."""
            J = [[[smooth, smooth], [smooth, smooth]] for _ in range(k)]
            for i in range(k):
                for t in idx:
                    s = 1 if struct_cal[i][t] else 0
                    x = 1 if exec_cal[i][t] else 0
                    J[i][s][x] += 1.0
                total = sum(J[i][s][x] for s in (0, 1) for x in (0, 1))
                for s in (0, 1):
                    for x in (0, 1):
                        J[i][s][x] /= total
            return J

        def _marg(cells: list[list[list[float]]], axis: str) -> list[float]:
            """Marginalize the cells to per-LLM single-axis probability."""
            out = []
            for i in range(k):
                if axis == "struct":
                    p = cells[i][1][0] + cells[i][1][1]
                else:
                    p = cells[i][0][1] + cells[i][1][1]
                out.append(p)
            return out

        def _rho(
            idx: list[int],
            votes: Sequence[Sequence[bool]],
            mu: list[float],
        ) -> list[list[float]]:
            """Standardized cov of (votes_i, votes_j) over idx."""
            R = [[0.0] * k for _ in range(k)]
            if not idx:
                return R
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    mu_i, mu_j = mu[i], mu[j]
                    s = 0.0
                    for t in idx:
                        vi = 1.0 if votes[i][t] else 0.0
                        vj = 1.0 if votes[j][t] else 0.0
                        s += (vi - mu_i) * (vj - mu_j)
                    cov = s / len(idx)
                    var_i = max(mu_i * (1 - mu_i), 1e-6)
                    var_j = max(mu_j * (1 - mu_j), 1e-6)
                    raw = cov / math.sqrt(var_i * var_j)
                    R[i][j] = (1.0 - rho_shrinkage) * raw
            return R

        cells_pos = _cells(correct_idx)
        cells_neg = _cells(wrong_idx)
        mu_pos_s = _marg(cells_pos, "struct")
        mu_neg_s = _marg(cells_neg, "struct")
        mu_pos_x = _marg(cells_pos, "exec")
        mu_neg_x = _marg(cells_neg, "exec")
        rho_pos_s = _rho(correct_idx, struct_cal, mu_pos_s)
        rho_neg_s = _rho(wrong_idx, struct_cal, mu_neg_s)
        rho_pos_x = _rho(correct_idx, exec_cal, mu_pos_x)
        rho_neg_x = _rho(wrong_idx, exec_cal, mu_neg_x)

        return cls(
            extractor_ids=list(extractor_ids),
            cells_pos=cells_pos,
            cells_neg=cells_neg,
            rho_pos_struct=rho_pos_s,
            rho_neg_struct=rho_neg_s,
            rho_pos_exec=rho_pos_x,
            rho_neg_exec=rho_neg_x,
            mu_pos_struct=mu_pos_s,
            mu_neg_struct=mu_neg_s,
            mu_pos_exec=mu_pos_x,
            mu_neg_exec=mu_neg_x,
            prior_correct=prior,
            rho_shrinkage=rho_shrinkage,
        )


def _bsia_log_likelihood(
    struct_votes: Sequence,
    exec_votes: Sequence,
    cells: list[list[list[float]]],
    rho_struct: list[list[float]],
    rho_exec: list[list[float]],
    mu_struct: list[float],
    mu_exec: list[float],
) -> float:
    """Compute the BSIA log-likelihood under one class.

    log P(s, x | C) = sum_i log P_i(s_i, x_i | C)              [cells]
                      + sum_{i<j} rho_ij^s phi^s_ij             [within-struct]
                      + sum_{i<j} rho_ij^x phi^x_ij             [within-exec]
                      + 0                                       [cross-LLM x-mod, pinned]

    None votes are marginalized by skipping that LLM's cell and any
    pairwise terms it participates in.
    """
    k = len(cells)
    if k == 0:
        return 0.0
    active = [
        i for i in range(k)
        if struct_votes[i] is not None and exec_votes[i] is not None
    ]
    if not active:
        return 0.0

    log_p = 0.0
    for i in active:
        s = 1 if struct_votes[i] else 0
        x = 1 if exec_votes[i] else 0
        p_cell = max(cells[i][s][x], 1e-12)
        log_p += math.log(p_cell)

    for ii, i in enumerate(active):
        for j in active[ii + 1:]:
            si = 1.0 if struct_votes[i] else 0.0
            sj = 1.0 if struct_votes[j] else 0.0
            log_p += rho_struct[i][j] * (si - mu_struct[i]) * (sj - mu_struct[j])

            xi = 1.0 if exec_votes[i] else 0.0
            xj = 1.0 if exec_votes[j] else 0.0
            log_p += rho_exec[i][j] * (xi - mu_exec[i]) * (xj - mu_exec[j])
    return log_p


def bsia_aggregate(
    struct_votes: Sequence,
    exec_votes: Sequence,
    calibration: "BlockSparseIsingCalibration",
    accept_threshold: float = 0.95,
    abstain_threshold: float = 0.50,
) -> dict:
    """Block-Sparse Ising aggregation of the (s, x) verifier vector.

    Returns dict with the same keys as ``dajv_aggregate``.  The
    ``stage`` field is ``"bsia"`` on the standard path.
    """
    k = len(calibration.extractor_ids)
    if not struct_votes or not exec_votes:
        return {"P_correct": None, "recommendation": "ABSTAIN_NO_VERIFIERS",
                "stage": "bsia_no_inputs"}
    if len(struct_votes) != k or len(exec_votes) != k:
        raise ValueError(
            f"struct_votes / exec_votes must have length k={k}, got "
            f"{len(struct_votes)} / {len(exec_votes)}"
        )
    n_active = sum(
        1 for i in range(k)
        if struct_votes[i] is not None and exec_votes[i] is not None
    )
    if n_active == 0:
        return {"P_correct": None, "recommendation": "ABSTAIN_NO_VERIFIERS",
                "stage": "bsia_no_active"}

    log_p_correct = _bsia_log_likelihood(
        struct_votes, exec_votes,
        calibration.cells_pos,
        calibration.rho_pos_struct, calibration.rho_pos_exec,
        calibration.mu_pos_struct, calibration.mu_pos_exec,
    )
    log_p_wrong = _bsia_log_likelihood(
        struct_votes, exec_votes,
        calibration.cells_neg,
        calibration.rho_neg_struct, calibration.rho_neg_exec,
        calibration.mu_neg_struct, calibration.mu_neg_exec,
    )

    log_prior_correct = math.log(max(calibration.prior_correct, 1e-9))
    log_prior_wrong = math.log(max(1 - calibration.prior_correct, 1e-9))
    log_post_correct = log_p_correct + log_prior_correct
    log_post_wrong = log_p_wrong + log_prior_wrong

    mx = max(log_post_correct, log_post_wrong)
    P_correct = math.exp(log_post_correct - mx) / (
        math.exp(log_post_correct - mx) + math.exp(log_post_wrong - mx)
    )

    # Apply temperature scaling if the calibration was fit with it.
    temperature = getattr(calibration, "temperature", 1.0)
    if temperature != 1.0:
        log_post_correct_t = log_post_correct / temperature
        log_post_wrong_t = log_post_wrong / temperature
        mxt = max(log_post_correct_t, log_post_wrong_t)
        P_correct = math.exp(log_post_correct_t - mxt) / (
            math.exp(log_post_correct_t - mxt) + math.exp(log_post_wrong_t - mxt)
        )

    if P_correct >= accept_threshold:
        rec = "COMMIT"
    elif P_correct < abstain_threshold:
        rec = "ABSTAIN_LIKELY_WRONG"
    else:
        rec = "ESCALATE"
    return {
        "P_correct": P_correct,
        "lower": None, "upper": None,
        "recommendation": rec,
        "stage": "bsia",
        "n_active": n_active,
    }


def fit_bsia_temperature(
    cal: "BlockSparseIsingCalibration",
    struct_cal: Sequence[Sequence[bool]],
    exec_cal: Sequence[Sequence[bool]],
    problem_correct_cal: Sequence[bool],
    grid: Sequence[float] = (
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0,
    ),
) -> float:
    """Fit a single-parameter temperature for BSIA on calibration data.

    Minimizes negative log-likelihood of the binary labels under the
    temperature-scaled BSIA posterior.  Returns the chosen temperature
    in ``grid``.  The temperature can then be assigned to the
    calibration object via ``cal.temperature = T`` so that
    ``bsia_aggregate`` applies it automatically.
    """
    k = len(cal.extractor_ids)
    if k == 0:
        return 1.0

    def nll(T: float) -> float:
        if T <= 0:
            return float("inf")
        s = 0.0
        for j in range(len(problem_correct_cal)):
            sv = [bool(struct_cal[i][j]) for i in range(k)]
            xv = [bool(exec_cal[i][j]) for i in range(k)]
            log_p1 = _bsia_log_likelihood(
                sv, xv, cal.cells_pos,
                cal.rho_pos_struct, cal.rho_pos_exec,
                cal.mu_pos_struct, cal.mu_pos_exec,
            ) + math.log(max(cal.prior_correct, 1e-9))
            log_p0 = _bsia_log_likelihood(
                sv, xv, cal.cells_neg,
                cal.rho_neg_struct, cal.rho_neg_exec,
                cal.mu_neg_struct, cal.mu_neg_exec,
            ) + math.log(max(1 - cal.prior_correct, 1e-9))
            log_p1_t = log_p1 / T
            log_p0_t = log_p0 / T
            mx = max(log_p1_t, log_p0_t)
            denom = math.exp(log_p1_t - mx) + math.exp(log_p0_t - mx)
            p1 = math.exp(log_p1_t - mx) / denom
            p1 = min(max(p1, 1e-12), 1 - 1e-12)
            y = 1 if problem_correct_cal[j] else 0
            s -= y * math.log(p1) + (1 - y) * math.log(1 - p1)
        return s

    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        cur = nll(T)
        if cur < best_nll:
            best_nll, best_T = cur, T
    return best_T


def _isotonic_pav(
    raw_probs: Sequence[float], labels: Sequence[bool]
) -> tuple[list[float], list[float]]:
    """Pool-Adjacent-Violators isotonic regression.

    Maps ``raw_probs`` to monotone-calibrated probabilities under the
    empirical labels.  Returns sorted ``(xs, ys)`` representing the
    piecewise-constant calibrator (interpolate stepwise at deploy
    time).
    """
    n = len(raw_probs)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: raw_probs[i])
    xs = [raw_probs[i] for i in order]
    ys = [1.0 if labels[i] else 0.0 for i in order]
    # Levels: list of (mean_y, weight, lo_x, hi_x).
    levels: list[list[float]] = [[ys[i], 1.0, xs[i], xs[i]] for i in range(n)]
    i = 0
    while i < len(levels) - 1:
        if levels[i][0] > levels[i + 1][0] + 1e-12:
            v0, w0, lo0, hi0 = levels[i]
            v1, w1, lo1, hi1 = levels[i + 1]
            merged = [
                (v0 * w0 + v1 * w1) / (w0 + w1),
                w0 + w1, lo0, hi1,
            ]
            levels = levels[:i] + [merged] + levels[i + 2:]
            if i > 0:
                i -= 1
        else:
            i += 1
    cal_xs = [lvl[2] for lvl in levels]
    cal_ys = [lvl[0] for lvl in levels]
    return cal_xs, cal_ys


def _isotonic_predict(
    raw_p: float, cal_xs: Sequence[float], cal_ys: Sequence[float]
) -> float:
    """Look up the calibrated probability for ``raw_p``.

    Piecewise-constant interpolation: for each level with lo_x = cal_xs[i],
    the calibrated value applies to all raw values in [cal_xs[i], cal_xs[i+1]).
    Below the lowest cal_x, return the lowest cal_y; above the highest,
    return the highest cal_y.
    """
    if not cal_xs:
        return raw_p
    if raw_p <= cal_xs[0]:
        return cal_ys[0]
    if raw_p >= cal_xs[-1]:
        return cal_ys[-1]
    lo, hi = 0, len(cal_xs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cal_xs[mid] <= raw_p:
            lo = mid
        else:
            hi = mid
    return cal_ys[lo]


def fit_bsia_isotonic(
    cal: "BlockSparseIsingCalibration",
    struct_cal: Sequence[Sequence[bool]],
    exec_cal: Sequence[Sequence[bool]],
    problem_correct_cal: Sequence[bool],
) -> tuple[list[float], list[float]]:
    """Fit isotonic recalibration on top of BSIA on calibration data."""
    k = len(cal.extractor_ids)
    raws: list[float] = []
    labels: list[bool] = []
    for j in range(len(problem_correct_cal)):
        sv = [bool(struct_cal[i][j]) for i in range(k)]
        xv = [bool(exec_cal[i][j]) for i in range(k)]
        out = bsia_aggregate(sv, xv, cal)
        p = out.get("P_correct")
        if p is None:
            continue
        raws.append(p)
        labels.append(problem_correct_cal[j])
    return _isotonic_pav(raws, labels)


def bsia_ensemble_aggregate(
    struct_votes: Sequence,
    exec_votes: Sequence,
    cal_fixed: "BlockSparseIsingCalibration",
    cal_temp: "BlockSparseIsingCalibration",
    iso_xs: Sequence[float],
    iso_ys: Sequence[float],
    accept_threshold: float = 0.95,
    abstain_threshold: float = 0.50,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Bayesian model averaging across three BSIA variants.

    Combines the raw BSIA, temperature-scaled BSIA, and
    isotonic-recalibrated BSIA posteriors with optional weights
    (default equal).  Returns a dict with the averaged posterior and
    each individual posterior for inspection.
    """
    w_fixed, w_temp, w_iso = weights
    out_f = bsia_aggregate(struct_votes, exec_votes, cal_fixed,
                           accept_threshold=accept_threshold,
                           abstain_threshold=abstain_threshold)
    out_t = bsia_aggregate(struct_votes, exec_votes, cal_temp,
                           accept_threshold=accept_threshold,
                           abstain_threshold=abstain_threshold)
    out_i = bsia_isotonic_aggregate(struct_votes, exec_votes,
                                    cal_fixed, iso_xs, iso_ys,
                                    accept_threshold=accept_threshold,
                                    abstain_threshold=abstain_threshold)
    ps = [out_f.get("P_correct"), out_t.get("P_correct"),
          out_i.get("P_correct")]
    ws = [w_fixed, w_temp, w_iso]
    pw = [(p, w) for p, w in zip(ps, ws) if p is not None]
    if not pw:
        return {"P_correct": None, "recommendation": "ABSTAIN_NO_VERIFIERS",
                "stage": "bsia_ensemble_no_inputs"}
    total_w = sum(w for _, w in pw)
    p_avg = sum(p * w for p, w in pw) / max(total_w, 1e-9)
    if p_avg >= accept_threshold:
        rec = "COMMIT"
    elif p_avg < abstain_threshold:
        rec = "ABSTAIN_LIKELY_WRONG"
    else:
        rec = "ESCALATE"
    return {
        "P_correct": p_avg,
        "P_correct_fixed": ps[0],
        "P_correct_temp": ps[1],
        "P_correct_iso": ps[2],
        "lower": None, "upper": None,
        "recommendation": rec,
        "stage": "bsia_ensemble",
    }


def bsia_isotonic_aggregate(
    struct_votes: Sequence,
    exec_votes: Sequence,
    calibration: "BlockSparseIsingCalibration",
    cal_xs: Sequence[float],
    cal_ys: Sequence[float],
    accept_threshold: float = 0.95,
    abstain_threshold: float = 0.50,
) -> dict:
    """BSIA aggregation with isotonic post-hoc calibration."""
    raw = bsia_aggregate(struct_votes, exec_votes, calibration,
                         accept_threshold=accept_threshold,
                         abstain_threshold=abstain_threshold)
    p_raw = raw.get("P_correct")
    if p_raw is None:
        return raw
    p_cal = _isotonic_predict(p_raw, cal_xs, cal_ys)
    if p_cal >= accept_threshold:
        rec = "COMMIT"
    elif p_cal < abstain_threshold:
        rec = "ABSTAIN_LIKELY_WRONG"
    else:
        rec = "ESCALATE"
    return {
        "P_correct": p_cal,
        "P_correct_raw": p_raw,
        "lower": None, "upper": None,
        "recommendation": rec,
        "stage": "bsia_isotonic",
        "n_active": raw.get("n_active"),
    }
