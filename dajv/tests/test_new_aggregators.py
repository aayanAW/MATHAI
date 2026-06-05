"""Smoke tests for the new aggregator + analysis modules."""
from __future__ import annotations

import math

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.verge_proxy import verge_proxy_aggregate


def test_dajv_zero_rho_collapses_to_marginal_product():
    """If rho_pos == rho_neg == 0, DAJV should reduce to a Bayesian
    naive aggregator: P(correct | v) = prior * prod(pi_i^pos)^v_i /
    [prior * prod(pi_i^pos)^v_i + (1-prior) * prod(pi_i^neg)^v_i]
    (with pi_i raised to v_i and 1-pi_i raised to 1-v_i)."""
    k = 3
    cal = DajvCalibration(
        extractor_ids=["a", "b", "c"],
        pi_pos=[0.9, 0.8, 0.85],
        pi_neg=[0.1, 0.2, 0.05],
        rho_pos=[[0.0] * k for _ in range(k)],
        rho_neg=[[0.0] * k for _ in range(k)],
        prior_correct=0.5,
    )
    out = dajv_aggregate([True, True, True], cal)
    # Hand-compute the posterior numerator/denominator
    num = 0.5 * 0.9 * 0.8 * 0.85
    den = num + 0.5 * 0.1 * 0.2 * 0.05
    expected = num / den
    assert math.isclose(out["P_correct"], expected, abs_tol=1e-6)
    assert out["recommendation"] in {"COMMIT", "ESCALATE", "ABSTAIN_LIKELY_WRONG"}


def test_verge_proxy_threshold_4of4():
    """min_agree=4 with k=4 requires unanimous classification+accept."""
    out = verge_proxy_aggregate(
        votes=[True, True, True, True],
        classifications=["working"] * 4,
        candidate_verdicts=[True, True, True, True],
        min_agree=4,
    )
    assert out["recommendation"] == "COMMIT"


def test_verge_proxy_threshold_2of4_lowers_min_agree():
    """Test min_agree=2 — only 2 working+accept required."""
    out = verge_proxy_aggregate(
        votes=[True, True, False, False],
        classifications=["working", "working", "wrong_spec", "UNVERIFIABLE"],
        candidate_verdicts=[True, True, False, None],
        min_agree=2,
    )
    assert out["recommendation"] == "COMMIT"


def test_dajv_calibration_fit_handles_empty_correct():
    """If calibration set has no correct examples, prior=0 and DAJV
    should not crash (uses marginal default of 0.5)."""
    cal = DajvCalibration.fit(
        accept_cal=[[False, True, False], [False, False, True],
                    [True, False, False], [False, True, False]],
        problem_correct_cal=[False, False, False],
        extractor_ids=["a", "b", "c", "d"],
    )
    # prior_correct should be 0
    assert cal.prior_correct == 0.0
    # pi_pos defaults to 0.5 when no correct examples
    assert all(p == 0.5 for p in cal.pi_pos)
