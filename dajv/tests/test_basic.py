"""Smoke tests covering the core algorithms.

Run from dajv/:

    PYTHONPATH=. python3 -m pytest tests/

These tests use synthetic data only; no LLM API calls.
"""
from __future__ import annotations

import pytest

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_majority, naive_unanimous
from verifyensemble.aggregate.posterior import clopper_pearson
from verifyensemble.dependency.joint_fp import joint_fp_rate
from verifyensemble.dependency.kappa import cohen_kappa
from verifyensemble.dependency.matrix import DependencyMatrix
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.evaluation.mcnemar import mcnemar_mid_p
from verifyensemble.evaluation.risk_coverage import risk_coverage_auc
from verifyensemble.theory import (
    dajv_upper_bound,
    independence_lower_bound,
    required_n,
    union_upper_bound,
)


def test_clopper_pearson_basic():
    lo, hi = clopper_pearson(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    lo0, hi0 = clopper_pearson(0, 10)
    assert lo0 == 0.0
    assert hi0 < 1.0


def test_naive_unanimous_all_accept():
    out = naive_unanimous([True, True, True])
    assert out["recommendation"] == "COMMIT"
    assert out["n_accept"] == 3


def test_naive_unanimous_mixed():
    out = naive_unanimous([True, False, True])
    assert out["recommendation"] == "ABSTAIN"


def test_naive_majority():
    assert naive_majority([True, True, False])["recommendation"] == "COMMIT"
    assert naive_majority([True, False, False])["recommendation"] == "ABSTAIN"


def test_cohen_kappa_perfect_agreement():
    a = [True, False, True, False]
    assert cohen_kappa(a, a) == pytest.approx(1.0)


def test_cohen_kappa_independence():
    a = [True, True, False, False]
    b = [True, False, True, False]
    k = cohen_kappa(a, b)
    assert abs(k) < 0.1


def test_joint_fp_rate_independent_case():
    # Two verifiers, half the wrong items accepted by each, but
    # acceptance is independent across them
    accept_i = [True, False, True, False, True, False]
    accept_j = [True, True, False, False, True, False]
    correct = [False] * 6
    out = joint_fp_rate(accept_i, accept_j, correct)
    assert out["n_wrong"] == 6
    assert 0 <= out["ratio"]


def test_dajv_calibration_runs():
    # 3 verifiers x 20 problems; binary acceptance; mixed labels
    accept = [
        [True] * 10 + [False] * 10,
        [True] * 8 + [False] * 2 + [False] * 10,
        [True] * 9 + [False] * 1 + [False] * 10,
    ]
    correct = [True] * 10 + [False] * 10
    cal = DajvCalibration.fit(accept, correct, ["E1", "E2", "E3"])
    out = dajv_aggregate([True, True, True], cal)
    assert out["recommendation"] in ("COMMIT", "ESCALATE")
    assert 0.0 <= out["P_correct"] <= 1.0


def test_dependency_matrix_construction():
    accept = [
        [True, False, True, False],
        [True, True, False, False],
        [False, True, True, False],
    ]
    correct = [False, False, False, False]
    D = DependencyMatrix.from_accept(accept, correct, ["A", "B", "C"])
    assert D.n_problems == 4
    assert D.n_wrong == 4
    assert len(D.pi) == 3


def test_ece_brier_basic():
    confs = [0.9, 0.8, 0.7, 0.1]
    correct = [True, True, False, False]
    ece = expected_calibration_error(confs, correct, n_bins=2)
    brier = brier_score(confs, correct)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 1.0


def test_risk_coverage_auc():
    confs = [0.9, 0.7, 0.5, 0.3]
    correct = [True, True, False, False]
    auc = risk_coverage_auc(confs, correct)
    assert 0.0 <= auc <= 1.0


def test_mcnemar_mid_p_basic():
    # A always correct, B always wrong: discordant pairs = (0, n)
    a = [True] * 10
    b = [False] * 10
    out = mcnemar_mid_p(a, b)
    assert out["b01"] == 0
    assert out["b10"] == 10
    assert out["mid_p"] < 0.01


def test_theorem1_bound_ordering():
    # independence_lower_bound <= dajv_upper_bound <= union_upper_bound
    pi = [0.1, 0.2, 0.15]
    rho = [[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]]
    indep = independence_lower_bound(pi)
    dajv = dajv_upper_bound(pi, rho)
    union = union_upper_bound(pi)
    assert indep <= dajv <= union + 1e-9


def test_theorem2_required_n_monotone():
    # Tighter eps requires larger n
    n_small = required_n(4, 0.20, delta=0.05)
    n_large = required_n(4, 0.05, delta=0.05)
    assert n_large > n_small
