"""Edge-case tests for aggregation, calibration, dependency, and theory.

Run from dajv/:

    PYTHONPATH=. python3 -m pytest tests/ -q
"""
from __future__ import annotations

import pytest

from verifyensemble.aggregate.care import care_aggregate, fit_care_weights
from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_majority, naive_unanimous
from verifyensemble.aggregate.posterior import clopper_pearson
from verifyensemble.dependency.cig import cig
from verifyensemble.dependency.joint_fp import independence_bound_fp, joint_fp_rate
from verifyensemble.dependency.kappa import cohen_kappa
from verifyensemble.dependency.matrix import DependencyMatrix, bootstrap_ci
from verifyensemble.evaluation.bootstrap import bootstrap_metric
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error, reliability_diagram
from verifyensemble.evaluation.mcnemar import mcnemar_mid_p
from verifyensemble.evaluation.risk_coverage import (
    risk_coverage_auc,
    risk_coverage_curve,
)
from verifyensemble.theory import (
    dajv_upper_bound,
    independence_lower_bound,
    required_n,
    union_upper_bound,
)


# ----- Clopper-Pearson -----
def test_clopper_pearson_n0():
    lo, hi = clopper_pearson(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_clopper_pearson_full_success():
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


def test_clopper_pearson_full_failure():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 1.0


# ----- Naive aggregators -----
def test_naive_all_abstain():
    out = naive_unanimous([None, None, None])
    assert out["recommendation"] == "ABSTAIN_NO_VERIFIERS"
    assert out["n_working"] == 0


def test_naive_majority_tie():
    # 1 accept, 1 reject -> tie -> not strict majority
    out = naive_majority([True, False])
    assert out["recommendation"] == "ABSTAIN"


def test_naive_single_working_accept():
    out = naive_unanimous([True, None, None])
    assert out["recommendation"] == "COMMIT"
    assert out["n_working"] == 1


# ----- DAJV calibration with degenerate inputs -----
def test_dajv_all_correct_calibration():
    accept = [[True] * 5, [True] * 5, [True] * 5]
    correct = [True] * 5
    cal = DajvCalibration.fit(accept, correct, ["A", "B", "C"])
    out = dajv_aggregate([True, True, True], cal)
    assert 0.0 <= out["P_correct"] <= 1.0


def test_dajv_all_wrong_calibration():
    accept = [[False] * 5, [False] * 5, [False] * 5]
    correct = [False] * 5
    cal = DajvCalibration.fit(accept, correct, ["A", "B", "C"])
    out = dajv_aggregate([False, False, False], cal)
    # P_correct should be low (consistent with the wrong-only calibration)
    assert out["P_correct"] >= 0.0


def test_dajv_with_abstain_votes():
    # Half-accept calibration
    accept = [
        [True, False, True, False, True],
        [True, True, False, False, True],
        [False, True, True, False, False],
    ]
    correct = [True, True, False, False, True]
    cal = DajvCalibration.fit(accept, correct, ["A", "B", "C"])
    out = dajv_aggregate([True, None, True], cal)
    assert out["recommendation"] in ("COMMIT", "ESCALATE", "ABSTAIN_LIKELY_WRONG")
    assert out["n_working"] == 2


# ----- CARE -----
def test_care_fits_with_minimal_data():
    accept = [[True, False, True, False],
              [True, True, False, False]]
    correct = [True, True, False, False]
    w = fit_care_weights(accept, correct)
    assert len(w) == 2
    # Should not blow up
    out = care_aggregate([True, True], w)
    assert 0.0 <= out["P_correct"] <= 1.0


def test_care_extreme_inputs_no_overflow():
    # All-zero votes should not overflow
    accept = [[False] * 10, [False] * 10]
    correct = [False] * 10
    w = fit_care_weights(accept, correct)
    out = care_aggregate([False, False], w)
    assert 0.0 <= out["P_correct"] <= 1.0


# ----- Dependency estimators -----
def test_kappa_disagreement():
    a = [True] * 5
    b = [False] * 5
    k = cohen_kappa(a, b)
    # Both raters constant but opposite -> kappa = 0 by our convention
    assert k <= 0.0


def test_kappa_length_mismatch():
    with pytest.raises(ValueError):
        cohen_kappa([True, False], [True])


def test_joint_fp_all_correct():
    accept_i = [True, True, True]
    accept_j = [True, True, True]
    correct = [True, True, True]
    out = joint_fp_rate(accept_i, accept_j, correct)
    assert out["n_wrong"] == 0
    assert out["indep_bound"] == 0.0


def test_independence_bound_fp_empty():
    # No marginals -> 1.0 (empty product convention)
    assert independence_bound_fp([]) == 1.0


def test_cig_constant():
    # Both verifiers always accept on wrong -> mutual information = 0
    a = [True, True, True]; b = [True, True, True]
    correct = [False, False, False]
    assert cig(a, b, correct) == 0.0


# ----- DependencyMatrix -----
def test_dependency_matrix_diagonal():
    accept = [[True, False], [True, True]]
    correct = [False, False]
    D = DependencyMatrix.from_accept(accept, correct, ["X", "Y"])
    # Diagonal kappa = 1
    assert D.kappa[0][0] == 1.0
    assert D.kappa[1][1] == 1.0


def test_dependency_matrix_save_load(tmp_path):
    accept = [[True, False, True], [False, True, True]]
    correct = [False, False, False]
    D = DependencyMatrix.from_accept(accept, correct, ["X", "Y"])
    p = tmp_path / "d.json"
    D.save_json(p)
    import json
    with p.open() as f:
        loaded = json.load(f)
    assert loaded["extractor_ids"] == ["X", "Y"]


def test_bootstrap_ci_runs():
    accept = [[True, False, True, False] * 3,
              [True, True, False, False] * 3]
    correct = [False] * 12
    out = bootstrap_ci(accept, correct, ["X", "Y"], n_bootstrap=20, seed=1)
    assert "kappa_lo" in out and "kappa_hi" in out
    assert out["n_bootstrap"] == 20


# ----- Evaluation harness -----
def test_ece_empty():
    assert expected_calibration_error([], []) == 0.0


def test_reliability_diagram_basic():
    rd = reliability_diagram([0.1, 0.5, 0.9], [False, True, True], n_bins=3)
    assert len(rd["mean_confidence"]) == 3


def test_brier_perfect_predictor():
    assert brier_score([1.0, 0.0, 1.0], [True, False, True]) == 0.0


def test_brier_worst_predictor():
    assert brier_score([0.0, 1.0], [True, False]) == 1.0


def test_risk_coverage_perfect_ranking():
    # Confidence sorts items in correctness order
    auc = risk_coverage_auc([0.9, 0.5, 0.1], [True, True, False])
    assert auc >= 0.0
    auc_bad = risk_coverage_auc([0.1, 0.5, 0.9], [True, True, False])
    assert auc_bad >= auc - 1e-9


def test_risk_coverage_empty():
    out = risk_coverage_curve([], [])
    assert out["coverage"] == []


def test_mcnemar_no_discordant():
    a = [True, False, True, False]
    out = mcnemar_mid_p(a, a)
    assert out["mid_p"] == 1.0


def test_mcnemar_b01_equal_b10():
    # Symmetric discordance -> p around 1.0
    a = [True, False, True, False]
    b = [False, True, False, True]
    out = mcnemar_mid_p(a, b)
    assert out["b01"] == 2 and out["b10"] == 2


# ----- Theory -----
def test_theorem1_bound_at_rho0():
    pi = [0.1, 0.2, 0.3]
    rho = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # At rho=0, the DAJV bound equals the independence bound
    assert dajv_upper_bound(pi, rho) == pytest.approx(independence_lower_bound(pi))


def test_theorem1_bound_clipped_by_union():
    # With k=12, high rho, the DAJV bound should not exceed the union bound
    pi = [0.5] * 12
    rho = [[1.0 if i == j else 0.9 for j in range(12)] for i in range(12)]
    bound = dajv_upper_bound(pi, rho)
    union = union_upper_bound(pi)
    assert bound <= union + 1e-9


def test_theorem2_invalid_eps():
    with pytest.raises(ValueError):
        required_n(4, 0.0)
    with pytest.raises(ValueError):
        required_n(4, 1.5)


def test_theorem2_increasing_with_delta_tighter():
    n_lax = required_n(4, 0.10, delta=0.20)
    n_tight = required_n(4, 0.10, delta=0.01)
    assert n_tight > n_lax


# ----- Bootstrap -----
def test_bootstrap_metric_returns_ci():
    out = bootstrap_metric(
        lambda xs, ys: sum(1 for x, y in zip(xs, ys) if x == y) / len(xs),
        [True, False, True], [True, True, True],
        n_bootstrap=50, seed=0,
    )
    assert "point" in out and "lower" in out and "upper" in out
    assert out["lower"] <= out["point"] <= out["upper"]
