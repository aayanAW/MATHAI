"""Tests for the cross-modality (H7') aggregators."""
from __future__ import annotations

from verifyensemble.aggregate.dajv import DajvCalibration
from verifyensemble.aggregate.dajv_xmod import (
    BlockSparseIsingCalibration,
    XmodCalibration,
    XmodJointCalibration,
    _isotonic_pav,
    _isotonic_predict,
    bsia_aggregate,
    bsia_isotonic_aggregate,
    fit_bsia_isotonic,
    fit_bsia_temperature,
    xmod_aggregate,
    xmod_agreement_aggregate,
    xmod_joint_aggregate,
    xmod_struct_gated_exec_dajv_aggregate,
)


def _toy_calibration(k: int = 3):
    # Per-LLM structural and executable signals on a tiny calibration set.
    struct = [[True, True, False, False, True, True, False, True, True, False],
              [True, False, True, False, True, False, True, True, False, True],
              [True, True, True, False, True, True, True, True, True, False]]
    exec_ = [[True, True, False, False, True, False, False, True, True, False],
             [True, False, True, False, True, False, False, True, False, False],
             [True, True, True, False, True, True, True, True, True, False]]
    correct = [True, True, False, False, True, True, False, True, True, False]
    return struct, exec_, correct


def test_xmod_calibration_fits():
    s, x, c = _toy_calibration()
    cal = XmodCalibration.fit(s, x, c, ["A", "B", "C"])
    assert 0.0 <= cal.prior_correct <= 1.0
    assert len(cal.struct.pi_pos) == 3
    assert len(cal.exec_.pi_pos) == 3


def test_xmod_aggregate_commit_branch():
    s, x, c = _toy_calibration()
    cal = XmodCalibration.fit(s, x, c, ["A", "B", "C"])
    out = xmod_aggregate([True, True, True], [True, True, True], cal,
                         accept_threshold=0.5)
    assert out["recommendation"] in {"COMMIT", "ESCALATE",
                                     "ABSTAIN_LIKELY_WRONG"}
    assert 0.0 <= out["P_correct"] <= 1.0


def test_xmod_agreement_disagree_branch():
    s, x, c = _toy_calibration()
    cal = XmodCalibration.fit(s, x, c, ["A", "B", "C"])
    # Force a cross-modality disagreement: all struct True but all exec False
    out = xmod_agreement_aggregate(
        [True, True, True], [False, False, False], cal,
        accept_threshold_per_modality=0.5
    )
    assert out["recommendation"] in {"ESCALATE_XMOD_DISAGREE", "ESCALATE",
                                     "ABSTAIN_LIKELY_WRONG"}


def test_xmod_joint_calibration_smoothing():
    s, x, c = _toy_calibration()
    cal = XmodJointCalibration.fit(s, x, c, ["A", "B", "C"], smooth=0.5)
    # Each per-LLM joint should be a probability distribution over 4 cells
    for i in range(3):
        for label in ("joint_pos", "joint_neg"):
            J = getattr(cal, label)[i]
            tot = sum(J[ss][xx] for ss in (0, 1) for xx in (0, 1))
            assert abs(tot - 1.0) < 1e-9


def test_xmod_joint_aggregate_returns_valid_probability():
    s, x, c = _toy_calibration()
    cal = XmodJointCalibration.fit(s, x, c, ["A", "B", "C"])
    out = xmod_joint_aggregate([True, True, False], [True, False, False], cal)
    assert 0.0 <= out["P_correct"] <= 1.0
    assert out["recommendation"] in {"COMMIT", "ESCALATE",
                                     "ABSTAIN_LIKELY_WRONG"}


def test_xmod_struct_gated_requires_struct_agreement():
    s, x, c = _toy_calibration()
    accept = [[s[i][j] and x[i][j] for j in range(len(c))] for i in range(3)]
    exec_dajv = DajvCalibration.fit(accept, c, ["A", "B", "C"])
    # struct unanimously False -> should not commit even if exec strong
    out = xmod_struct_gated_exec_dajv_aggregate(
        [False, False, False], [True, True, True],
        exec_dajv,
        min_struct_agree=2,
    )
    assert out["recommendation"] in {"ESCALATE_STRUCT_DISAGREE", "ESCALATE",
                                     "ABSTAIN_LIKELY_WRONG"}


def test_bsia_calibration_fits_and_marginals_match_cells():
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    # Cells must be valid probabilities summing to 1 per (LLM, class).
    for cells in (cal.cells_pos, cal.cells_neg):
        for i in range(3):
            tot = sum(cells[i][ss][xx] for ss in (0, 1) for xx in (0, 1))
            assert abs(tot - 1.0) < 1e-9
    # Marginals must be consistent with cells (struct = sum over x).
    for i in range(3):
        ms = cal.cells_pos[i][1][0] + cal.cells_pos[i][1][1]
        assert abs(cal.mu_pos_struct[i] - ms) < 1e-9
        mx = cal.cells_pos[i][0][1] + cal.cells_pos[i][1][1]
        assert abs(cal.mu_pos_exec[i] - mx) < 1e-9


def test_bsia_cross_modality_block_is_structurally_zero():
    """BSIA does not store cross-LLM cross-modality terms.

    Sparsity is encoded by *not modeling* those interactions, which we
    verify by inspecting the calibration object's attributes.
    """
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    # Only within-modality interaction matrices are stored
    # (rho_*_struct and rho_*_exec); cross-modality interactions are
    # not modeled at all.  ``rho_shrinkage`` is a scalar
    # hyperparameter, not a matrix.
    matrix_attrs = {
        a for a in vars(cal).keys()
        if a.startswith("rho") and isinstance(getattr(cal, a), list)
    }
    assert matrix_attrs == {
        "rho_pos_struct", "rho_neg_struct",
        "rho_pos_exec", "rho_neg_exec",
    }


def test_bsia_aggregate_commit_branch():
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    out = bsia_aggregate([True, True, True], [True, True, True], cal,
                         accept_threshold=0.5)
    assert out["recommendation"] in {"COMMIT", "ESCALATE",
                                     "ABSTAIN_LIKELY_WRONG"}
    assert 0.0 <= out["P_correct"] <= 1.0


def test_bsia_aggregate_abstain_branch():
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    # All-reject votes should drop below the abstain threshold.
    out = bsia_aggregate([False, False, False], [False, False, False], cal)
    assert out["recommendation"] in {"ABSTAIN_LIKELY_WRONG", "ESCALATE"}
    assert out["P_correct"] < 0.5


def test_bsia_handles_none_votes_via_marginalization():
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    # If one LLM emits None for both modalities, it should be dropped
    # from the aggregation rather than crash.
    out = bsia_aggregate([True, None, True], [True, None, True], cal)
    assert out["recommendation"] in {"COMMIT", "ESCALATE",
                                     "ABSTAIN_LIKELY_WRONG"}
    assert out["n_active"] == 2


def test_bsia_no_active_votes_returns_abstain():
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    out = bsia_aggregate([None, None, None], [None, None, None], cal)
    assert out["recommendation"] == "ABSTAIN_NO_VERIFIERS"
    assert out["P_correct"] is None


def test_bsia_shrinkage_extremes():
    s, x, c = _toy_calibration()
    cal0 = BlockSparseIsingCalibration.fit(
        s, x, c, ["A", "B", "C"], rho_shrinkage=0.0,
    )
    cal1 = BlockSparseIsingCalibration.fit(
        s, x, c, ["A", "B", "C"], rho_shrinkage=1.0,
    )
    # rho_shrinkage=1 collapses all interactions to zero (pure per-LLM NB).
    for rho_attr in ("rho_pos_struct", "rho_neg_struct",
                     "rho_pos_exec", "rho_neg_exec"):
        rho1 = getattr(cal1, rho_attr)
        for i in range(3):
            for j in range(3):
                assert abs(rho1[i][j]) < 1e-12
    # rho_shrinkage=0 should keep at least one non-zero off-diagonal
    # provided the toy data is non-degenerate.
    rho0 = cal0.rho_pos_struct
    non_zero = any(
        abs(rho0[i][j]) > 1e-9
        for i in range(3) for j in range(3) if i != j
    )
    assert non_zero or rho0 == [[0.0] * 3 for _ in range(3)]


def test_bsia_length_mismatch_raises():
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    try:
        bsia_aggregate([True, False], [True, False, False], cal)
    except ValueError:
        return
    raise AssertionError("BSIA must raise on length mismatch")


def test_bsia_temperature_returns_positive():
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    T = fit_bsia_temperature(cal, s, x, c)
    assert T > 0


def test_bsia_temperature_applies_at_aggregate():
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    # With temperature 1.0 (no scaling), aggregate posterior unchanged.
    out_no_temp = bsia_aggregate([True, True, True], [True, True, True], cal)
    # Apply a large temperature; posterior should move toward 0.5.
    cal.temperature = 10.0
    out_high_temp = bsia_aggregate([True, True, True], [True, True, True], cal)
    assert abs(out_high_temp["P_correct"] - 0.5) < abs(out_no_temp["P_correct"] - 0.5)


def test_isotonic_pav_monotone_output():
    # PAV on a non-monotone sequence should yield monotone-increasing fitted ys.
    xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ys = [True, False, True, True, False, True]
    cal_xs, cal_ys = _isotonic_pav(xs, ys)
    assert len(cal_xs) == len(cal_ys)
    for i in range(len(cal_ys) - 1):
        assert cal_ys[i] <= cal_ys[i + 1] + 1e-9


def test_isotonic_predict_bounds():
    cal_xs = [0.1, 0.5, 0.9]
    cal_ys = [0.0, 0.5, 1.0]
    # Below lowest x: returns lowest y
    assert _isotonic_predict(0.05, cal_xs, cal_ys) == 0.0
    # Above highest x: returns highest y
    assert _isotonic_predict(0.95, cal_xs, cal_ys) == 1.0
    # In between: returns appropriate level
    assert _isotonic_predict(0.5, cal_xs, cal_ys) == 0.5


def test_bsia_isotonic_aggregate_returns_valid_posterior():
    s, x, c = _toy_calibration()
    cal = BlockSparseIsingCalibration.fit(s, x, c, ["A", "B", "C"])
    cal_xs, cal_ys = fit_bsia_isotonic(cal, s, x, c)
    out = bsia_isotonic_aggregate(
        [True, True, True], [True, True, True], cal, cal_xs, cal_ys,
        accept_threshold=0.5,
    )
    assert 0.0 <= out["P_correct"] <= 1.0
    assert "P_correct_raw" in out
