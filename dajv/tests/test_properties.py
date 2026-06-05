"""Property-based tests via hypothesis.

Each test asserts an invariant that must hold for ALL inputs in a
parameterized class, not just hand-picked examples.

Run from dajv/:

    PYTHONPATH=. python3 -m pytest tests/test_properties.py -q
"""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_unanimous
from verifyensemble.aggregate.posterior import clopper_pearson
from verifyensemble.dependency.kappa import cohen_kappa
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.theory import (
    dajv_upper_bound,
    independence_lower_bound,
    required_n,
    union_upper_bound,
)


# ---------------------------------------------------------------------------
# Clopper-Pearson invariants
# ---------------------------------------------------------------------------
@given(
    n=st.integers(min_value=1, max_value=1000),
    data=st.data(),
)
def test_clopper_pearson_in_unit_interval(n, data):
    k = data.draw(st.integers(min_value=0, max_value=n))
    lo, hi = clopper_pearson(k, n)
    assert 0.0 <= lo <= hi <= 1.0
    # Point estimate lies inside the interval (modulo CI width)
    p = k / n
    assert lo - 1e-9 <= p <= hi + 1e-9


@given(
    n=st.integers(min_value=1, max_value=200),
    data=st.data(),
)
def test_clopper_pearson_interval_shrinks_with_n(n, data):
    """At fixed proportion, interval width must be monotone-non-increasing in n."""
    k_small = data.draw(st.integers(min_value=0, max_value=n))
    if k_small / n < 0.5:
        # only test for proportions away from 0/1 to avoid degenerate intervals
        return
    lo_small, hi_small = clopper_pearson(k_small, n)
    # double n while preserving the same proportion
    n_big = 2 * n
    k_big = 2 * k_small
    lo_big, hi_big = clopper_pearson(k_big, n_big)
    # Wider intervals at smaller n
    assert (hi_big - lo_big) <= (hi_small - lo_small) + 1e-6


# ---------------------------------------------------------------------------
# Cohen's kappa is symmetric and bounded
# ---------------------------------------------------------------------------
@given(
    a=st.lists(st.booleans(), min_size=2, max_size=100),
    b=st.lists(st.booleans(), min_size=2, max_size=100),
)
def test_cohen_kappa_bounded_and_symmetric(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    kab = cohen_kappa(a, b)
    kba = cohen_kappa(b, a)
    assert -1.0 - 1e-9 <= kab <= 1.0 + 1e-9
    assert abs(kab - kba) < 1e-9


# ---------------------------------------------------------------------------
# Theorem 1 bound never below independence, never above union
# ---------------------------------------------------------------------------
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
@given(
    k=st.integers(min_value=2, max_value=8),
    data=st.data(),
)
def test_theorem1_sandwich(k, data):
    pi = data.draw(
        st.lists(st.floats(min_value=0.01, max_value=0.5),
                 min_size=k, max_size=k)
    )
    # Symmetric correlation matrix with diagonal 1, off-diag in [-0.5, 0.5]
    rho = [[0.0] * k for _ in range(k)]
    for i in range(k):
        rho[i][i] = 1.0
    for i in range(k):
        for j in range(i + 1, k):
            r = data.draw(st.floats(min_value=-0.5, max_value=0.5))
            rho[i][j] = r
            rho[j][i] = r

    indep = independence_lower_bound(pi)
    dajv = dajv_upper_bound(pi, rho)
    union = union_upper_bound(pi)
    assert dajv >= indep - 1e-9
    assert dajv <= union + 1e-9


# ---------------------------------------------------------------------------
# Theorem 2 required_n is monotone in 1/eps and 1/delta
# ---------------------------------------------------------------------------
@given(
    k=st.integers(min_value=2, max_value=30),
    eps=st.floats(min_value=0.01, max_value=0.5),
)
def test_theorem2_monotone_in_delta(k, eps):
    n_lax = required_n(k, eps, delta=0.50)
    n_tight = required_n(k, eps, delta=0.01)
    assert n_tight >= n_lax


# ---------------------------------------------------------------------------
# Brier score is bounded in [0, 1] given binary outcomes
# ---------------------------------------------------------------------------
@given(
    confs=st.lists(st.floats(min_value=0.0, max_value=1.0),
                   min_size=1, max_size=100),
    data=st.data(),
)
def test_brier_score_bounded(confs, data):
    correct = data.draw(
        st.lists(st.booleans(), min_size=len(confs), max_size=len(confs))
    )
    b = brier_score(confs, correct)
    assert 0.0 <= b <= 1.0


# ---------------------------------------------------------------------------
# Naive unanimous is permutation-invariant
# ---------------------------------------------------------------------------
@given(
    votes=st.lists(
        st.one_of(st.booleans(), st.just(None)),
        min_size=1, max_size=10,
    )
)
def test_naive_unanimous_permutation_invariant(votes):
    import random
    a = naive_unanimous(votes)
    perm = votes[:]
    random.Random(0).shuffle(perm)
    b = naive_unanimous(perm)
    assert a["recommendation"] == b["recommendation"]
    assert a["n_working"] == b["n_working"]
    assert a["n_accept"] == b["n_accept"]


# ---------------------------------------------------------------------------
# DAJV: posterior P_correct is in [0, 1]
# ---------------------------------------------------------------------------
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
@given(
    k=st.integers(min_value=2, max_value=5),
    n=st.integers(min_value=10, max_value=50),
    data=st.data(),
)
def test_dajv_posterior_in_unit(k, n, data):
    accept_cal = [
        data.draw(st.lists(st.booleans(), min_size=n, max_size=n))
        for _ in range(k)
    ]
    correct = data.draw(st.lists(st.booleans(), min_size=n, max_size=n))
    # Force at least one correct and one wrong; else calibration degenerates
    if all(correct) or not any(correct):
        return
    cal = DajvCalibration.fit(accept_cal, correct,
                              [f"E{i}" for i in range(k)])
    votes = data.draw(st.lists(st.booleans(), min_size=k, max_size=k))
    out = dajv_aggregate(votes, cal)
    if out["P_correct"] is not None:
        assert 0.0 <= out["P_correct"] <= 1.0


# ---------------------------------------------------------------------------
# ECE is in [0, 1]
# ---------------------------------------------------------------------------
@given(
    confs=st.lists(st.floats(min_value=0.0, max_value=1.0),
                   min_size=2, max_size=50),
    data=st.data(),
)
def test_ece_bounded(confs, data):
    correct = data.draw(
        st.lists(st.booleans(), min_size=len(confs), max_size=len(confs))
    )
    e = expected_calibration_error(confs, correct, n_bins=5)
    assert 0.0 - 1e-9 <= e <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Independence-bound monotonicity
# ---------------------------------------------------------------------------
@given(
    pi=st.lists(st.floats(min_value=0.01, max_value=0.5),
                min_size=2, max_size=10),
)
def test_independence_bound_monotone_in_k(pi):
    """Adding a verifier with pi_k > 0 should reduce the joint product."""
    base = independence_lower_bound(pi)
    extended = independence_lower_bound(pi + [0.5])
    assert extended <= base * 0.5 + 1e-9


# ---------------------------------------------------------------------------
# Union bound monotonicity
# ---------------------------------------------------------------------------
@given(
    pi=st.lists(st.floats(min_value=0.01, max_value=0.5),
                min_size=2, max_size=10),
)
def test_union_bound_monotone_in_k(pi):
    """Adding a verifier with pi_k > 0 cannot decrease the union bound."""
    base = union_upper_bound(pi)
    extended = union_upper_bound(pi + [0.1])
    assert extended >= base - 1e-9
