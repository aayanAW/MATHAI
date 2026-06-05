"""Tests for input-validation fixes from the code-review pass.

Run from dajv/:

    PYTHONPATH=. python3 -m pytest tests/test_validation.py -q
"""
from __future__ import annotations

import pytest

from verifyensemble.aggregate.dajv import DajvCalibration
from verifyensemble.dependency.matrix import DependencyMatrix


def test_dajv_fit_dim2_mismatch_raises_value_error():
    accept_cal = [[True, False, True], [True, True]]   # row 1 has length 2, row 0 has 3
    correct = [True, False, True]                       # length 3
    with pytest.raises(ValueError, match="length"):
        DajvCalibration.fit(accept_cal, correct, ["A", "B"])


def test_dajv_fit_k_mismatch_raises_value_error():
    accept_cal = [[True, False], [True, False]]
    correct = [True, False]
    with pytest.raises(ValueError, match="k mismatch"):
        DajvCalibration.fit(accept_cal, correct, ["A", "B", "C"])


def test_dependency_from_accept_dim2_mismatch_raises_value_error():
    accept = [[True, False, True], [True, True]]
    correct = [False, False, False]
    with pytest.raises(ValueError, match="length"):
        DependencyMatrix.from_accept(accept, correct, ["A", "B"])


def test_dependency_from_accept_k_mismatch_raises_value_error():
    accept = [[True, False], [True, False]]
    correct = [False, False]
    with pytest.raises(ValueError, match="rows"):
        DependencyMatrix.from_accept(accept, correct, ["A", "B", "C"])
