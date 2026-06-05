"""Tests for the VERGE-proxy aggregation rule."""
from __future__ import annotations

import pytest

from verifyensemble.aggregate.verge_proxy import verge_proxy_aggregate


def _classifications_all(c: str, k: int) -> list[str]:
    return [c] * k


def test_commit_when_all_pass():
    out = verge_proxy_aggregate(
        votes=[True, True, True, True],
        classifications=_classifications_all("working", 4),
        candidate_verdicts=[True, True, True, True],
        min_agree=3,
    )
    assert out["recommendation"] == "COMMIT"
    assert out["P_correct"] == 0.97
    assert out["stage"] == "formal_pass"


def test_commit_mcs_one_drop():
    """Consensus passes (4 working+accept), but exec disagrees on one.
    n_exec_true = 3 < min_agree=4 -> MCS one-drop path."""
    out = verge_proxy_aggregate(
        votes=[True, True, True, True],
        classifications=["working"] * 4,
        # 1 of 4 exec verdicts disagrees -> MCS path
        candidate_verdicts=[True, True, True, False],
        min_agree=4,
    )
    assert out["recommendation"] == "COMMIT_MCS"
    assert out["P_correct"] == 0.80
    assert out["stage"] == "mcs_one_drop"


def test_abstain_when_no_working():
    out = verge_proxy_aggregate(
        votes=[False, False, False, False],
        classifications=["UNVERIFIABLE"] * 4,
        candidate_verdicts=[None, None, None, None],
        min_agree=3,
    )
    assert out["recommendation"] == "ABSTAIN_NO_VERIFIERS"
    assert out["stage"] == "no_working_verifiers"


def test_abstain_consensus_fail():
    out = verge_proxy_aggregate(
        votes=[True, False, False, False],
        classifications=["working", "working", "wrong_spec", "wrong_spec"],
        candidate_verdicts=[True, False, False, False],
        min_agree=3,
    )
    assert out["recommendation"] == "ABSTAIN"
    assert out["stage"] == "consensus_fail"


def test_abstain_formal_fail():
    """Consensus passes (4 working+accept) but n_exec_true=2 falls
    below min_agree-1=3 with min_agree=4 -> formal_fail."""
    out = verge_proxy_aggregate(
        votes=[True, True, True, True],
        classifications=["working"] * 4,
        candidate_verdicts=[True, True, False, False],
        min_agree=4,
    )
    assert out["recommendation"] == "ABSTAIN"
    assert out["stage"] == "formal_fail"


def test_invalid_length_raises():
    with pytest.raises(ValueError):
        verge_proxy_aggregate(
            votes=[True, True, True],
            classifications=["working"] * 4,
            candidate_verdicts=[True] * 4,
            min_agree=3,
        )


def test_p_correct_field_present():
    out = verge_proxy_aggregate(
        votes=[True] * 4,
        classifications=["working"] * 4,
        candidate_verdicts=[True] * 4,
        min_agree=3,
    )
    assert "P_correct" in out
    assert "n_working" in out
    assert "n_accept" in out
    assert "n_consensus" in out
