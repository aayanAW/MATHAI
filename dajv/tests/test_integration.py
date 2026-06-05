"""End-to-end integration tests using cached extractor data.

These tests verify that the full DAJV pipeline (cache load → align →
fit → aggregate) produces the headline-result numbers cited in the
paper, within tolerance.

If a test fails, either the cache moved, the algorithm changed, or
the paper number is stale; one of those needs reconciling.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_unanimous
from verifyensemble.utils.io import align_extractor_caches

CACHES = {
    "E05": "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06": "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07": "../MATHAI/results/exp47_claude_extractor.json",
    "E09": "../MATHAI/results/exp48_qwen3coder_extractor.json",
}

DAJV_ROOT = Path(__file__).resolve().parent.parent


def _have_caches() -> bool:
    return all((DAJV_ROOT / p).exists() for p in CACHES.values())


@pytest.mark.skipif(not _have_caches(),
                    reason="cached extractor caches not present")
def test_paper_headline_math175_dajv_precision():
    """Reproduce the headline result: DAJV 32/33 = 0.970 precision on
    math175 with seed=42 / 70-30 split."""
    paths = {k: DAJV_ROOT / v for k, v in CACHES.items()}
    aligned = align_extractor_caches(paths, bench="math175")
    n = len(aligned["problem_ids"])
    rng = random.Random(42)
    idx = list(range(n))
    rng.shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    k = len(aligned["extractor_ids"])

    accept = aligned["accept"]
    correct = aligned["solver_correct"]
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    correct_cal = [correct[j] for j in cal_idx]
    cal = DajvCalibration.fit(accept_cal, correct_cal, aligned["extractor_ids"])

    n_commit = n_commit_correct = 0
    for j in test_idx:
        votes = [bool(accept[i][j]) for i in range(k)]
        out = dajv_aggregate(votes, cal)
        if out.get("recommendation") == "COMMIT":
            n_commit += 1
            if correct[j]:
                n_commit_correct += 1

    assert n_commit == 33, f"expected 33 DAJV commits, got {n_commit}"
    assert n_commit_correct == 32, \
        f"expected 32 correct of 33 DAJV commits, got {n_commit_correct}"
    precision = n_commit_correct / n_commit
    assert abs(precision - 32/33) < 1e-6


@pytest.mark.skipif(not _have_caches(),
                    reason="cached extractor caches not present")
def test_paper_headline_naive_baseline_math175():
    """Naive unanimous baseline matches paper: 25/26 = 0.962 precision."""
    paths = {k: DAJV_ROOT / v for k, v in CACHES.items()}
    aligned = align_extractor_caches(paths, bench="math175")
    n = len(aligned["problem_ids"])
    rng = random.Random(42)
    idx = list(range(n))
    rng.shuffle(idx)
    cal_n = int(0.7 * n)
    test_idx = idx[cal_n:]
    k = len(aligned["extractor_ids"])
    accept = aligned["accept"]
    correct = aligned["solver_correct"]
    n_naive = n_naive_correct = 0
    for j in test_idx:
        votes = [bool(accept[i][j]) for i in range(k)]
        out = naive_unanimous(votes)
        if out.get("recommendation") == "COMMIT":
            n_naive += 1
            if correct[j]:
                n_naive_correct += 1
    assert n_naive == 26, f"expected 26 naive commits, got {n_naive}"
    assert n_naive_correct == 25, \
        f"expected 25 correct of 26 naive commits, got {n_naive_correct}"
