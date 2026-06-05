"""Tests for the None-handling fixes flagged by the session-3 review.

Run from dajv/:

    PYTHONPATH=. python3 -m pytest tests/test_script_edge_cases.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(script: Path):
    spec = importlib.util.spec_from_file_location(script.stem, script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_matched_coverage_precision_handles_zero_target():
    """matched_coverage_precision must not crash when target_k <= 0."""
    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here))
    mod = _load(here / "scripts" / "run_prm_matched_coverage.py")
    prec, commits = mod.matched_coverage_precision([0.1, 0.2, 0.3], [True, False, True], 0)
    assert prec is None
    assert commits == [False, False, False]


def test_matched_coverage_precision_handles_empty():
    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here))
    mod = _load(here / "scripts" / "run_prm_matched_coverage.py")
    prec, commits = mod.matched_coverage_precision([], [], 5)
    assert prec is None
    assert commits == []
