"""Loaders for legacy X-SGRV cache files.

The X-SGRV cache files live in ``MATHAI/results/exp*_*_extractor.json``.
Each cache has the structure:

    {
        "extractor": "<model id>",
        "results": [
            {
                "id": "<problem id>",
                "outcome": "<verifier outcome label>",
                "classification": "working" | "wrong_spec" | "trivial_or_broken" | "UNVERIFIABLE" | ...,
                "script": "<verifier source>",
                "gold": "<gold answer>",
                "candidate": "<solver's candidate answer>",
                "solver_correct": bool,
                "gold_verdict": True | False | None,
                "candidate_verdict": True | False | None,
                "adv_verdicts": [...]  # optional
            },
            ...
        ]
    }

This module loads those caches and aligns them across extractors so that
each problem id is associated with one binary acceptance per extractor.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional  # noqa: F401


def load_xsgrv_cache(path: str | Path) -> dict[str, Any]:
    """Load a single legacy X-SGRV extractor cache."""
    p = Path(path)
    with p.open() as f:
        d = json.load(f)
    if "results" not in d:
        raise ValueError(f"{path} does not look like an X-SGRV cache "
                         f"(no 'results' key; top-level keys: {list(d.keys())})")
    return d


def align_extractor_caches(
    cache_paths: dict[str, str | Path],
    bench: Optional[str] = None,
    require_solver_correct: Optional[bool] = None,
) -> dict[str, Any]:
    """Align multiple extractor caches on shared problem ids.

    Args:
        cache_paths: mapping ``extractor_id -> cache_file_path``.
        bench: if provided, restrict to records with ``record['bench'] == bench``.
            Caches without a 'bench' field are kept unrestricted.
        require_solver_correct: if True, restrict to problems where the
            solver got the gold answer; if False, restrict to ones it
            got wrong; if None (default), keep all.

    Returns:
        dict with:
            'extractor_ids':    [E1, E2, ...]
            'problem_ids':      [pid1, pid2, ...]
            'gold':             [gold1, gold2, ...]
            'candidate':        [cand1, cand2, ...]
            'solver_correct':   [bool, ...]
            'classification':   array shape (n_extractors, n_problems) of strings
            'candidate_verdict':array shape (n_extractors, n_problems) of {True, False, None}
            'gold_verdict':     array shape (n_extractors, n_problems) of {True, False, None}
            'working':          array shape (n_extractors, n_problems) of bool
            'accept':           array shape (n_extractors, n_problems) of bool
                                (True iff classification=='working' AND candidate_verdict==True)
    """
    caches = {eid: load_xsgrv_cache(p) for eid, p in cache_paths.items()}

    # Per-extractor record dicts keyed by problem id
    per_ext: dict[str, dict[str, dict]] = {}
    for eid, c in caches.items():
        per_ext[eid] = {}
        for rec in c["results"]:
            if bench is not None and "bench" in rec and rec["bench"] != bench:
                continue
            per_ext[eid][rec["id"]] = rec

    # Intersect problem ids across extractors
    shared_ids: set[str] = set.intersection(*(set(d.keys()) for d in per_ext.values()))
    # Stable ordering by id
    pids = sorted(shared_ids)

    # Optional solver-correctness filter (uses the FIRST extractor's record)
    if require_solver_correct is not None:
        first_eid = next(iter(per_ext))
        pids = [pid for pid in pids
                if bool(per_ext[first_eid][pid].get("solver_correct")) == require_solver_correct]

    extractor_ids = list(per_ext.keys())
    n_ext = len(extractor_ids)
    n_prob = len(pids)

    classification: list[list[Optional[str]]] = [[None] * n_prob for _ in range(n_ext)]
    candidate_verdict: list[list[Optional[bool]]] = [[None] * n_prob for _ in range(n_ext)]
    gold_verdict: list[list[Optional[bool]]] = [[None] * n_prob for _ in range(n_ext)]
    working: list[list[bool]] = [[False] * n_prob for _ in range(n_ext)]
    accept: list[list[bool]] = [[False] * n_prob for _ in range(n_ext)]

    gold: list[Optional[str]] = [None] * n_prob
    candidate: list[Optional[str]] = [None] * n_prob
    solver_correct: list[Optional[bool]] = [None] * n_prob

    for j, pid in enumerate(pids):
        first_rec = per_ext[extractor_ids[0]][pid]
        gold[j] = first_rec.get("gold")
        candidate[j] = first_rec.get("candidate")
        solver_correct[j] = bool(first_rec.get("solver_correct"))
        for i, eid in enumerate(extractor_ids):
            r = per_ext[eid][pid]
            classification[i][j] = r.get("classification")
            candidate_verdict[i][j] = r.get("candidate_verdict")
            gold_verdict[i][j] = r.get("gold_verdict")
            working[i][j] = (r.get("classification") == "working")
            accept[i][j] = (r.get("classification") == "working"
                            and r.get("candidate_verdict") is True)

    return {
        "extractor_ids": extractor_ids,
        "problem_ids": pids,
        "gold": gold,
        "candidate": candidate,
        "solver_correct": solver_correct,
        "classification": classification,
        "candidate_verdict": candidate_verdict,
        "gold_verdict": gold_verdict,
        "working": working,
        "accept": accept,
    }


def save_artifact(obj: Any, path: str | Path) -> None:
    """Save a Python object to disk (JSON if dict/list/primitive, pickle otherwise)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".json":
        with p.open("w") as f:
            json.dump(obj, f, indent=2, default=str)
    else:
        with p.open("wb") as f:
            pickle.dump(obj, f)


def load_artifact(path: str | Path) -> Any:
    """Load a saved artifact (JSON or pickle, by extension)."""
    p = Path(path)
    if p.suffix == ".json":
        with p.open() as f:
            return json.load(f)
    with p.open("rb") as f:
        return pickle.load(f)
