"""Cross-distribution validation on ProcessBench (small subset).

Runs 4 OpenAI extractors against a small ProcessBench sample (math
split, n=30) using the same DAJV pipeline as the math175 / AIME /
CleanMath cache. Tests whether DAJV calibration transfers to a new
problem distribution.

Output:
  results/exp60_processbench_<extractor>.json

Cost: ~120 API calls (~$2-5).
Time: ~15 min wall.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ENV_PATH = HERE.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from verifyensemble.extractors.api_wrappers import (
    EXTRACTOR_REGISTRY,
    make_extractor_call,
)
from verifyensemble.extractors.parser import extract_verifier
from verifyensemble.sandbox.adversarial import deployment_time_filter
from verifyensemble.sandbox.executor import execute_verifier
from verifyensemble.utils.io import save_artifact

MATHAI_RESULTS = HERE.parent / ".." / "MATHAI" / "results"

# 4 OpenAI extractors only (Together AI blocked, Anthropic key absent)
TARGET_EXTRACTORS = [
    "E06_gpt_5_mini",
    "E08S_gpt_4o",
    "E04S_gpt_4_1",
    "E10S_gpt_5",
]


def _extract_final_answer(last_step: str) -> str:
    """Heuristic: extract the final numeric/boxed answer from a step."""
    import re
    # \\boxed{X}
    m = re.search(r"\\boxed\{([^}]+)\}", last_step)
    if m:
        return m.group(1).strip()
    # "answer is X" / "= X"
    m = re.search(r"(?:answer is|equals|=)\s*\$?([+-]?\d+(?:\.\d+)?(?:/\d+)?)", last_step)
    if m:
        return m.group(1).strip()
    # Last integer in the last step
    m = re.findall(r"[+-]?\d+(?:\.\d+)?(?:/\d+)?", last_step)
    if m:
        return m[-1].strip()
    return ""


def load_processbench_math(n_max: int = 30, stratified: bool = True) -> list[dict]:
    """Load the math split of ProcessBench, return list of records.

    Gold-free mode: we extract candidate (solver's final answer) from
    the last reasoning step. The ground truth label is
    ``final_answer_correct``.

    When ``stratified=True`` (default), returns a 50/50 balanced
    sample of correct/wrong final-answer problems up to ``n_max``.
    """
    from datasets import load_dataset
    d = load_dataset("Qwen/ProcessBench", split="math")
    correct: list[dict] = []
    wrong: list[dict] = []
    for i, row in enumerate(d):
        steps = row.get("steps") or []
        if not steps:
            continue
        last_step = steps[-1] if isinstance(steps[-1], str) else str(steps[-1])
        candidate = _extract_final_answer(last_step)
        if not candidate:
            continue
        is_correct = bool(row.get("final_answer_correct"))
        rec = {
            "id": f"processbench_math_{i:03d}",
            "problem": row.get("problem", ""),
            "gold": "__GOLDFREE__",
            "candidate": candidate,
            "label": int(row.get("label", -1)),
            "is_correct": is_correct,
        }
        if is_correct:
            correct.append(rec)
        else:
            wrong.append(rec)
    if not stratified:
        return (correct + wrong)[:n_max]
    half = n_max // 2
    return correct[:half] + wrong[:half]


def classify_record(script: str | None, gold: str, candidate: str) -> dict:
    """Gold-free classification: 'working' iff candidate runs without
    error AND deployment-time adversarial filter is clean."""
    if not script:
        return {"classification": "UNVERIFIABLE", "script": None,
                "gold_verdict": None, "candidate_verdict": None,
                "adv_verdicts": [], "adv_fp_count": 0,
                "outcome": "abstained_or_no_script"}
    c = execute_verifier(script, candidate, timeout=8.0)
    broken, adv_results = deployment_time_filter(script, candidate, timeout=8.0)
    adv_verdicts = [r["verdict"] for r in adv_results]
    adv_fp = sum(1 for v in adv_verdicts if v is True)

    # Gold-free classification:
    # - candidate verdict None or error -> exec_error
    # - broken (any adv probe accepted) -> trivial_or_broken
    # - candidate verdict True and adv clean -> working (accepted)
    # - candidate verdict False and adv clean -> working (rejected)
    if c.verdict is None:
        classification = "exec_error"
    elif broken:
        classification = "trivial_or_broken"
    else:
        classification = "working"

    gold_verdict = None  # not measured in gold-free mode
    if gold and gold != "__GOLDFREE__":
        g = execute_verifier(script, gold, timeout=8.0)
        gold_verdict = g.verdict

    return {
        "classification": classification,
        "script": script,
        "gold_verdict": gold_verdict,
        "candidate_verdict": c.verdict,
        "adv_verdicts": adv_verdicts,
        "adv_broken": broken,
        "adv_fp_count": adv_fp,
        "outcome": "verifier_produced",
    }


def run_one_extractor(eid: str, problems: list[dict],
                      out_path: Path) -> None:
    done: dict[str, dict] = {}
    if out_path.exists():
        try:
            for r in json.load(open(out_path)).get("results", []):
                done[r["id"]] = r
        except Exception:
            pass
    call = make_extractor_call(eid)
    cfg = EXTRACTOR_REGISTRY[eid]
    print(f"\n[{eid}] {cfg['model']}  start (resume={len(done)})")
    t0 = time.time()
    for i, p in enumerate(problems):
        if p["id"] in done:
            continue
        try:
            ex = extract_verifier(p["problem"], call)
        except Exception as e:
            ex = type("Ex", (), {"script": None, "unverifiable": False,
                                  "raw_response": "", "error": str(e)})

        if ex.unverifiable:
            frag = {"classification": "UNVERIFIABLE", "script": None,
                    "gold_verdict": None, "candidate_verdict": None,
                    "adv_verdicts": [], "adv_fp_count": 0,
                    "outcome": "explicit_unverifiable"}
        elif ex.script is None:
            frag = {"classification": "parse_error", "script": None,
                    "gold_verdict": None, "candidate_verdict": None,
                    "adv_verdicts": [], "adv_fp_count": 0,
                    "outcome": "parse_error"}
        else:
            frag = classify_record(ex.script, p["gold"], p["candidate"])

        done[p["id"]] = {
            "id": p["id"],
            "bench": "processbench_math",
            "gold": p["gold"],
            "candidate": p["candidate"],
            "solver_correct": p["is_correct"],
            **frag,
            "elapsed": time.time() - t0,
        }
        if (i + 1) % 5 == 0:
            save_artifact({"extractor": eid, "model": cfg["model"],
                           "results": list(done.values())}, out_path)
            print(f"  [{eid}] {len(done)}/{len(problems)}")
    save_artifact({"extractor": eid, "model": cfg["model"],
                   "results": list(done.values())}, out_path)
    print(f"  [{eid}] done -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                    help="number of ProcessBench math problems to use")
    ap.add_argument("--extractors", nargs="*", default=TARGET_EXTRACTORS)
    args = ap.parse_args()

    problems = load_processbench_math(n_max=args.n)
    print(f"loaded {len(problems)} processbench problems")

    for eid in args.extractors:
        out = MATHAI_RESULTS / f"exp60_processbench_{eid}.json"
        try:
            run_one_extractor(eid, problems, out)
        except Exception as e:
            print(f"  [{eid}] ERROR: {e}")


if __name__ == "__main__":
    main()
