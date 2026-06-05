"""Solver-rotation analysis: does DAJV calibration generalize to a new solver?

Uses MATHAI/results/exp44_solver_rotation.json which has DeepSeek-V3
as the rotated solver (replacing Qwen2.5-7B-Instruct-Turbo), with both
Llama-3.3-70B and DeepSeek-V3 as extractors on 330 problems.

We compare:
  1. DAJV fit on Qwen-solver cache, deployed on Qwen-solver cache (baseline)
  2. DAJV fit on Qwen-solver cache, deployed on DeepSeek-solver cache (rotation)

If the calibration generalizes across solver families, the metrics
should be similar. If not, that's a caveat for the paper.

Output:
  artifacts/solver_rotation_dajv.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_unanimous
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.utils.io import save_artifact

EXP44 = HERE.parent / "../MATHAI/results/exp44_solver_rotation.json"
ARTIFACTS = HERE.parent / "artifacts"
SEED = 42


def load_exp44() -> dict:
    """Reshape exp44 into accept[k][n] / correct[n] / extractor_ids tensors."""
    with EXP44.open() as f:
        d = json.load(f)
    rows = d["rows"]
    accept_llama, accept_deepseek = [], []
    correct = []
    pids = []
    for r in rows:
        pids.append(r["id"])
        correct.append(bool(r["solver_correct"]))
        # accept = working AND candidate_verdict=True
        for tag, target in [("llama", accept_llama),
                             ("deepseek", accept_deepseek)]:
            cls = r.get(tag, {}).get("classification")
            cv = r.get(tag, {}).get("candidate_verdict")
            target.append(cls == "working" and cv is True)
    return {
        "extractor_ids": ["llama_3_3_70B", "deepseek_v3"],
        "accept": [accept_llama, accept_deepseek],
        "correct": correct,
        "pids": pids,
        "solver": d.get("solver", "?"),
    }


def split_and_fit(accept, correct, extractor_ids, seed=SEED) -> tuple[DajvCalibration, list, list]:
    n = len(correct)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(len(extractor_ids))]
    cal_correct = [correct[j] for j in cal_idx]
    cal = DajvCalibration.fit(accept_cal, cal_correct, extractor_ids)
    return cal, cal_idx, test_idx


def evaluate(accept, correct, extractor_ids, cal: DajvCalibration,
             test_idx: list[int]) -> dict:
    n_committed_d = 0; n_correct_d = 0
    n_committed_n = 0; n_correct_n = 0
    dconf, dy, nconf, ny = [], [], [], []
    for jj in test_idx:
        v = [accept[i][jj] for i in range(len(extractor_ids))]
        r_d = dajv_aggregate(v, cal)
        r_n = naive_unanimous(v)
        if r_d["recommendation"] == "COMMIT":
            n_committed_d += 1
            if correct[jj]: n_correct_d += 1
        if r_n["recommendation"] == "COMMIT":
            n_committed_n += 1
            if correct[jj]: n_correct_n += 1
        if r_d["P_correct"] is not None:
            dconf.append(r_d["P_correct"]); dy.append(correct[jj])
        if r_n["P_correct"] is not None:
            nconf.append(r_n["P_correct"]); ny.append(correct[jj])

    return {
        "n_test": len(test_idx),
        "dajv": {
            "n_committed": n_committed_d, "n_correct": n_correct_d,
            "precision_at_commit": n_correct_d / max(n_committed_d, 1),
            "coverage": n_committed_d / len(test_idx),
            "ece": expected_calibration_error(dconf, dy, n_bins=5) if dconf else None,
            "brier": brier_score(dconf, dy) if dconf else None,
        },
        "naive_unanimous": {
            "n_committed": n_committed_n, "n_correct": n_correct_n,
            "precision_at_commit": n_correct_n / max(n_committed_n, 1),
            "coverage": n_committed_n / len(test_idx),
            "ece": expected_calibration_error(nconf, ny, n_bins=5) if nconf else None,
            "brier": brier_score(nconf, ny) if nconf else None,
        },
    }


def main() -> None:
    rot = load_exp44()
    print(f"Loaded solver-rotation cache: solver={rot['solver']}, "
          f"n={len(rot['correct'])} k={len(rot['extractor_ids'])}")
    print(f"  n_correct={sum(rot['correct'])} n_wrong={sum(1 for c in rot['correct'] if not c)}")

    # Fit DAJV on the rotated-solver cache (the only data we have for
    # this 2-extractor setting); 70/30 split.
    cal, cal_idx, test_idx = split_and_fit(
        rot["accept"], rot["correct"], rot["extractor_ids"]
    )
    print(f"\nFitted DAJV on rotated solver (DeepSeek-V3) calibration "
          f"n_cal={len(cal_idx)}")
    print(f"  pi_pos = {[round(p, 3) for p in cal.pi_pos]}")
    print(f"  pi_neg = {[round(p, 3) for p in cal.pi_neg]}")
    print(f"  prior_correct = {round(cal.prior_correct, 3)}")

    metrics = evaluate(rot["accept"], rot["correct"], rot["extractor_ids"],
                       cal, test_idx)
    print(f"\n--- Test split metrics (n_test={metrics['n_test']}) ---")
    for method in ("dajv", "naive_unanimous"):
        m = metrics[method]
        ece_str = f"{m['ece']:.3f}" if m['ece'] is not None else "n/a"
        print(f"  {method}: {m['n_correct']}/{m['n_committed']} "
              f"prec={m['precision_at_commit']:.3f} "
              f"cov={m['coverage']:.3f} ECE={ece_str}")

    summary = {
        "solver": rot["solver"],
        "extractor_ids": rot["extractor_ids"],
        "n_problems": len(rot["correct"]),
        "n_cal": len(cal_idx),
        "n_test": len(test_idx),
        "calibration": {
            "pi_pos": cal.pi_pos,
            "pi_neg": cal.pi_neg,
            "prior_correct": cal.prior_correct,
        },
        "metrics": metrics,
    }
    save_artifact(summary, ARTIFACTS / "solver_rotation_dajv.json")
    print(f"\nWrote {ARTIFACTS / 'solver_rotation_dajv.json'}")


if __name__ == "__main__":
    main()
