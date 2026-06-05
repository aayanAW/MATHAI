"""End-to-end quickstart example.

Runs the full DAJV pipeline on cached extractor outputs from the
math175 benchmark, prints the headline result, and saves a small
demonstration artifact.

Usage:
    PYTHONPATH=. python3 scripts/example_quickstart.py
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
from verifyensemble.utils.io import align_extractor_caches

GROUP_B = {
    "gpt-oss":  HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "gpt-5-m":  HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "claude":   HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "qwen3":    HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}
SEED = 42


def main() -> None:
    # 1) Load aligned extractor outputs for the math175 bench
    aligned = align_extractor_caches(GROUP_B, bench="math175")
    accept = aligned["accept"]
    correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids)
    n = len(aligned["problem_ids"])
    print(f"Loaded n={n} problems across {k} cross-family extractors.")

    # 2) 70/30 calibration / test split
    rng = random.Random(SEED)
    idx = list(range(n)); rng.shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    correct_cal = [correct[j] for j in cal_idx]
    print(f"Calibration n={len(cal_idx)}, test n={len(test_idx)}.")

    # 3) Fit DAJV calibration
    calibration = DajvCalibration.fit(accept_cal, correct_cal, extractor_ids)
    print("Fitted calibration:")
    print(f"  pi_pos = {[round(p, 3) for p in calibration.pi_pos]}")
    print(f"  pi_neg = {[round(p, 3) for p in calibration.pi_neg]}")
    print(f"  prior_correct = {round(calibration.prior_correct, 3)}")

    # 4) Deploy DAJV + naive on test split
    naive_committed = 0; naive_correct = 0
    dajv_committed = 0;  dajv_correct = 0
    dajv_confs, dajv_ys = [], []
    naive_confs, naive_ys = [], []
    for jj, j in enumerate(test_idx):
        votes = [accept[i][j] for i in range(k)]
        out_n = naive_unanimous(votes)
        out_d = dajv_aggregate(votes, calibration)

        if out_n["recommendation"] == "COMMIT":
            naive_committed += 1
            if correct[j]:
                naive_correct += 1
        if out_d["recommendation"] == "COMMIT":
            dajv_committed += 1
            if correct[j]:
                dajv_correct += 1

        if out_n["P_correct"] is not None:
            naive_confs.append(out_n["P_correct"])
            naive_ys.append(correct[j])
        if out_d["P_correct"] is not None:
            dajv_confs.append(out_d["P_correct"])
            dajv_ys.append(correct[j])

    print("\n--- Headline results ---")
    print(f"Naive unanimous: {naive_correct}/{naive_committed} = "
          f"{naive_correct / max(naive_committed,1):.3f} precision, "
          f"{naive_committed / len(test_idx):.3f} coverage")
    print(f"DAJV:            {dajv_correct}/{dajv_committed} = "
          f"{dajv_correct / max(dajv_committed,1):.3f} precision, "
          f"{dajv_committed / len(test_idx):.3f} coverage")

    print("\n--- Calibration quality ---")
    print(f"Naive ECE (5 bins): {expected_calibration_error(naive_confs, naive_ys, n_bins=5):.3f}")
    print(f"DAJV  ECE (5 bins): {expected_calibration_error(dajv_confs, dajv_ys, n_bins=5):.3f}")
    print(f"Naive Brier:        {brier_score(naive_confs, naive_ys):.3f}")
    print(f"DAJV  Brier:        {brier_score(dajv_confs, dajv_ys):.3f}")

    # 5) Save artifact
    out_path = HERE.parent / "artifacts" / "quickstart_demo.json"
    with out_path.open("w") as f:
        json.dump({
            "n_test": len(test_idx),
            "naive": {"n_correct": naive_correct, "n_committed": naive_committed},
            "dajv":  {"n_correct": dajv_correct, "n_committed": dajv_committed},
            "calibration_marginals_pos": calibration.pi_pos,
            "calibration_marginals_neg": calibration.pi_neg,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
