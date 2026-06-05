"""Threshold-sensitivity sweep.

Sweeps the DAJV accept_threshold over a range and reports the resulting
(coverage, precision) on the math175 test split. Mirrors the
risk-coverage curve but exposes the explicit operating-point knob so
deployment teams can pick a threshold.

Saves: artifacts/threshold_sensitivity.json
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}

ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
SEED = 42
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99, 0.995]


def run_on_bench(bench: str | None, tag: str) -> dict:
    aligned = align_extractor_caches(GROUP_B, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "bench": bench, "n": n, "skipped": "too few problems"}

    accept = aligned["accept"]
    solver_correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids)

    rng = random.Random(SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    cal_n = int(0.7 * n)
    cal_idx = indices[:cal_n]
    test_idx = indices[cal_n:]

    def _slice(arr, idx):
        return [arr[i] for i in idx]

    accept_cal = [_slice(accept[i], cal_idx) for i in range(k)]
    accept_test = [_slice(accept[i], test_idx) for i in range(k)]
    correct_cal = _slice(solver_correct, cal_idx)
    correct_test = _slice(solver_correct, test_idx)

    calibration = DajvCalibration.fit(accept_cal, correct_cal, extractor_ids)

    # Score each test item once (P_correct does not depend on threshold)
    scored = []
    for j in range(len(test_idx)):
        votes_j = [bool(accept_test[i][j]) for i in range(k)]
        out = dajv_aggregate(votes_j, calibration)
        scored.append({
            "p": out.get("P_correct"),
            "correct": bool(correct_test[j]),
        })

    rows = []
    for thr in THRESHOLDS:
        committed = [s for s in scored
                     if s["p"] is not None and s["p"] >= thr]
        n_c = len(committed)
        n_correct = sum(1 for s in committed if s["correct"])
        rows.append({
            "threshold": thr,
            "n_committed": n_c,
            "n_correct": n_correct,
            "coverage": n_c / len(test_idx),
            "precision": (n_correct / n_c) if n_c else None,
        })

    return {
        "tag": tag,
        "bench": bench,
        "n_test": len(test_idx),
        "k_extractors": k,
        "rows": rows,
    }


def main() -> None:
    out = []
    for bench, tag in [("math175", "math175"), (None, "B_full"),
                       ("aime", "aime"), ("cleanmath", "cleanmath")]:
        try:
            r = run_on_bench(bench, tag)
            print(f"\n=== {tag} ===")
            if r.get("skipped"):
                print(f"  skipped: {r['skipped']}")
            else:
                for row in r["rows"]:
                    p = row["precision"]
                    pstr = f"{p:.3f}" if p is not None else "  --  "
                    print(f"  thr={row['threshold']:.3f}  "
                          f"cov={row['coverage']:.3f}  "
                          f"prec={pstr}  "
                          f"n_commit={row['n_committed']}")
            out.append(r)
        except Exception as e:
            print(f"  ERROR on tag={tag}: {e}")
    save_artifact(out, ARTIFACTS / "threshold_sensitivity.json")
    print(f"\nWrote {ARTIFACTS / 'threshold_sensitivity.json'}")


if __name__ == "__main__":
    main()
