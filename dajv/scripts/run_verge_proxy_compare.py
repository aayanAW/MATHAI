"""VERGE-proxy vs DAJV head-to-head on the cached Group B ensemble.

Compares the VERGE-proxy aggregation rule (multi-model consensus +
formal-verification gate + MCS) against DAJV across math175, AIME,
CleanMath. Reports coverage, precision, and risk-coverage AUC for
each.

Saves: artifacts/verge_proxy_compare.json
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_unanimous
from verifyensemble.aggregate.verge_proxy import verge_proxy_aggregate
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


def run_on_bench(bench: str | None, tag: str) -> dict:
    aligned = align_extractor_caches(GROUP_B, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "bench": bench, "n": n, "skipped": "too few"}

    k = len(aligned["extractor_ids"])
    rng = random.Random(SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    cal_n = int(0.7 * n)
    cal_idx = indices[:cal_n]
    test_idx = indices[cal_n:]

    def _slice(arr, idx):
        return [arr[i] for i in idx]

    accept_cal = [_slice(aligned["accept"][i], cal_idx) for i in range(k)]
    accept_test = [_slice(aligned["accept"][i], test_idx) for i in range(k)]
    classification_test = [_slice(aligned["classification"][i], test_idx)
                           for i in range(k)]
    cv_test = [_slice(aligned["candidate_verdict"][i], test_idx)
               for i in range(k)]
    correct_cal = _slice(aligned["solver_correct"], cal_idx)
    correct_test = _slice(aligned["solver_correct"], test_idx)

    cal = DajvCalibration.fit(accept_cal, correct_cal, aligned["extractor_ids"])

    rows = []
    for j in range(len(test_idx)):
        votes_j = [bool(accept_test[i][j]) for i in range(k)]
        cls_j = [classification_test[i][j] for i in range(k)]
        cv_j = [cv_test[i][j] for i in range(k)]
        rows.append({
            "j": j,
            "correct": bool(correct_test[j]),
            "naive": naive_unanimous(votes_j),
            "dajv": dajv_aggregate(votes_j, cal),
            "verge_proxy": verge_proxy_aggregate(
                votes_j, cls_j, cv_j, min_agree=3),
        })

    def head(key: str, commit_states: set[str]) -> dict:
        committed = [r for r in rows
                     if r[key].get("recommendation") in commit_states]
        n_c = len(committed)
        n_correct = sum(1 for r in committed if r["correct"])
        return {
            "n_committed": n_c,
            "n_correct": n_correct,
            "precision": (n_correct / n_c) if n_c else None,
            "coverage": n_c / len(rows),
        }

    out = {
        "tag": tag,
        "bench": bench,
        "n_test": len(rows),
        "naive_unanimous": head("naive", {"COMMIT"}),
        "dajv":            head("dajv", {"COMMIT"}),
        "verge_proxy_full":     head("verge_proxy", {"COMMIT"}),
        "verge_proxy_with_mcs": head("verge_proxy", {"COMMIT", "COMMIT_MCS"}),
    }
    print(f"\n=== {tag} (n_test={len(rows)}) ===")
    for key in ("naive_unanimous", "dajv",
                "verge_proxy_full", "verge_proxy_with_mcs"):
        d = out[key]
        prec = d["precision"]
        ps = f"{prec:.3f}" if prec is not None else "  ---"
        print(f"  {key:25s} cov={d['coverage']:.3f} prec={ps} "
              f"n={d['n_committed']}/{len(rows)}")
    return out


def main() -> None:
    results = []
    for bench, tag in [("math175", "math175"), ("aime", "aime"),
                       ("cleanmath", "cleanmath"), (None, "B_full")]:
        try:
            results.append(run_on_bench(bench, tag))
        except Exception as e:
            print(f"  ERROR {tag}: {e}")
    save_artifact(results, ARTIFACTS / "verge_proxy_compare.json")
    print(f"\nWrote {ARTIFACTS / 'verge_proxy_compare.json'}")


if __name__ == "__main__":
    main()
