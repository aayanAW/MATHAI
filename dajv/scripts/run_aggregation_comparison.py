"""Compare aggregation rules on the cached X-SGRV data.

Uses the 4-extractor Group B (gptoss, gpt5, claude, qwen3) on the
combined math175 + AIME + CleanMath problem set (n=330). Splits
randomly into calibration (70%) and test (30%); fits the DAJV
calibration on the calibration split; evaluates naive consensus and
DAJV on the test split.

Reports:
  - precision@coverage curves
  - ECE
  - Brier score
  - point precision at unanimous-accept operating point
  - McNemar mid-p between DAJV and naive consensus

Saves: artifacts/aggregation_results.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.care import care_aggregate, fit_care_weights
from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_majority, naive_unanimous
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.evaluation.mcnemar import mcnemar_mid_p
from verifyensemble.evaluation.risk_coverage import risk_coverage_auc, risk_coverage_curve
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}

ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Reproducibility
SEED = 42


def run_on_bench(bench: str | None, tag: str) -> dict:
    aligned = align_extractor_caches(GROUP_B, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "bench": bench, "n": n, "skipped": "too few problems"}

    accept = aligned["accept"]
    solver_correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids)

    # 70/30 calibration/test split
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

    print(f"\n=== bench={bench} (tag={tag}) ===")
    print(f"  n_cal={cal_n} n_test={len(test_idx)} k={k}")
    print(f"  cal: n_correct={sum(correct_cal)} n_wrong={sum(1 for c in correct_cal if not c)}")
    print(f"  test: n_correct={sum(correct_test)} n_wrong={sum(1 for c in correct_test if not c)}")

    # Fit calibrations
    dajv = DajvCalibration.fit(accept_cal, correct_cal, extractor_ids)
    care_weights = fit_care_weights(accept_cal, correct_cal)

    # Evaluate
    results_test = []
    for j in range(len(test_idx)):
        votes_j = [bool(accept_test[i][j]) for i in range(k)]
        out_naive = naive_unanimous(votes_j)
        out_maj = naive_majority(votes_j)
        out_dajv = dajv_aggregate(votes_j, dajv)
        out_care = care_aggregate(votes_j, care_weights, threshold=0.5)
        results_test.append({
            "j": j,
            "votes": votes_j,
            "correct": bool(correct_test[j]),
            "naive_unanimous": out_naive,
            "naive_majority": out_maj,
            "dajv": out_dajv,
            "care": out_care,
        })

    # Headline operating point: only items where at least one method commits
    def committed_precision(method_key: str) -> dict:
        committed = [r for r in results_test
                     if r[method_key].get("recommendation") == "COMMIT"]
        if not committed:
            return {"n_committed": 0, "n_correct": 0,
                    "precision": None, "coverage": 0.0}
        n_c = len(committed)
        n_correct = sum(1 for r in committed if r["correct"])
        return {"n_committed": n_c, "n_correct": n_correct,
                "precision": n_correct / n_c,
                "coverage": n_c / len(results_test)}

    head_naive_u = committed_precision("naive_unanimous")
    head_naive_m = committed_precision("naive_majority")
    head_dajv = committed_precision("dajv")
    head_care = committed_precision("care")

    # ECE / Brier (only items with a non-None P_correct)
    def calib_metrics(method_key: str) -> dict:
        items = [(r[method_key]["P_correct"], r["correct"])
                 for r in results_test if r[method_key].get("P_correct") is not None]
        if not items:
            return {"ece": None, "brier": None, "n": 0}
        confs, ys = zip(*items)
        return {
            "ece": expected_calibration_error(confs, ys, n_bins=5),
            "brier": brier_score(confs, ys),
            "n": len(items),
        }

    calib_naive_u = calib_metrics("naive_unanimous")
    calib_naive_m = calib_metrics("naive_majority")
    calib_dajv = calib_metrics("dajv")
    calib_care = calib_metrics("care")

    # Risk-coverage curves using P_correct as the score
    def rc(method_key: str):
        items = [(r[method_key]["P_correct"], r["correct"])
                 for r in results_test if r[method_key].get("P_correct") is not None]
        if not items:
            return {"auc": None, "curve": None, "n": 0}
        scores, ys = zip(*items)
        return {
            "auc": risk_coverage_auc(scores, ys),
            "curve": risk_coverage_curve(scores, ys),
            "n": len(items),
        }
    rc_naive_u = rc("naive_unanimous")
    rc_naive_m = rc("naive_majority")
    rc_dajv = rc("dajv")
    rc_care = rc("care")

    # McNemar: at the unanimous-accept operating point, do dajv and naive
    # produce different commit decisions?
    # Matched-binary: did each method's commit==True align with correct?
    # We compare "method committed AND was correct" as the success.
    def commit_correct(method_key: str):
        return [(r[method_key].get("recommendation") == "COMMIT") and r["correct"]
                for r in results_test]
    mcnemar_dajv_vs_naive = mcnemar_mid_p(commit_correct("naive_unanimous"),
                                          commit_correct("dajv"))

    summary = {
        "tag": tag,
        "bench": bench,
        "n_test": len(test_idx),
        "k_extractors": k,
        "extractors": extractor_ids,
        "headline": {
            "naive_unanimous": head_naive_u,
            "naive_majority": head_naive_m,
            "dajv": head_dajv,
            "care": head_care,
        },
        "calibration": {
            "naive_unanimous": calib_naive_u,
            "naive_majority": calib_naive_m,
            "dajv": calib_dajv,
            "care": calib_care,
        },
        "risk_coverage_auc": {
            "naive_unanimous": rc_naive_u["auc"],
            "naive_majority": rc_naive_m["auc"],
            "dajv": rc_dajv["auc"],
            "care": rc_care["auc"],
        },
        "mcnemar_dajv_vs_naive_unanimous": mcnemar_dajv_vs_naive,
        "calibration_marginals_pos": dajv.pi_pos,
        "calibration_marginals_neg": dajv.pi_neg,
        "prior_correct": dajv.prior_correct,
    }
    print(json.dumps(summary["headline"], indent=2))
    print(f"  ECE  -- naive: {calib_naive_u['ece']}, DAJV: {calib_dajv['ece']}, CARE: {calib_care['ece']}")
    print(f"  Brier-- naive: {calib_naive_u['brier']}, DAJV: {calib_dajv['brier']}, CARE: {calib_care['brier']}")
    return summary


def main() -> None:
    all_results = []
    for bench, tag in [(None, "B_full"), ("math175", "B_math175"),
                        ("aime", "B_aime"), ("cleanmath", "B_cleanmath")]:
        try:
            r = run_on_bench(bench, tag)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR on tag={tag}: {e}")
    save_artifact(all_results, ARTIFACTS / "aggregation_results.json")
    print(f"\nWrote {ARTIFACTS / 'aggregation_results.json'}")


if __name__ == "__main__":
    main()
