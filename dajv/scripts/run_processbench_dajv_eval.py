"""Evaluate DAJV on ProcessBench math (cross-distribution validation).

Loads the 4 OpenAI extractor caches collected by
``run_processbench_validation.py``, fits a fresh DAJV calibration on
half the problems (gold-free mode: classification is from
adversarial-filter only, not gold-classify), evaluates on the other
half. Reports ECE, Brier, risk-coverage AUC, and matched-coverage
precision.

This is the cross-distribution validation point that supports H4'
re-calibration recommendation: DAJV trained on the ProcessBench
half can predict final_answer_correct on the held-out half.

Saves: artifacts/processbench_dajv_eval.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_majority, naive_unanimous
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.evaluation.risk_coverage import risk_coverage_auc
from verifyensemble.utils.io import save_artifact

RESULTS = HERE.parent / ".." / "MATHAI" / "results"
CACHES = {
    "E06_gpt_5_mini": RESULTS / "exp60_processbench_E06_gpt_5_mini.json",
    "E08S_gpt_4o":    RESULTS / "exp60_processbench_E08S_gpt_4o.json",
    "E04S_gpt_4_1":   RESULTS / "exp60_processbench_E04S_gpt_4_1.json",
}

ARTIFACTS = HERE.parent / "artifacts"
SEED = 42


def align_processbench(caches: dict[str, Path]) -> dict | None:
    """Build aligned struct, exec, accept, label arrays."""
    records_per_eid: dict[str, dict[str, dict]] = {}
    for eid, path in caches.items():
        if not path.exists():
            print(f"  [{eid}] cache missing")
            return None
        d = json.load(open(path))
        records_per_eid[eid] = {r["id"]: r for r in d.get("results", [])}

    shared_ids = set.intersection(*(set(d.keys()) for d in records_per_eid.values()))
    pids = sorted(shared_ids)
    if len(pids) < 10:
        print(f"  shared problem count {len(pids)} too low")
        return None
    eids = list(caches.keys())
    k = len(eids)
    n = len(pids)

    accept = [[False] * n for _ in range(k)]
    struct = [[False] * n for _ in range(k)]
    exec_ = [[False] * n for _ in range(k)]
    correct = [False] * n
    for j, pid in enumerate(pids):
        first = records_per_eid[eids[0]][pid]
        correct[j] = bool(first.get("solver_correct"))
        for i, eid in enumerate(eids):
            r = records_per_eid[eid][pid]
            cls_ = r.get("classification")
            cv = r.get("candidate_verdict")
            struct[i][j] = (cls_ == "working")
            exec_[i][j] = (cv is True)
            accept[i][j] = (cls_ == "working" and cv is True)
    return {
        "extractor_ids": eids,
        "problem_ids": pids,
        "accept": accept,
        "struct": struct,
        "exec": exec_,
        "solver_correct": correct,
    }


def main() -> None:
    aligned = align_processbench(CACHES)
    if aligned is None:
        return
    n = len(aligned["problem_ids"])
    k = len(aligned["extractor_ids"])
    print(f"ProcessBench math: n={n} problems, k={k} extractors")

    rng = random.Random(SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    cal_n = max(int(0.6 * n), 8)
    cal_idx, test_idx = indices[:cal_n], indices[cal_n:]

    def _slice(arr, idx):
        return [arr[i] for i in idx]

    accept_cal = [_slice(aligned["accept"][i], cal_idx) for i in range(k)]
    accept_test = [_slice(aligned["accept"][i], test_idx) for i in range(k)]
    correct_cal = _slice(aligned["solver_correct"], cal_idx)
    correct_test = _slice(aligned["solver_correct"], test_idx)

    cal = DajvCalibration.fit(accept_cal, correct_cal, aligned["extractor_ids"])

    # Solver baseline: predict "correct" = always (or always wrong)
    n_actually_correct_test = sum(1 for c in correct_test if c)
    n_test = len(test_idx)
    print(f"  cal: {cal_n} problems, {sum(correct_cal)} correct ({100*sum(correct_cal)/cal_n:.0f}%)")
    print(f"  test: {n_test} problems, {n_actually_correct_test} correct ({100*n_actually_correct_test/n_test:.0f}%)")

    rows = []
    for j in range(n_test):
        votes = [bool(accept_test[i][j]) for i in range(k)]
        rows.append({
            "j": j,
            "label_correct": correct_test[j],
            "naive_unan": naive_unanimous(votes),
            "naive_maj":  naive_majority(votes),
            "dajv":       dajv_aggregate(votes, cal),
        })

    def head(key: str) -> dict:
        n_TP = n_FP = n_TN = n_FN = 0
        for r in rows:
            rec = r[key].get("recommendation")
            committed = rec == "COMMIT"
            label = r["label_correct"]
            if committed and label:
                n_TP += 1
            elif committed and not label:
                n_FP += 1
            elif not committed and label:
                n_FN += 1
            else:
                n_TN += 1
        n_committed = n_TP + n_FP
        n_total = n_TP + n_FP + n_TN + n_FN
        precision = (n_TP / n_committed) if n_committed else None
        coverage = n_committed / n_total if n_total else 0.0
        # Specificity = true negative rate on wrong-candidate subset
        n_neg = n_TN + n_FP
        specificity = (n_TN / n_neg) if n_neg else None
        return {
            "TP": n_TP, "FP": n_FP, "TN": n_TN, "FN": n_FN,
            "precision": precision,
            "coverage": coverage,
            "specificity": specificity,
        }

    def calib(key: str) -> dict:
        items = [(r[key].get("P_correct"), r["label_correct"])
                 for r in rows if r[key].get("P_correct") is not None]
        if not items:
            return {"ece": None, "brier": None, "rc_auc": None, "n": 0}
        confs, ys = zip(*items)
        return {
            "ece": expected_calibration_error(confs, ys, n_bins=5),
            "brier": brier_score(confs, ys),
            "rc_auc": risk_coverage_auc(confs, ys),
            "n": len(items),
        }

    summary = {
        "n_problems": n,
        "n_cal": cal_n,
        "n_test": n_test,
        "k_extractors": k,
        "extractor_ids": aligned["extractor_ids"],
        "label_correct_rate_test": n_actually_correct_test / n_test,
        "operating_points": {
            "naive_unanimous": head("naive_unan"),
            "naive_majority":  head("naive_maj"),
            "dajv":            head("dajv"),
        },
        "calibration": {
            "naive_unanimous": calib("naive_unan"),
            "naive_majority":  calib("naive_maj"),
            "dajv":            calib("dajv"),
        },
    }
    print("\nOperating points (TP/FP/TN/FN):")
    for k_ in ("naive_unanimous", "naive_majority", "dajv"):
        op = summary["operating_points"][k_]
        cb = summary["calibration"][k_]
        prec = op["precision"]
        spec = op["specificity"]
        ps = f"{prec:.3f}" if prec is not None else "  ---"
        ss = f"{spec:.3f}" if spec is not None else "  ---"
        ece = cb["ece"]
        es = f"{ece:.3f}" if ece is not None else "  ---"
        auc = cb["rc_auc"]
        as_ = f"{auc:.3f}" if auc is not None else "  ---"
        print(f"  {k_:18s} TP/FP/TN/FN={op['TP']}/{op['FP']}/{op['TN']}/{op['FN']}  "
              f"cov={op['coverage']:.3f} prec={ps} spec={ss} "
              f"ECE={es} AUC={as_}")
    save_artifact(summary, ARTIFACTS / "processbench_dajv_eval.json")
    print(f"\nWrote {ARTIFACTS / 'processbench_dajv_eval.json'}")


if __name__ == "__main__":
    main()
