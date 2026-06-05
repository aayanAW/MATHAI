"""Sensitivity sweep for the Block-Sparse Ising Aggregator (BSIA).

Sweeps ``rho_shrinkage`` and ``smooth`` jointly using
``nested cross-validation``: on each calibration split, run a
secondary CV within the calibration data to pick the best
hyperparameter set, then evaluate on the test split. This avoids
test-set tuning.

Saves: artifacts/bsia_sensitivity.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv_xmod import (
    BlockSparseIsingCalibration,
    bsia_aggregate,
)
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
    "E08S_gpt_4o":           HERE.parent / "../MATHAI/results/exp55_gpt4o_extractor.json",
    "E04S_gpt_4_1":          HERE.parent / "../MATHAI/results/exp56_gpt41_extractor.json",
}
_GPT5 = HERE.parent / "../MATHAI/results/exp57_gpt5_extractor.json"
if _GPT5.exists():
    try:
        if len(json.load(open(_GPT5)).get("results", [])) >= 300:
            GROUP_B["E10S_gpt_5"] = _GPT5
    except Exception:
        pass
_OPUS47 = HERE.parent / "../MATHAI/results/exp58_opus47_extractor.json"
if _OPUS47.exists():
    try:
        if len(json.load(open(_OPUS47)).get("results", [])) >= 300:
            GROUP_B["E01A_claude_opus_4_7"] = _OPUS47
    except Exception:
        pass
_OPUS46 = HERE.parent / "../MATHAI/results/exp59_opus46_extractor.json"
if _OPUS46.exists():
    try:
        if len(json.load(open(_OPUS46)).get("results", [])) >= 300:
            GROUP_B["E02A_claude_opus_4_6"] = _OPUS46
    except Exception:
        pass
_HAIKU45 = HERE.parent / "../MATHAI/results/exp61_haiku45_extractor.json"
if _HAIKU45.exists():
    try:
        if len(json.load(open(_HAIKU45)).get("results", [])) >= 300:
            GROUP_B["E03A_claude_haiku_4_5"] = _HAIKU45
    except Exception:
        pass
_LLAMA33 = HERE.parent / "../MATHAI/results/exp62_llama33_70b_extractor.json"
if _LLAMA33.exists():
    try:
        if len(json.load(open(_LLAMA33)).get("results", [])) >= 300:
            GROUP_B["E13_llama_3_3_70B"] = _LLAMA33
    except Exception:
        pass
_QWEN235 = HERE.parent / "../MATHAI/results/exp63_qwen3_235b_extractor.json"
if _QWEN235.exists():
    try:
        if len(json.load(open(_QWEN235)).get("results", [])) >= 300:
            GROUP_B["E14_qwen_3_235B"] = _QWEN235
    except Exception:
        pass

ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _signals(aligned: dict, idx: list[int]):
    k = len(aligned["extractor_ids"])
    cls_ = aligned["classification"]
    cv = aligned["candidate_verdict"]
    struct = [[(cls_[i][j] == "working") for j in idx] for i in range(k)]
    exec_ = [[(cv[i][j] is True) for j in idx] for i in range(k)]
    return struct, exec_


def _slice(arr, idx):
    return [arr[i] for i in idx]


def _eval_split(
    cal_idx: list[int],
    test_idx: list[int],
    aligned: dict,
    rho_shrinkage: float,
    smooth: float,
) -> dict:
    k = len(aligned["extractor_ids"])
    correct = [bool(c) for c in aligned["solver_correct"]]
    s_cal, x_cal = _signals(aligned, cal_idx)
    s_test, x_test = _signals(aligned, test_idx)
    c_cal = _slice(correct, cal_idx)
    c_test = _slice(correct, test_idx)
    cal = BlockSparseIsingCalibration.fit(
        s_cal, x_cal, c_cal, aligned["extractor_ids"],
        smooth=smooth, rho_shrinkage=rho_shrinkage,
    )
    confs: list[float] = []
    ys: list[bool] = []
    n_commit = n_commit_correct = 0
    for j in range(len(test_idx)):
        s = [bool(s_test[i][j]) for i in range(k)]
        x = [bool(x_test[i][j]) for i in range(k)]
        out = bsia_aggregate(s, x, cal)
        p = out.get("P_correct")
        if p is not None:
            confs.append(p)
            ys.append(c_test[j])
        if out.get("recommendation") == "COMMIT":
            n_commit += 1
            if c_test[j]:
                n_commit_correct += 1
    n_test = len(test_idx)
    return {
        "coverage": n_commit / n_test if n_test else 0.0,
        "precision": (n_commit_correct / n_commit) if n_commit else None,
        "n_commit": n_commit, "n_correct": n_commit_correct,
        "ece": expected_calibration_error(confs, ys, n_bins=5) if confs else None,
        "brier": brier_score(confs, ys) if confs else None,
    }


def nested_cv_one_outer(
    aligned: dict,
    outer_seed: int,
    grid: list[tuple[float, float]],
) -> dict:
    """Outer-split, inner-CV hyperparameter selection."""
    n = len(aligned["problem_ids"])
    rng = random.Random(outer_seed)
    idx = list(range(n))
    rng.shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]

    # Inner 3-fold over cal_idx: pick the (rho_shrinkage, smooth) maximizing
    # a "score" = coverage at the >= 0.95 commit threshold subject to
    # precision >= 0.95.  (We use a slightly lower precision floor inside
    # CV so that selection is not all degenerate.)
    inner_folds = 3
    inner_n = len(cal_idx)
    fold_size = inner_n // inner_folds
    inner_idx = list(cal_idx)
    rng.shuffle(inner_idx)
    folds = [inner_idx[i * fold_size:(i + 1) * fold_size] for i in range(inner_folds)]

    def _inner_score(rho: float, sm: float) -> float:
        coverages: list[float] = []
        for f in range(inner_folds):
            test_i = folds[f]
            cal_i = [j for fi, fold in enumerate(folds) if fi != f for j in fold]
            r = _eval_split(cal_i, test_i, aligned, rho, sm)
            if r["precision"] is not None and r["precision"] >= 0.95:
                coverages.append(r["coverage"])
        return sum(coverages) / len(coverages) if coverages else 0.0

    inner_scores = [(rho, sm, _inner_score(rho, sm)) for rho, sm in grid]
    inner_scores.sort(key=lambda t: -t[2])
    best_rho, best_sm, best_score = inner_scores[0]

    # Final eval on outer test split with chosen hyperparams.
    final = _eval_split(cal_idx, test_idx, aligned, best_rho, best_sm)
    return {
        "outer_seed": outer_seed,
        "best_rho_shrinkage": best_rho,
        "best_smooth": best_sm,
        "inner_score": best_score,
        "inner_grid": [{"rho": r, "smooth": s, "score": sc}
                       for r, s, sc in inner_scores],
        "final": final,
    }


def main() -> None:
    aligned = align_extractor_caches(GROUP_B, bench="math175")
    print(f"n={len(aligned['problem_ids'])} k={len(aligned['extractor_ids'])}")

    grid = []
    for rho in (0.0, 0.25, 0.5, 0.75):
        for sm in (0.1, 0.5, 1.0):
            grid.append((rho, sm))

    out = []
    for seed in range(1, 11):
        r = nested_cv_one_outer(aligned, seed, grid)
        out.append(r)
        f = r["final"]
        print(f"seed={seed}: best (rho={r['best_rho_shrinkage']:.2f}, "
              f"sm={r['best_smooth']:.2f}) cov={f['coverage']:.3f} "
              f"prec={(f['precision'] or 0):.3f} ECE={(f['ece'] or 0):.3f}")

    covs = [r["final"]["coverage"] for r in out]
    precs = [r["final"]["precision"] for r in out if r["final"]["precision"] is not None]
    eces = [r["final"]["ece"] for r in out if r["final"]["ece"] is not None]

    def _ms(v):
        m = sum(v) / len(v)
        s = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
        return {"mean": m, "std": s, "n": len(v)}

    summary = {
        "n_seeds": len(out),
        "coverage": _ms(covs),
        "precision": _ms(precs) if precs else None,
        "ece": _ms(eces) if eces else None,
        "best_hp_distribution": {},
    }
    # Track which hyperparam configuration was chosen most often.
    chosen = [(r["best_rho_shrinkage"], r["best_smooth"]) for r in out]
    for hp in chosen:
        key = f"rho={hp[0]:.2f}_sm={hp[1]:.2f}"
        summary["best_hp_distribution"][key] = (
            summary["best_hp_distribution"].get(key, 0) + 1
        )

    print("\n=== nested-CV BSIA on math175 ===")
    print(f"coverage  : {summary['coverage']['mean']:.3f} +/- {summary['coverage']['std']:.3f}")
    if summary["precision"]:
        print(f"precision : {summary['precision']['mean']:.3f} +/- {summary['precision']['std']:.3f}")
    if summary["ece"]:
        print(f"ECE       : {summary['ece']['mean']:.3f} +/- {summary['ece']['std']:.3f}")
    print(f"chosen HPs: {summary['best_hp_distribution']}")

    save_artifact(
        {"per_seed": out, "summary": summary},
        ARTIFACTS / "bsia_sensitivity.json",
    )
    print(f"\nWrote {ARTIFACTS / 'bsia_sensitivity.json'}")


if __name__ == "__main__":
    main()
