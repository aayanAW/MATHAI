"""Evaluate the cross-modality DAJV (xmod) aggregator.

Hypothesis H7' (per PRE_REGISTRATION_v2.md): the cross-modality
independence measurement implies that a posterior factorized across
the structural and executable modalities should improve the calibrated
operating point at the default $\\tau = 0.95$ threshold.

Test against default DAJV (accept only) and against the naive 2k-signal
hybrid. Average across 10 calibration/test seeds on the 6-extractor or
7-extractor cache.

Saves: artifacts/xmod_dajv.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.dajv_xmod import (
    BlockSparseIsingCalibration,
    XmodCalibration,
    XmodJointCalibration,
    bsia_aggregate,
    bsia_ensemble_aggregate,
    bsia_isotonic_aggregate,
    fit_bsia_isotonic,
    fit_bsia_temperature,
    xmod_aggregate,
    xmod_agreement_aggregate,
    xmod_joint_aggregate,
    xmod_struct_gated_exec_dajv_aggregate,
)
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.evaluation.risk_coverage import risk_coverage_auc
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


def _slice(arr, idx):
    return [arr[i] for i in idx]


def _build_signals(aligned: dict, idx: list[int]):
    k = len(aligned["extractor_ids"])
    cls_ = aligned["classification"]
    cv = aligned["candidate_verdict"]
    accept = aligned["accept"]
    struct = [[(cls_[i][j] == "working") for j in idx] for i in range(k)]
    exec_ = [[(cv[i][j] is True) for j in idx] for i in range(k)]
    acc = [[bool(accept[i][j]) for j in idx] for i in range(k)]
    return struct, exec_, acc


def evaluate_one_seed(aligned: dict, seed: int) -> dict:
    n = len(aligned["problem_ids"])
    k = len(aligned["extractor_ids"])
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = indices[:cal_n], indices[cal_n:]
    correct = [bool(c) for c in aligned["solver_correct"]]

    struct_cal, exec_cal, accept_cal = _build_signals(aligned, cal_idx)
    struct_test, exec_test, accept_test = _build_signals(aligned, test_idx)
    correct_cal = _slice(correct, cal_idx)
    correct_test = _slice(correct, test_idx)

    # Fit
    dajv_default = DajvCalibration.fit(
        accept_cal, correct_cal, aligned["extractor_ids"]
    )
    xmod_cal = XmodCalibration.fit(
        struct_cal, exec_cal, correct_cal, aligned["extractor_ids"]
    )
    xmod_joint_cal = XmodJointCalibration.fit(
        struct_cal, exec_cal, correct_cal, aligned["extractor_ids"]
    )
    # exec-only DAJV (for struct-gated variant)
    exec_dajv = DajvCalibration.fit(
        exec_cal, correct_cal,
        [f"x_{e}" for e in aligned["extractor_ids"]]
    )
    # Block-Sparse Ising aggregator (BSIA) -- H7' deeper attempt.
    bsia_cal = BlockSparseIsingCalibration.fit(
        struct_cal, exec_cal, correct_cal, aligned["extractor_ids"]
    )
    # BSIA + temperature scaling (post-hoc calibration on cal set).
    bsia_temp_cal = BlockSparseIsingCalibration.fit(
        struct_cal, exec_cal, correct_cal, aligned["extractor_ids"]
    )
    bsia_temp_cal.temperature = fit_bsia_temperature(
        bsia_temp_cal, struct_cal, exec_cal, correct_cal,
    )
    # BSIA + isotonic recalibration (post-hoc, monotone).
    iso_xs, iso_ys = fit_bsia_isotonic(
        bsia_cal, struct_cal, exec_cal, correct_cal,
    )

    # Evaluate
    def _eval(predict_fn):
        confs: list[float] = []
        ys: list[bool] = []
        n_commit = n_commit_correct = 0
        n_test = len(test_idx)
        for j in range(n_test):
            out = predict_fn(j)
            p = out.get("P_correct")
            if p is not None:
                confs.append(p)
                ys.append(correct_test[j])
            if out.get("recommendation") == "COMMIT":
                n_commit += 1
                if correct_test[j]:
                    n_commit_correct += 1
        return {
            "coverage": n_commit / n_test if n_test else 0.0,
            "precision": (n_commit_correct / n_commit) if n_commit else None,
            "n_commit": n_commit,
            "n_correct": n_commit_correct,
            "ece": expected_calibration_error(confs, ys, n_bins=5) if confs else None,
            "brier": brier_score(confs, ys) if confs else None,
            "rc_auc": risk_coverage_auc(confs, ys) if confs else None,
        }

    def _default(j: int):
        votes = [bool(accept_test[i][j]) for i in range(k)]
        return dajv_aggregate(votes, dajv_default)

    def _xmod(j: int):
        s = [bool(struct_test[i][j]) for i in range(k)]
        x = [bool(exec_test[i][j]) for i in range(k)]
        return xmod_aggregate(s, x, xmod_cal)

    def _xmod_agree(j: int):
        s = [bool(struct_test[i][j]) for i in range(k)]
        x = [bool(exec_test[i][j]) for i in range(k)]
        return xmod_agreement_aggregate(s, x, xmod_cal)

    def _xmod_joint(j: int):
        s = [bool(struct_test[i][j]) for i in range(k)]
        x = [bool(exec_test[i][j]) for i in range(k)]
        return xmod_joint_aggregate(s, x, xmod_joint_cal)

    def _xmod_struct_gated(j: int):
        s = [bool(struct_test[i][j]) for i in range(k)]
        x = [bool(exec_test[i][j]) for i in range(k)]
        return xmod_struct_gated_exec_dajv_aggregate(
            s, x, exec_dajv,
            min_struct_agree=max(3, 2 * k // 3),
        )

    def _bsia(j: int):
        s = [bool(struct_test[i][j]) for i in range(k)]
        x = [bool(exec_test[i][j]) for i in range(k)]
        return bsia_aggregate(s, x, bsia_cal)

    def _bsia_temp(j: int):
        s = [bool(struct_test[i][j]) for i in range(k)]
        x = [bool(exec_test[i][j]) for i in range(k)]
        return bsia_aggregate(s, x, bsia_temp_cal)

    def _bsia_iso(j: int):
        s = [bool(struct_test[i][j]) for i in range(k)]
        x = [bool(exec_test[i][j]) for i in range(k)]
        return bsia_isotonic_aggregate(s, x, bsia_cal, iso_xs, iso_ys)

    def _bsia_ens(j: int):
        s = [bool(struct_test[i][j]) for i in range(k)]
        x = [bool(exec_test[i][j]) for i in range(k)]
        return bsia_ensemble_aggregate(
            s, x, bsia_cal, bsia_temp_cal, iso_xs, iso_ys,
        )

    return {
        "seed": seed, "n_test": len(test_idx), "k": k,
        "default_dajv":     _eval(_default),
        "xmod_dajv":        _eval(_xmod),
        "xmod_agreement":   _eval(_xmod_agree),
        "xmod_joint":       _eval(_xmod_joint),
        "xmod_struct_gated": _eval(_xmod_struct_gated),
        "bsia":             _eval(_bsia),
        "bsia_temp":        _eval(_bsia_temp),
        "bsia_iso":         _eval(_bsia_iso),
        "bsia_ensemble":    _eval(_bsia_ens),
        "bsia_temperature": bsia_temp_cal.temperature,
    }


def run_on_bench(bench: str | None, tag: str, seeds: list[int]) -> dict:
    aligned = align_extractor_caches(GROUP_B, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "bench": bench, "n": n, "skipped": "too few"}

    per_seed = [evaluate_one_seed(aligned, s) for s in seeds]

    def agg(method: str, metric: str):
        vals = [r[method][metric] for r in per_seed
                if r[method].get(metric) is not None]
        if not vals:
            return None
        m = sum(vals) / len(vals)
        denom = max(len(vals) - 1, 1)
        v = sum((x - m) ** 2 for x in vals) / denom if len(vals) > 1 else 0.0
        return {"mean": m, "std": v ** 0.5, "n": len(vals)}

    summary = {}
    for method in ("default_dajv", "xmod_dajv", "xmod_agreement", "xmod_joint", "xmod_struct_gated", "bsia", "bsia_temp", "bsia_iso", "bsia_ensemble"):
        summary[method] = {
            "coverage": agg(method, "coverage"),
            "precision": agg(method, "precision"),
            "ece": agg(method, "ece"),
            "brier": agg(method, "brier"),
            "rc_auc": agg(method, "rc_auc"),
        }
    return {
        "tag": tag, "bench": bench, "n_total": n,
        "k_extractors": len(aligned["extractor_ids"]),
        "seeds": seeds,
        "summary": summary,
        "per_seed": per_seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="xmod_dajv.json")
    args = ap.parse_args()

    seeds = list(range(1, 11))
    out = []
    for bench, tag in [("math175", "math175"), (None, "B_full")]:
        r = run_on_bench(bench, tag, seeds)
        out.append(r)
        if r.get("skipped"):
            continue
        print(f"\n=== {tag} k={r['k_extractors']} ===")
        for method in ("default_dajv", "xmod_dajv", "xmod_agreement", "xmod_joint", "xmod_struct_gated", "bsia", "bsia_temp", "bsia_iso", "bsia_ensemble"):
            s = r["summary"][method]
            def fmt(x):
                if x is None: return "n/a"
                return f"{x['mean']:.3f}±{x['std']:.3f}"
            print(f"  {method:14s} cov={fmt(s['coverage'])} "
                  f"prec={fmt(s['precision'])} ECE={fmt(s['ece'])} "
                  f"AUC={fmt(s['rc_auc'])}")
    save_artifact(out, ARTIFACTS / args.out)
    print(f"\nWrote {ARTIFACTS / args.out}")


if __name__ == "__main__":
    main()
