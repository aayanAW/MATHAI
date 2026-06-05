"""Copula-order ablation.

Compares DAJV with the second-order interaction term against an
independence-only variant (rho set to zero). Both share the same
per-extractor marginals fitted on the same calibration split.

Tests whether the dependency-aware term contributes meaningfully to
the headline ECE / coverage / precision relative to a Bayesian
naive-aggregator baseline (which is what DAJV reduces to when rho == 0).

Saves: artifacts/copula_ablation.json
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}

ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _eval_variant(
    cal: DajvCalibration,
    accept_test: list,
    correct_test: list,
    k: int,
) -> dict:
    confs: list[float] = []
    ys: list[bool] = []
    n_commit = n_commit_correct = 0
    n_test = len(correct_test)
    for j in range(n_test):
        votes_j = [bool(accept_test[i][j]) for i in range(k)]
        out = dajv_aggregate(votes_j, cal)
        if out.get("P_correct") is not None:
            confs.append(out["P_correct"])
            ys.append(bool(correct_test[j]))
        if out.get("recommendation") == "COMMIT":
            n_commit += 1
            if correct_test[j]:
                n_commit_correct += 1
    if not confs:
        return {"ece": None, "brier": None, "coverage": 0.0,
                "precision": None, "n_committed": 0}
    return {
        "ece": expected_calibration_error(confs, ys, n_bins=5),
        "brier": brier_score(confs, ys),
        "coverage": n_commit / n_test,
        "precision": (n_commit_correct / n_commit) if n_commit else None,
        "n_committed": n_commit,
        "n_correct": n_commit_correct,
    }


def _zero_rho(calibration: DajvCalibration) -> DajvCalibration:
    """Return a copy with all rho_pos / rho_neg interactions zeroed.

    This reduces the second-order copula to a Bayesian naive aggregator
    that still uses the calibrated per-extractor marginals."""
    k = len(calibration.extractor_ids)
    zero = [[0.0] * k for _ in range(k)]
    return DajvCalibration(
        extractor_ids=list(calibration.extractor_ids),
        pi_pos=list(calibration.pi_pos),
        pi_neg=list(calibration.pi_neg),
        rho_pos=copy.deepcopy(zero),
        rho_neg=copy.deepcopy(zero),
        prior_correct=calibration.prior_correct,
    )


def run_on_bench(bench: str | None, tag: str, seeds: list[int]) -> dict:
    aligned = align_extractor_caches(GROUP_B, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "bench": bench, "n": n, "skipped": "too few problems"}

    accept = aligned["accept"]
    solver_correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids)

    per_seed = []
    for seed in seeds:
        rng = random.Random(seed)
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

        full = DajvCalibration.fit(accept_cal, correct_cal, extractor_ids)
        indep = _zero_rho(full)

        per_seed.append({
            "seed": seed,
            "indep_only": _eval_variant(
                indep, accept_test, correct_test, k),
            "second_order": _eval_variant(
                full, accept_test, correct_test, k),
        })

    def agg(key1, key2):
        vals = [s[key1][key2] for s in per_seed
                if s[key1].get(key2) is not None]
        if not vals:
            return None
        m = sum(vals) / len(vals)
        # Sample std (Bessel-corrected); falls back to 0 for n < 2.
        denom = max(len(vals) - 1, 1) if len(vals) > 1 else 1
        v = sum((x - m) ** 2 for x in vals) / denom if len(vals) > 1 else 0.0
        return {"mean": m, "std": v ** 0.5, "n": len(vals)}

    summary = {
        "tag": tag,
        "bench": bench,
        "n_total": n,
        "k_extractors": k,
        "seeds": seeds,
        "per_seed": per_seed,
        "summary": {
            "indep_only": {
                "ece": agg("indep_only", "ece"),
                "brier": agg("indep_only", "brier"),
                "coverage": agg("indep_only", "coverage"),
                "precision": agg("indep_only", "precision"),
            },
            "second_order": {
                "ece": agg("second_order", "ece"),
                "brier": agg("second_order", "brier"),
                "coverage": agg("second_order", "coverage"),
                "precision": agg("second_order", "precision"),
            },
        },
    }
    print(f"\n=== bench={bench} ===")
    print(json.dumps(summary["summary"], indent=2))
    return summary


def main() -> None:
    seeds = list(range(1, 11))
    out = []
    for bench, tag in [("math175", "math175"), (None, "B_full")]:
        try:
            out.append(run_on_bench(bench, tag, seeds))
        except Exception as e:
            print(f"  ERROR on tag={tag}: {e}")
    save_artifact(out, ARTIFACTS / "copula_ablation.json")
    print(f"\nWrote {ARTIFACTS / 'copula_ablation.json'}")


if __name__ == "__main__":
    main()
