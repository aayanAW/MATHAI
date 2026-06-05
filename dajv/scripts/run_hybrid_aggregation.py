"""Hybrid aggregation: pure-LLM vs hybrid struct×exec DAJV.

Builds four DAJV calibrations on Group B and compares:

  Variant         | k | Signals fed in
  ----------------+---+--------------------------------------------------
  llm-accept (default DAJV)   | 4 | accept[i]   = work && cand_verdict
  structural only             | 4 | struct[i]   = classification=='working'
  executable only             | 4 | exec[i]     = candidate_verdict is True
  HYBRID (struct + exec)      | 8 | struct[0..3] ++ exec[0..3]

Hypothesis (from the cross-modality independence finding in
``run_hybrid_modality.py``): the 8-extractor hybrid ensemble carries
additional axes of independence and so achieves better calibration
(ECE, Brier) at the same precision than any of the 4-extractor pure
variants.

Saves: artifacts/hybrid_aggregation.json
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_unanimous
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


def _build_signals(aligned: dict) -> dict[str, tuple[list[list[bool]], list[str]]]:
    """Return {variant_name: (signal_matrix, signal_ids)}."""
    k = len(aligned["extractor_ids"])
    eids = aligned["extractor_ids"]
    accept = aligned["accept"]
    classification = aligned["classification"]
    cand_verdict = aligned["candidate_verdict"]
    n = len(aligned["problem_ids"])

    structural = [[(classification[i][j] == "working")
                   for j in range(n)]
                  for i in range(k)]
    executable = [[(cand_verdict[i][j] is True)
                   for j in range(n)]
                  for i in range(k)]

    variants: dict[str, tuple[list[list[bool]], list[str]]] = {
        "llm_accept": (
            [[bool(accept[i][j]) for j in range(n)] for i in range(k)],
            [f"acc_{e}" for e in eids],
        ),
        "structural_only": (
            structural,
            [f"struct_{e}" for e in eids],
        ),
        "executable_only": (
            executable,
            [f"exec_{e}" for e in eids],
        ),
        "hybrid_8": (
            structural + executable,
            [f"struct_{e}" for e in eids] + [f"exec_{e}" for e in eids],
        ),
    }
    return variants


def _evaluate(signals: list[list[bool]], signal_ids: list[str],
              correct: list[bool], cal_frac: float, seed: int) -> dict:
    n = len(correct)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    cal_n = int(cal_frac * n)
    cal_idx = indices[:cal_n]
    test_idx = indices[cal_n:]
    k = len(signals)

    accept_cal = [[signals[i][j] for j in cal_idx] for i in range(k)]
    accept_test = [[signals[i][j] for j in test_idx] for i in range(k)]
    correct_cal = [correct[j] for j in cal_idx]
    correct_test = [correct[j] for j in test_idx]

    try:
        cal = DajvCalibration.fit(accept_cal, correct_cal, signal_ids)
    except Exception as e:
        return {"error": str(e)}

    confs: list[float] = []
    ys: list[bool] = []
    n_commit = n_commit_correct = 0
    n_naive = n_naive_correct = 0
    for j in range(len(test_idx)):
        votes_j = [bool(accept_test[i][j]) for i in range(k)]
        out_d = dajv_aggregate(votes_j, cal)
        out_n = naive_unanimous(votes_j)
        if out_d.get("P_correct") is not None:
            confs.append(out_d["P_correct"])
            ys.append(bool(correct_test[j]))
        if out_d.get("recommendation") == "COMMIT":
            n_commit += 1
            if correct_test[j]:
                n_commit_correct += 1
        if out_n.get("recommendation") == "COMMIT":
            n_naive += 1
            if correct_test[j]:
                n_naive_correct += 1

    n_test = len(test_idx)
    return {
        "k": k,
        "n_cal": cal_n,
        "n_test": n_test,
        "dajv": {
            "coverage": n_commit / n_test if n_test else 0.0,
            "precision": (n_commit_correct / n_commit) if n_commit else None,
            "n_committed": n_commit,
            "n_correct": n_commit_correct,
            "ece": expected_calibration_error(confs, ys, n_bins=5) if confs else None,
            "brier": brier_score(confs, ys) if confs else None,
        },
        "naive_unanimous": {
            "coverage": n_naive / n_test if n_test else 0.0,
            "precision": (n_naive_correct / n_naive) if n_naive else None,
            "n_committed": n_naive,
        },
    }


def _agg(per_seed: list[dict], metric_key: str,
         method: str = "dajv") -> dict | None:
    vals = [s[method][metric_key] for s in per_seed
            if isinstance(s, dict) and method in s
            and s[method].get(metric_key) is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    denom = max(len(vals) - 1, 1) if len(vals) > 1 else 1
    v = sum((x - m) ** 2 for x in vals) / denom if len(vals) > 1 else 0.0
    return {"mean": m, "std": v ** 0.5, "n": len(vals)}


def run_on_bench(bench: str | None, tag: str, seeds: list[int]) -> dict:
    aligned = align_extractor_caches(GROUP_B, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "bench": bench, "n": n, "skipped": "too few"}

    correct = [bool(c) for c in aligned["solver_correct"]]
    variants = _build_signals(aligned)

    out: dict = {"tag": tag, "bench": bench, "n_total": n, "variants": {}}
    for variant_name, (signals, signal_ids) in variants.items():
        per_seed = []
        for seed in seeds:
            per_seed.append(_evaluate(signals, signal_ids, correct, 0.7, seed))

        out["variants"][variant_name] = {
            "n_signals": len(signals),
            "dajv": {
                "coverage": _agg(per_seed, "coverage", "dajv"),
                "precision": _agg(per_seed, "precision", "dajv"),
                "ece": _agg(per_seed, "ece", "dajv"),
                "brier": _agg(per_seed, "brier", "dajv"),
            },
            "naive_unanimous": {
                "coverage": _agg(per_seed, "coverage", "naive_unanimous"),
                "precision": _agg(per_seed, "precision", "naive_unanimous"),
            },
            "per_seed": per_seed,
        }
    print(f"\n=== {tag} ===")
    for vname, vdat in out["variants"].items():
        d = vdat["dajv"]
        ece = d["ece"]["mean"] if d["ece"] else None
        cov = d["coverage"]["mean"] if d["coverage"] else None
        prec = d["precision"]["mean"] if d["precision"] else None
        brier = d["brier"]["mean"] if d["brier"] else None

        def fmt(x: float | None) -> str:
            return f"{x:.3f}" if x is not None else " --- "

        print(
            f"  {vname:18s} k={vdat['n_signals']:2d}  "
            f"cov={fmt(cov)} prec={fmt(prec)} "
            f"ECE={fmt(ece)} Brier={fmt(brier)}"
        )
    return out


def main() -> None:
    seeds = list(range(1, 11))
    out = []
    for bench, tag in [("math175", "math175"), (None, "B_full")]:
        out.append(run_on_bench(bench, tag, seeds))
    save_artifact(out, ARTIFACTS / "hybrid_aggregation.json")
    print(f"\nWrote {ARTIFACTS / 'hybrid_aggregation.json'}")


if __name__ == "__main__":
    main()
