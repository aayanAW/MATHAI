"""H6 within-lab vs cross-lab significance with bootstrap CIs.

Computes:
  - Pairwise Cohen's κ on wrong-candidate subset, all available pairs.
  - Bootstrap 95% CI on median within-lab κ and median cross-lab κ.
  - Permutation-test p-value: is median(within) > median(cross) more
    extreme than the null distribution where lab labels are shuffled?
  - Per-bench breakdown.

Uses any extractor caches that have >= ``min-records`` records.
Saves: artifacts/h6_significance.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.dependency.kappa import cohen_kappa
from verifyensemble.utils.io import align_extractor_caches, save_artifact

RESULTS = HERE.parent / ".." / "MATHAI" / "results"

ENSEMBLE = {
    "E05_gpt_oss_120B":      ("openai",    RESULTS / "exp46_gptoss_extractor.json"),
    "E06_gpt_5_mini":        ("openai",    RESULTS / "exp50_gpt5_extractor.json"),
    "E07_claude_sonnet_4_6": ("anthropic", RESULTS / "exp47_claude_extractor.json"),
    "E09_qwen3_coder_480B":  ("alibaba",   RESULTS / "exp48_qwen3coder_extractor.json"),
    "E08S_gpt_4o":           ("openai",    RESULTS / "exp55_gpt4o_extractor.json"),
    "E04S_gpt_4_1":          ("openai",    RESULTS / "exp56_gpt41_extractor.json"),
    "E10S_gpt_5":            ("openai",    RESULTS / "exp57_gpt5_extractor.json"),
    "E01A_claude_opus_4_7":  ("anthropic", RESULTS / "exp58_opus47_extractor.json"),
    "E02A_claude_opus_4_6":  ("anthropic", RESULTS / "exp59_opus46_extractor.json"),
    "E03A_claude_haiku_4_5": ("anthropic", RESULTS / "exp61_haiku45_extractor.json"),
    "E13_llama_3_3_70B":     ("meta",      RESULTS / "exp62_llama33_70b_extractor.json"),
    "E14_qwen_3_235B":       ("alibaba",   RESULTS / "exp63_qwen3_235b_extractor.json"),
}

ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _bootstrap_median_ci(values: list[float], n_boot: int = 1000,
                          seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap percentile 95% CI for median."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(median(sample))
    boots.sort()
    return (median(values), boots[int(0.025 * n_boot)],
            boots[int(0.975 * n_boot) - 1])


def _permutation_test(within: list[float], cross: list[float],
                       n_perm: int = 10000, seed: int = 7) -> dict:
    """Single-sided permutation p-value: median(within) > median(cross).

    Null: lab labels are exchangeable. Permute the labels across all
    pairs and recompute the median-difference each time.
    """
    if not within or not cross:
        return {"observed_diff": None, "p_value": None,
                "n_perm": 0, "skipped": "empty group"}
    obs_diff = median(within) - median(cross)
    pooled = within + cross
    rng = random.Random(seed)
    n_w = len(within)
    n_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        w = pooled[:n_w]
        c = pooled[n_w:]
        diff = median(w) - median(c)
        if diff >= obs_diff:
            n_extreme += 1
    return {
        "observed_diff": obs_diff,
        "p_value": n_extreme / n_perm,
        "n_perm": n_perm,
    }


def run_on_bench(bench: str | None, tag: str,
                 extractor_paths: dict, labs: dict[str, str]) -> dict:
    aligned = align_extractor_caches(extractor_paths, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "bench": bench, "n": n, "skipped": "too few problems"}
    k = len(aligned["extractor_ids"])
    eids = aligned["extractor_ids"]

    wrong_idx = [j for j, c in enumerate(aligned["solver_correct"]) if not c]
    if len(wrong_idx) < 10:
        return {"tag": tag, "bench": bench, "n_wrong": len(wrong_idx),
                "skipped": "too few wrong candidates"}

    accept_w = [[bool(aligned["accept"][i][j]) for j in wrong_idx]
                for i in range(k)]

    within: list[float] = []
    cross: list[float] = []
    within_pairs: list[tuple[str, str]] = []
    cross_pairs: list[tuple[str, str]] = []
    for i in range(k):
        for j in range(i + 1, k):
            kij = cohen_kappa(accept_w[i], accept_w[j])
            if labs[eids[i]] == labs[eids[j]]:
                within.append(kij)
                within_pairs.append((eids[i], eids[j]))
            else:
                cross.append(kij)
                cross_pairs.append((eids[i], eids[j]))

    within_ci = _bootstrap_median_ci(within)
    cross_ci = _bootstrap_median_ci(cross)
    perm = _permutation_test(within, cross)

    return {
        "tag": tag,
        "bench": bench,
        "k_extractors": k,
        "extractor_ids": eids,
        "n_wrong": len(wrong_idx),
        "within_lab": {
            "n_pairs": len(within),
            "median": within_ci[0],
            "ci95_low": within_ci[1],
            "ci95_high": within_ci[2],
            "values": within,
            "pairs": within_pairs,
        },
        "cross_lab": {
            "n_pairs": len(cross),
            "median": cross_ci[0],
            "ci95_low": cross_ci[1],
            "ci95_high": cross_ci[2],
            "values": cross,
            "pairs": cross_pairs,
        },
        "permutation_test": perm,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-records", type=int, default=200)
    ap.add_argument("--out", default="h6_significance.json")
    args = ap.parse_args()

    avail = {}
    labs = {}
    for eid, (lab, path) in ENSEMBLE.items():
        if path.exists():
            try:
                d = json.load(open(path))
                if len(d.get("results", [])) >= args.min_records:
                    avail[eid] = path
                    labs[eid] = lab
            except Exception:
                pass
    print(f"Available: {list(avail)}")

    if len(avail) < 4:
        print("Need >= 4 extractors")
        return

    out = []
    for bench, tag in [("math175", "math175"), (None, "B_full")]:
        try:
            r = run_on_bench(bench, tag, avail, labs)
            out.append(r)
            if not r.get("skipped"):
                print(f"\n=== {tag} k={r['k_extractors']} n_wrong={r['n_wrong']} ===")
                w = r["within_lab"]
                c = r["cross_lab"]
                p = r["permutation_test"]
                wci = (f"[{w['ci95_low']:.3f}, {w['ci95_high']:.3f}]"
                       if w['ci95_low'] is not None else "n/a")
                cci = (f"[{c['ci95_low']:.3f}, {c['ci95_high']:.3f}]"
                       if c['ci95_low'] is not None else "n/a")
                obs = (f"{p['observed_diff']:.4f}"
                       if p['observed_diff'] is not None else "n/a")
                pval = (f"{p['p_value']:.4f}"
                        if p['p_value'] is not None else "n/a")
                print(f"  within  n={w['n_pairs']:2d} median={w['median']:.3f} 95% CI {wci}")
                print(f"  cross   n={c['n_pairs']:2d} median={c['median']:.3f} 95% CI {cci}")
                print(f"  obs diff median(W) - median(C) = {obs}")
                print(f"  permutation single-sided p-value: {pval}")
        except Exception as e:
            print(f"  ERROR {tag}: {e}")
    save_artifact(out, ARTIFACTS / args.out)
    print(f"\nWrote {ARTIFACTS / args.out}")


if __name__ == "__main__":
    main()
