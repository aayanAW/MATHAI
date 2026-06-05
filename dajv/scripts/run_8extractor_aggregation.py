"""8-extractor DAJV aggregation on the scale-up cache.

After the 4 new extractors finish (E08S gpt-4o, E04S gpt-4.1, E10S gpt-5,
E12 Gemini-2.5-Flash), this script builds the 8-extractor aligned cache
and runs DAJV + naive baselines + VERGE proxy.

Within-lab structure on the 8-extractor set:
  OpenAI: E05 gpt-oss-120B, E06 gpt-5-mini, E08S gpt-4o,
          E04S gpt-4.1, E10S gpt-5  -> 5 extractors, 10 within-lab pairs
  Anthropic: E07 claude-sonnet-4-6  -> 1 extractor
  Alibaba: E09 qwen3-coder-480B     -> 1 extractor
  Google: E12 gemini-2.5-flash      -> 1 extractor

This unlocks H6 (within-lab vs cross-lab dependency) with 10 within-lab
pairs (all OpenAI) and 18 cross-lab pairs.

Saves: artifacts/aggregation_8extractor.json
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
from verifyensemble.aggregate.verge_proxy import verge_proxy_aggregate
from verifyensemble.dependency.kappa import cohen_kappa
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.utils.io import align_extractor_caches, save_artifact

RESULTS = HERE.parent / ".." / "MATHAI" / "results"

# 8-extractor ensemble; lab tag per extractor for H6 analysis
ENSEMBLE = {
    "E05_gpt_oss_120B":      ("openai", RESULTS / "exp46_gptoss_extractor.json"),
    "E06_gpt_5_mini":        ("openai", RESULTS / "exp50_gpt5_extractor.json"),
    "E07_claude_sonnet_4_6": ("anthropic", RESULTS / "exp47_claude_extractor.json"),
    "E09_qwen3_coder_480B":  ("alibaba",  RESULTS / "exp48_qwen3coder_extractor.json"),
    "E08S_gpt_4o":           ("openai",   RESULTS / "exp55_gpt4o_extractor.json"),
    "E04S_gpt_4_1":          ("openai",   RESULTS / "exp56_gpt41_extractor.json"),
    "E10S_gpt_5":            ("openai",   RESULTS / "exp57_gpt5_extractor.json"),
    "E01A_claude_opus_4_7":  ("anthropic", RESULTS / "exp58_opus47_extractor.json"),
    "E02A_claude_opus_4_6":  ("anthropic", RESULTS / "exp59_opus46_extractor.json"),
    "E03A_claude_haiku_4_5": ("anthropic", RESULTS / "exp61_haiku45_extractor.json"),
    "E13_llama_3_3_70B":     ("meta",      RESULTS / "exp62_llama33_70b_extractor.json"),
    "E14_qwen_3_235B":       ("alibaba",   RESULTS / "exp63_qwen3_235b_extractor.json"),
    # E12_gemini_2_5_flash dropped: free-tier daily cap hit at 20 RPM
    # AND thinking-budget bug truncated most responses (parse_error
    # rate 95%). Re-enable when billing-enabled key + thinking_budget=0
    # are both confirmed working.
}

ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
SEED = 42


def _check_caches() -> dict[str, int]:
    """Return record-count per available cache. Skip missing files."""
    counts: dict[str, int] = {}
    for eid, (_, path) in ENSEMBLE.items():
        if path.exists():
            try:
                d = json.load(open(path))
                counts[eid] = len(d.get("results", []))
            except Exception as e:
                counts[eid] = -1
                print(f"  {eid}: load error: {e}")
        else:
            counts[eid] = 0
    return counts


def _within_lab_pairs(extractor_ids: list[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (within_lab_pairs, cross_lab_pairs)."""
    labs = [ENSEMBLE[e][0] for e in extractor_ids]
    within: list[tuple[int, int]] = []
    cross: list[tuple[int, int]] = []
    k = len(extractor_ids)
    for i in range(k):
        for j in range(i + 1, k):
            if labs[i] == labs[j]:
                within.append((i, j))
            else:
                cross.append((i, j))
    return within, cross


def run_on_bench(bench: str | None, tag: str,
                 extractor_paths: dict) -> dict:
    aligned = align_extractor_caches(extractor_paths, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "bench": bench, "n": n, "skipped": "too few"}

    k = len(aligned["extractor_ids"])
    extractor_ids = aligned["extractor_ids"]

    # Dependency on the wrong-candidate subset
    wrong_idx = [j for j, c in enumerate(aligned["solver_correct"]) if not c]
    if len(wrong_idx) >= 10:
        accept_w = [[bool(aligned["accept"][i][j]) for j in wrong_idx]
                    for i in range(k)]
        within_pairs, cross_pairs = _within_lab_pairs(extractor_ids)
        within_kappa = [cohen_kappa(accept_w[i], accept_w[j])
                        for i, j in within_pairs]
        cross_kappa = [cohen_kappa(accept_w[i], accept_w[j])
                       for i, j in cross_pairs]

        def med(vals: list[float]) -> float | None:
            if not vals:
                return None
            s = sorted(vals)
            return s[len(s)//2] if len(s) % 2 else (s[len(s)//2-1]+s[len(s)//2])/2

        h6 = {
            "n_within_pairs": len(within_pairs),
            "n_cross_pairs": len(cross_pairs),
            "within_lab_kappa_median": med(within_kappa),
            "cross_lab_kappa_median": med(cross_kappa),
            "within_lab_kappa_values": within_kappa,
            "cross_lab_kappa_values": cross_kappa,
            "pair_assignments": {
                "within_pairs": [(extractor_ids[i], extractor_ids[j])
                                 for i, j in within_pairs],
                "cross_pairs":  [(extractor_ids[i], extractor_ids[j])
                                 for i, j in cross_pairs],
            },
        }
    else:
        h6 = {"skipped": "too few wrong candidates"}

    # Calibration / test split on the full bench
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

    cal = DajvCalibration.fit(accept_cal, correct_cal, extractor_ids)

    rows = []
    for j in range(len(test_idx)):
        votes_j = [bool(accept_test[i][j]) for i in range(k)]
        cls_j = [classification_test[i][j] for i in range(k)]
        cv_j = [cv_test[i][j] for i in range(k)]
        rows.append({
            "correct": bool(correct_test[j]),
            "naive_u": naive_unanimous(votes_j),
            "naive_m": naive_majority(votes_j),
            "dajv": dajv_aggregate(votes_j, cal),
            "verge_proxy": verge_proxy_aggregate(
                votes_j, cls_j, cv_j,
                min_agree=max(3, k * 3 // 4)),
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

    def calib(key: str) -> dict:
        items = [(r[key]["P_correct"], r["correct"])
                 for r in rows if r[key].get("P_correct") is not None]
        if not items:
            return {"ece": None, "brier": None}
        confs, ys = zip(*items)
        return {
            "ece": expected_calibration_error(confs, ys, n_bins=5),
            "brier": brier_score(confs, ys),
        }

    out = {
        "tag": tag,
        "bench": bench,
        "k_extractors": k,
        "extractor_ids": extractor_ids,
        "n_total": n,
        "n_wrong": len(wrong_idx),
        "n_cal": cal_n,
        "n_test": len(test_idx),
        "h6_within_vs_cross_lab": h6,
        "operating_points": {
            "naive_unanimous":      head("naive_u", {"COMMIT"}),
            "naive_majority":       head("naive_m", {"COMMIT"}),
            "dajv":                 head("dajv", {"COMMIT"}),
            "verge_proxy_full":     head("verge_proxy", {"COMMIT"}),
            "verge_proxy_with_mcs": head("verge_proxy", {"COMMIT", "COMMIT_MCS"}),
        },
        "calibration": {
            "naive_unanimous": calib("naive_u"),
            "dajv": calib("dajv"),
        },
    }
    print(f"\n=== {tag} k={k} n_test={len(test_idx)} ===")
    if not isinstance(h6, dict) or "skipped" not in h6:
        wm = h6["within_lab_kappa_median"]
        cm = h6["cross_lab_kappa_median"]
        wm_s = f"{wm:.3f}" if wm is not None else "  ---"
        cm_s = f"{cm:.3f}" if cm is not None else "  ---"
        print(f"  H6: within-lab κ median={wm_s}  cross-lab κ median={cm_s}  "
              f"(n_within={h6['n_within_pairs']}, "
              f"n_cross={h6['n_cross_pairs']})")
    for label in ("naive_unanimous", "naive_majority", "dajv",
                  "verge_proxy_full", "verge_proxy_with_mcs"):
        d = out["operating_points"][label]
        prec = d["precision"]
        ps = f"{prec:.3f}" if prec is not None else "  ---"
        print(f"  {label:25s} cov={d['coverage']:.3f} prec={ps}")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-records", type=int, default=200,
                    help="drop caches with fewer than this many records")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="exclude these extractor ids")
    ap.add_argument("--out", default="aggregation_8extractor.json")
    args = ap.parse_args()

    counts = _check_caches()
    print("=== cache record counts ===")
    avail: dict = {}
    for eid, n in counts.items():
        marker = " (NEW)" if eid not in {
            "E05_gpt_oss_120B", "E06_gpt_5_mini",
            "E07_claude_sonnet_4_6", "E09_qwen3_coder_480B",
        } else ""
        status = ("OK" if n >= args.min_records
                  else ("partial" if n > 0 else "missing"))
        print(f"  {eid}: {n} records [{status}]{marker}")
        if eid in args.exclude:
            print("    -> excluded by --exclude")
            continue
        if n >= args.min_records:
            avail[eid] = ENSEMBLE[eid][1]
    print(f"\nCandidates available: {len(avail)} of {len(ENSEMBLE)}")
    if len(avail) < 4:
        print("Need at least 4 caches; exiting.")
        return

    results = []
    for bench, tag in [("math175", "math175"), ("aime", "aime"),
                       ("cleanmath", "cleanmath"), (None, "full")]:
        try:
            results.append(run_on_bench(bench, tag, avail))
        except Exception as e:
            print(f"  ERROR {tag}: {e}")
    save_artifact(results, ARTIFACTS / args.out)
    print(f"\nWrote {ARTIFACTS / args.out}")


if __name__ == "__main__":
    main()
