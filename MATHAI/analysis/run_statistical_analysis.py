#!/usr/bin/env python3
"""
Comprehensive statistical analysis for ExeVer paper.

Reads result files from experiments 5, 7, 8, 11 and computes:
1. Confidence calibration analysis (verdict -> accuracy)
2. Bootstrap confidence intervals for all main accuracy numbers
3. Coverage-conditioned accuracy
4. Error type analysis (by subject and level)
5. Echo chamber detailed analysis

Outputs results to stdout and saves to results/statistical_analysis.json.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/aayanalwani/MATHAI")
RESULTS = BASE / "results"

EXP5_PATH = RESULTS / "exp5_math500_full.json"
EXP7_PATH = RESULTS / "exp7_symcode_baseline.json"
EXP8_PATH = RESULTS / "exp8_selfcorrect.json"
EXP11_PATH = RESULTS / "exp11_llm_judge.json"
OUTPUT_PATH = RESULTS / "statistical_analysis.json"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("ExeVer Statistical Analysis")
print("=" * 70)

with open(EXP5_PATH) as f:
    exp5 = json.load(f)
with open(EXP7_PATH) as f:
    exp7 = json.load(f)
with open(EXP8_PATH) as f:
    exp8 = json.load(f)
with open(EXP11_PATH) as f:
    exp11 = json.load(f)

exever_results = exp5["exever_results"]  # 500 per-problem dicts
selfcorrect_results = exp8["results"]     # 500 per-problem dicts
symcode_results = exp7["results"]         # 500 per-problem dicts
judge_sampled = exp11["sampled_results"]  # 500 per-problem dicts

n = len(exever_results)
assert n == 500, f"Expected 500 problems, got {n}"

# Build lookup by ID for cross-referencing
exever_by_id = {r["id"]: r for r in exever_results}
selfcorrect_by_id = {r["id"]: r for r in selfcorrect_results}
symcode_by_id = {r["id"]: r for r in symcode_results}
judge_by_id = {r["id"]: r for r in judge_sampled}

# Seed for reproducibility
rng = np.random.default_rng(42)

output: dict[str, Any] = {}

# =========================================================================
# 1. CONFIDENCE CALIBRATION ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("1. CONFIDENCE CALIBRATION ANALYSIS")
print("=" * 70)

# Group problems by verdict
verdict_groups: dict[str, list[dict]] = defaultdict(list)
for r in exever_results:
    verdict_groups[r["verdict"]].append(r)

# Define fallback verdicts
FALLBACK_VERDICTS = {"SYNTAX_ERROR", "RUNTIME_ERROR", "TIMEOUT", "FAIL_STEP_-1"}

# Compute accuracy for each verdict group
calibration: dict[str, dict[str, Any]] = {}
for verdict, items in sorted(verdict_groups.items(), key=lambda x: -len(x[1])):
    correct = sum(1 for r in items if r["answer_correct"])
    total = len(items)
    acc = correct / total if total > 0 else 0.0
    calibration[verdict] = {
        "n": total,
        "n_correct": correct,
        "accuracy": round(acc, 4),
    }
    print(f"  {verdict:25s}: {correct:3d}/{total:3d} = {acc:.1%}")

# Aggregate groups
verified_items = verdict_groups["ALL_PASS"]
repaired_items = verdict_groups["REPAIRED"] + verdict_groups.get("REPAIRED_UNVERIFIED", [])
fallback_items = []
for v in FALLBACK_VERDICTS:
    fallback_items.extend(verdict_groups.get(v, []))

def group_accuracy(items: list[dict]) -> tuple[int, int, float]:
    correct = sum(1 for r in items if r["answer_correct"])
    total = len(items)
    return correct, total, correct / total if total > 0 else 0.0

print("\n  --- Aggregated Groups ---")
for name, items in [("ALL_PASS (verified)", verified_items),
                     ("REPAIRED (any)", repaired_items),
                     ("FALLBACK (errors)", fallback_items)]:
    c, t, a = group_accuracy(items)
    print(f"  {name:30s}: {c:3d}/{t:3d} = {a:.1%}")

calibration_summary = {
    "ALL_PASS": {"n": len(verified_items), **dict(zip(["n_correct", "total", "accuracy"], group_accuracy(verified_items)))},
    "REPAIRED": {"n": len(repaired_items), **dict(zip(["n_correct", "total", "accuracy"], group_accuracy(repaired_items)))},
    "FALLBACK": {"n": len(fallback_items), **dict(zip(["n_correct", "total", "accuracy"], group_accuracy(fallback_items)))},
}

# Is ALL_PASS a reliable signal?
ap_acc = group_accuracy(verified_items)[2]
fb_acc = group_accuracy(fallback_items)[2]
print(f"\n  Verdict as confidence signal:")
print(f"    ALL_PASS accuracy: {ap_acc:.1%}  vs  FALLBACK accuracy: {fb_acc:.1%}")
print(f"    Difference: {(ap_acc - fb_acc)*100:+.1f}pp")
print(f"    -> {'YES' if ap_acc > fb_acc + 0.05 else 'WEAK/NO'}: ALL_PASS is "
      f"{'a reliable' if ap_acc > fb_acc + 0.05 else 'NOT a strong'} confidence signal")

output["confidence_calibration"] = {
    "per_verdict": calibration,
    "aggregated": calibration_summary,
    "all_pass_vs_fallback_pp": round((ap_acc - fb_acc) * 100, 2),
}

# =========================================================================
# 2. BOOTSTRAP CONFIDENCE INTERVALS
# =========================================================================
print("\n" + "=" * 70)
print("2. BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)")
print("=" * 70)

N_BOOT = 10_000

def bootstrap_ci(correct_array: np.ndarray, n_boot: int = N_BOOT,
                 alpha: float = 0.05) -> dict[str, float]:
    """Compute bootstrap CI for accuracy from binary correct/incorrect array."""
    n = len(correct_array)
    observed = correct_array.mean()
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = correct_array[idx].mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    se = boot_means.std()
    return {
        "observed": round(float(observed), 4),
        "ci_lower": round(float(lo), 4),
        "ci_upper": round(float(hi), 4),
        "se": round(float(se), 4),
        "width_pp": round(float((hi - lo) * 100), 2),
    }


# Build binary arrays for each method
# IMPORTANT: answer_correct in exever_results reflects the FINAL pipeline answer,
# not the original greedy. For REPAIRED problems, the answer may differ from greedy.
# Use selfcorrect initial_correct as ground truth for greedy correctness.
greedy_correct = np.array(
    [selfcorrect_by_id[r["id"]]["initial_correct"] for r in exever_results], dtype=float
)
# ExeVer correctness = answer_correct in exever_results (final pipeline output)
exever_correct = np.array([r["answer_correct"] for r in exever_results], dtype=float)
majority4_correct = np.array([judge_by_id[r["id"]]["majority4_correct"] for r in exever_results], dtype=float)
best4_correct = np.array([judge_by_id[r["id"]]["best4_correct"] for r in exever_results], dtype=float)

# Verify accuracy matches expectations
print(f"\n  Accuracy verification:")
print(f"    Greedy:    {greedy_correct.mean():.3f} (expected 0.832)")
print(f"    ExeVer:    {exever_correct.mean():.3f} (expected 0.834)")
print(f"    Majority@4: {majority4_correct.mean():.3f} (expected 0.848)")
print(f"    Best-of-4: {best4_correct.mean():.3f} (expected 0.888)")

# Compute CIs
methods = {
    "greedy_cot": greedy_correct,
    "exever": exever_correct,
    "majority_4": majority4_correct,
    "best_of_4": best4_correct,
}

bootstrap_results: dict[str, dict] = {}
print(f"\n  {'Method':20s} {'Accuracy':>10s} {'95% CI':>20s} {'Width':>8s}")
print(f"  {'-'*20} {'-'*10} {'-'*20} {'-'*8}")
for name, arr in methods.items():
    ci = bootstrap_ci(arr)
    bootstrap_results[name] = ci
    print(f"  {name:20s} {ci['observed']:10.1%} [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}] {ci['width_pp']:7.1f}pp")

# Check overlap: ExeVer vs Greedy
g_ci = bootstrap_results["greedy_cot"]
e_ci = bootstrap_results["exever"]
overlap = e_ci["ci_lower"] <= g_ci["ci_upper"] and g_ci["ci_lower"] <= e_ci["ci_upper"]
print(f"\n  ExeVer vs Greedy CIs overlap: {overlap}")
print(f"    Greedy CI:  [{g_ci['ci_lower']:.4f}, {g_ci['ci_upper']:.4f}]")
print(f"    ExeVer CI:  [{e_ci['ci_lower']:.4f}, {e_ci['ci_upper']:.4f}]")
print(f"    -> The +0.2pp difference is {'NOT statistically significant' if overlap else 'statistically significant'}")

# Paired bootstrap: direct difference ExeVer - Greedy
diff = exever_correct - greedy_correct
boot_diffs = np.empty(N_BOOT)
for i in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    boot_diffs[i] = diff[idx].mean()

diff_lo = np.percentile(boot_diffs, 2.5)
diff_hi = np.percentile(boot_diffs, 97.5)
diff_mean = diff.mean()
p_value_positive = (boot_diffs <= 0).mean()  # proportion of bootstrap diffs <= 0

print(f"\n  Paired bootstrap: ExeVer - Greedy")
print(f"    Observed difference: {diff_mean:+.4f} ({diff_mean*100:+.2f}pp)")
print(f"    95% CI: [{diff_lo:.4f}, {diff_hi:.4f}] ({diff_lo*100:.2f}pp to {diff_hi*100:+.2f}pp)")
print(f"    P(diff <= 0): {p_value_positive:.4f}")
print(f"    -> {'Significant' if diff_lo > 0 else 'NOT significant'} at alpha=0.05")

# Also compute paired bootstrap for ExeVer vs Majority@4
diff_maj = exever_correct - majority4_correct
boot_diffs_maj = np.empty(N_BOOT)
for i in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    boot_diffs_maj[i] = diff_maj[idx].mean()

dm_lo = np.percentile(boot_diffs_maj, 2.5)
dm_hi = np.percentile(boot_diffs_maj, 97.5)
dm_mean = diff_maj.mean()

print(f"\n  Paired bootstrap: ExeVer - Majority@4")
print(f"    Observed difference: {dm_mean:+.4f} ({dm_mean*100:+.2f}pp)")
print(f"    95% CI: [{dm_lo:.4f}, {dm_hi:.4f}] ({dm_lo*100:.2f}pp to {dm_hi*100:+.2f}pp)")
print(f"    -> ExeVer {'significantly' if dm_hi < 0 else 'does NOT significantly'} "
      f"{'underperforms' if dm_mean < 0 else 'outperforms'} Majority@4")

bootstrap_results["exever_minus_greedy"] = {
    "observed_diff": round(float(diff_mean), 4),
    "ci_lower": round(float(diff_lo), 4),
    "ci_upper": round(float(diff_hi), 4),
    "p_value_positive": round(float(p_value_positive), 4),
    "significant": bool(diff_lo > 0),
}
bootstrap_results["exever_minus_majority4"] = {
    "observed_diff": round(float(dm_mean), 4),
    "ci_lower": round(float(dm_lo), 4),
    "ci_upper": round(float(dm_hi), 4),
    "significant": bool(dm_hi < 0),
}

output["bootstrap_confidence_intervals"] = bootstrap_results

# =========================================================================
# 3. COVERAGE-CONDITIONED ACCURACY
# =========================================================================
print("\n" + "=" * 70)
print("3. COVERAGE-CONDITIONED ACCURACY")
print("=" * 70)

# Verification worked = ALL_PASS or REPAIRED or REPAIRED_UNVERIFIED
VERIFIED_VERDICTS = {"ALL_PASS", "REPAIRED", "REPAIRED_UNVERIFIED"}
verified_ids = {r["id"] for r in exever_results if r["verdict"] in VERIFIED_VERDICTS}
fallback_ids = {r["id"] for r in exever_results if r["verdict"] in FALLBACK_VERDICTS}

print(f"\n  Verification worked: {len(verified_ids)} problems")
print(f"  Fallback (errors):   {len(fallback_ids)} problems")
print(f"  Other (FAIL_STEP):   {n - len(verified_ids) - len(fallback_ids)} problems")

# ExeVer accuracy on each subset
exever_verified = [r for r in exever_results if r["id"] in verified_ids]
exever_fallback = [r for r in exever_results if r["id"] in fallback_ids]

ev_acc = sum(r["answer_correct"] for r in exever_verified) / len(exever_verified)
ef_acc = sum(r["answer_correct"] for r in exever_fallback) / len(exever_fallback)

# Greedy accuracy on the SAME subsets (to check if verification-eligible problems are easier)
greedy_on_verified = sum(r["answer_correct"] for r in exever_verified) / len(exever_verified)
greedy_on_fallback = sum(r["answer_correct"] for r in exever_fallback) / len(exever_fallback)

# For greedy, answer_correct IS the greedy correctness, so this is valid for fallback.
# For verified problems, greedy_correct = answer_correct in the exever record
# (since ALL_PASS keeps greedy answer).
# NOTE: For REPAIRED problems, answer_correct might reflect the repaired answer, not greedy.
# Let's cross-reference with selfcorrect data which has initial_correct = greedy correctness.
greedy_on_verified_v2 = sum(
    selfcorrect_by_id[r["id"]]["initial_correct"]
    for r in exever_verified
) / len(exever_verified)
greedy_on_fallback_v2 = sum(
    selfcorrect_by_id[r["id"]]["initial_correct"]
    for r in exever_fallback
) / len(exever_fallback)

print(f"\n  {'Subset':35s} {'ExeVer':>10s} {'Greedy':>10s} {'N':>6s}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*6}")
print(f"  {'Verification worked':35s} {ev_acc:10.1%} {greedy_on_verified_v2:10.1%} {len(exever_verified):6d}")
print(f"  {'Fallback (errors)':35s} {ef_acc:10.1%} {greedy_on_fallback_v2:10.1%} {len(exever_fallback):6d}")

# Is verification eligibility correlated with problem difficulty?
print(f"\n  Selection bias check:")
print(f"    Greedy accuracy on verification-eligible: {greedy_on_verified_v2:.1%}")
print(f"    Greedy accuracy on fallback:              {greedy_on_fallback_v2:.1%}")
print(f"    Difference: {(greedy_on_verified_v2 - greedy_on_fallback_v2)*100:+.1f}pp")
if greedy_on_verified_v2 > greedy_on_fallback_v2 + 0.03:
    print(f"    -> YES: Verification-eligible problems are inherently easier.")
    print(f"       This means ExeVer's effective accuracy is partially a selection effect.")
else:
    print(f"    -> NO: Verification-eligible and fallback problems are of similar difficulty.")

# Bootstrap CI for coverage-conditioned accuracy difference
verified_greedy_arr = np.array([selfcorrect_by_id[r["id"]]["initial_correct"] for r in exever_verified], dtype=float)
fallback_greedy_arr = np.array([selfcorrect_by_id[r["id"]]["initial_correct"] for r in exever_fallback], dtype=float)

coverage_cond = {
    "verified": {
        "n": len(exever_verified),
        "exever_accuracy": round(ev_acc, 4),
        "greedy_accuracy": round(greedy_on_verified_v2, 4),
    },
    "fallback": {
        "n": len(exever_fallback),
        "exever_accuracy": round(ef_acc, 4),
        "greedy_accuracy": round(greedy_on_fallback_v2, 4),
    },
    "greedy_diff_pp": round((greedy_on_verified_v2 - greedy_on_fallback_v2) * 100, 2),
    "selection_bias": greedy_on_verified_v2 > greedy_on_fallback_v2 + 0.03,
}

# By level breakdown for verified vs fallback
print(f"\n  Coverage-conditioned by level:")
print(f"  {'Level':>5s} {'Verif N':>8s} {'Verif Acc':>10s} {'Fall N':>8s} {'Fall Acc':>10s} {'Greedy(V)':>10s} {'Greedy(F)':>10s}")
print(f"  {'-'*5} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
level_breakdown = {}
for level in range(1, 6):
    v_items = [r for r in exever_verified if r["level"] == level]
    f_items = [r for r in exever_fallback if r["level"] == level]
    v_acc = sum(r["answer_correct"] for r in v_items) / len(v_items) if v_items else 0
    f_acc = sum(r["answer_correct"] for r in f_items) / len(f_items) if f_items else 0
    gv = sum(selfcorrect_by_id[r["id"]]["initial_correct"] for r in v_items) / len(v_items) if v_items else 0
    gf = sum(selfcorrect_by_id[r["id"]]["initial_correct"] for r in f_items) / len(f_items) if f_items else 0
    print(f"  {level:5d} {len(v_items):8d} {v_acc:10.1%} {len(f_items):8d} {f_acc:10.1%} {gv:10.1%} {gf:10.1%}")
    level_breakdown[str(level)] = {
        "verified_n": len(v_items), "verified_acc": round(v_acc, 4),
        "fallback_n": len(f_items), "fallback_acc": round(f_acc, 4),
        "greedy_verified": round(gv, 4), "greedy_fallback": round(gf, 4),
    }

coverage_cond["by_level"] = level_breakdown
output["coverage_conditioned_accuracy"] = coverage_cond

# =========================================================================
# 4. ERROR TYPE ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("4. ERROR TYPE ANALYSIS")
print("=" * 70)

error_types = {}
for error_verdict in ["SYNTAX_ERROR", "RUNTIME_ERROR", "TIMEOUT", "FAIL_STEP_-1"]:
    items = verdict_groups.get(error_verdict, [])
    if not items:
        continue

    # By subject
    by_subject: dict[str, int] = Counter()
    by_level: dict[int, int] = Counter()
    by_subj_level: dict[str, int] = Counter()
    correct_count = 0

    for r in items:
        subj = r["type"]
        lvl = r["level"]
        by_subject[subj] += 1
        by_level[lvl] += 1
        by_subj_level[f"{subj}_L{lvl}"] += 1
        if r["answer_correct"]:
            correct_count += 1

    print(f"\n  {error_verdict} (n={len(items)}, greedy acc={correct_count/len(items):.1%})")
    print(f"    By subject:")
    for subj, cnt in sorted(by_subject.items(), key=lambda x: -x[1]):
        # Total problems of this subject for rate computation
        total_subj = sum(1 for r in exever_results if r["type"] == subj)
        print(f"      {subj:25s}: {cnt:3d}/{total_subj:3d} ({cnt/total_subj:.1%} error rate)")

    print(f"    By level:")
    for lvl in sorted(by_level.keys()):
        cnt = by_level[lvl]
        total_lvl = sum(1 for r in exever_results if r["level"] == lvl)
        print(f"      Level {lvl}: {cnt:3d}/{total_lvl:3d} ({cnt/total_lvl:.1%} error rate)")

    error_types[error_verdict] = {
        "n": len(items),
        "greedy_accuracy": round(correct_count / len(items), 4),
        "by_subject": dict(by_subject),
        "by_level": {str(k): v for k, v in by_level.items()},
    }

# Overall: which subject/level combos have worst verification coverage?
print(f"\n  Worst verification coverage (subject x level):")
print(f"  {'Subject':25s} {'Level':>5s} {'N':>4s} {'Errors':>6s} {'Error%':>7s}")
print(f"  {'-'*25} {'-'*5} {'-'*4} {'-'*6} {'-'*7}")
subj_level_errors: list[tuple[str, int, int, int, float]] = []
for subj in sorted(set(r["type"] for r in exever_results)):
    for lvl in range(1, 6):
        total = sum(1 for r in exever_results if r["type"] == subj and r["level"] == lvl)
        if total == 0:
            continue
        errors = sum(1 for r in exever_results
                     if r["type"] == subj and r["level"] == lvl
                     and r["verdict"] in FALLBACK_VERDICTS)
        subj_level_errors.append((subj, lvl, total, errors, errors / total))

# Sort by error rate descending, show top 10
subj_level_errors.sort(key=lambda x: -x[4])
for subj, lvl, total, errors, rate in subj_level_errors[:15]:
    if total >= 3:  # skip tiny cells
        print(f"  {subj:25s} {lvl:5d} {total:4d} {errors:6d} {rate:7.1%}")

error_types["worst_coverage_cells"] = [
    {"subject": s, "level": l, "n": t, "errors": e, "error_rate": round(r, 4)}
    for s, l, t, e, r in subj_level_errors[:15] if t >= 3
]
output["error_type_analysis"] = error_types

# =========================================================================
# 5. ECHO CHAMBER DETAILED ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("5. ECHO CHAMBER DETAILED ANALYSIS")
print("=" * 70)

all_pass_items = verdict_groups["ALL_PASS"]
echo_items = [r for r in all_pass_items if r.get("echo_chamber", False)]
non_echo_items = [r for r in all_pass_items if not r.get("echo_chamber", False)]

print(f"\n  ALL_PASS total: {len(all_pass_items)}")
print(f"    Echo chamber: {len(echo_items)}")
print(f"    Non-echo:     {len(non_echo_items)}")

# Echo chamber by subject
echo_by_subject: dict[str, int] = Counter()
non_echo_by_subject: dict[str, int] = Counter()
for r in echo_items:
    echo_by_subject[r["type"]] += 1
for r in non_echo_items:
    non_echo_by_subject[r["type"]] += 1

print(f"\n  Echo chamber by subject:")
print(f"  {'Subject':25s} {'Echo':>5s} {'Non-echo':>9s} {'Total AP':>9s} {'Echo%':>7s}")
print(f"  {'-'*25} {'-'*5} {'-'*9} {'-'*9} {'-'*7}")
all_subjects = sorted(set(r["type"] for r in all_pass_items))
echo_subj_data = {}
for subj in all_subjects:
    e = echo_by_subject.get(subj, 0)
    ne = non_echo_by_subject.get(subj, 0)
    total = e + ne
    rate = e / total if total > 0 else 0
    print(f"  {subj:25s} {e:5d} {ne:9d} {total:9d} {rate:7.1%}")
    echo_subj_data[subj] = {"echo": e, "non_echo": ne, "total": total, "echo_rate": round(rate, 4)}

# Echo chamber by level
echo_by_level: dict[int, int] = Counter()
non_echo_by_level: dict[int, int] = Counter()
for r in echo_items:
    echo_by_level[r["level"]] += 1
for r in non_echo_items:
    non_echo_by_level[r["level"]] += 1

print(f"\n  Echo chamber by level:")
print(f"  {'Level':>5s} {'Echo':>5s} {'Non-echo':>9s} {'Total AP':>9s} {'Echo%':>7s}")
print(f"  {'-'*5} {'-'*5} {'-'*9} {'-'*9} {'-'*7}")
echo_level_data = {}
for lvl in range(1, 6):
    e = echo_by_level.get(lvl, 0)
    ne = non_echo_by_level.get(lvl, 0)
    total = e + ne
    rate = e / total if total > 0 else 0
    print(f"  {lvl:5d} {e:5d} {ne:9d} {total:9d} {rate:7.1%}")
    echo_level_data[str(lvl)] = {"echo": e, "non_echo": ne, "total": total, "echo_rate": round(rate, 4)}

# Accuracy: echo vs non-echo
echo_correct = sum(1 for r in echo_items if r["answer_correct"])
non_echo_correct = sum(1 for r in non_echo_items if r["answer_correct"])
echo_acc = echo_correct / len(echo_items) if echo_items else 0
non_echo_acc = non_echo_correct / len(non_echo_items) if non_echo_items else 0

print(f"\n  Accuracy comparison:")
print(f"    Echo chamber (n={len(echo_items)}):  {echo_correct}/{len(echo_items)} = {echo_acc:.1%}")
print(f"    Non-echo (n={len(non_echo_items)}):   {non_echo_correct}/{len(non_echo_items)} = {non_echo_acc:.1%}")
print(f"    Difference: {(echo_acc - non_echo_acc)*100:+.1f}pp")

# Do echo cases correlate with correctness?
print(f"\n  Echo chamber correlation with correctness:")
# Among correct greedy answers that reached ALL_PASS, what fraction are echo?
correct_ap = [r for r in all_pass_items if r["answer_correct"]]
incorrect_ap = [r for r in all_pass_items if not r["answer_correct"]]
echo_given_correct = sum(1 for r in correct_ap if r.get("echo_chamber", False))
echo_given_incorrect = sum(1 for r in incorrect_ap if r.get("echo_chamber", False))

print(f"    P(echo | correct, ALL_PASS):   {echo_given_correct}/{len(correct_ap)} = "
      f"{echo_given_correct/len(correct_ap):.1%}" if correct_ap else "    N/A")
print(f"    P(echo | incorrect, ALL_PASS): {echo_given_incorrect}/{len(incorrect_ap)} = "
      f"{echo_given_incorrect/len(incorrect_ap):.1%}" if incorrect_ap else "    N/A")

if incorrect_ap and correct_ap:
    echo_rate_correct = echo_given_correct / len(correct_ap)
    echo_rate_incorrect = echo_given_incorrect / len(incorrect_ap)
    print(f"    -> Echo is {'MORE' if echo_rate_incorrect > echo_rate_correct else 'LESS'} "
          f"common among incorrect ALL_PASS answers")
    print(f"       This {'confirms' if echo_rate_incorrect > echo_rate_correct else 'contradicts'} "
          f"the concern that echo = false verification")

echo_analysis = {
    "n_all_pass": len(all_pass_items),
    "n_echo": len(echo_items),
    "n_non_echo": len(non_echo_items),
    "echo_accuracy": round(echo_acc, 4),
    "non_echo_accuracy": round(non_echo_acc, 4),
    "accuracy_diff_pp": round((echo_acc - non_echo_acc) * 100, 2),
    "by_subject": echo_subj_data,
    "by_level": echo_level_data,
    "echo_rate_given_correct_allpass": round(echo_given_correct / len(correct_ap), 4) if correct_ap else None,
    "echo_rate_given_incorrect_allpass": round(echo_given_incorrect / len(incorrect_ap), 4) if incorrect_ap else None,
}
output["echo_chamber_analysis"] = echo_analysis

# =========================================================================
# ADDITIONAL: Self-correction & SymCode cross-analysis
# =========================================================================
print("\n" + "=" * 70)
print("ADDITIONAL: Cross-method comparison on same problem subsets")
print("=" * 70)

# Self-correction: which problems did it change, and were those changes harmful?
changed = [r for r in selfcorrect_results if r["changed"]]
print(f"\n  Self-correction changed {len(changed)} problems:")
changed_improved = sum(1 for r in changed if not r["initial_correct"] and r["corrected_correct"])
changed_degraded = sum(1 for r in changed if r["initial_correct"] and not r["corrected_correct"])
changed_same = sum(1 for r in changed if r["initial_correct"] == r["corrected_correct"])
print(f"    Improved (wrong->right): {changed_improved}")
print(f"    Degraded (right->wrong): {changed_degraded}")
print(f"    Same correctness:        {changed_same}")

# Self-correction changes by level
print(f"\n  Self-correction degradation by level:")
for lvl in range(1, 6):
    lvl_changed = [r for r in changed if r["level"] == lvl]
    lvl_degraded = sum(1 for r in lvl_changed if r["initial_correct"] and not r["corrected_correct"])
    lvl_total = sum(1 for r in selfcorrect_results if r["level"] == lvl)
    print(f"    Level {lvl}: {lvl_degraded}/{lvl_total} degraded ({lvl_degraded/lvl_total:.1%})")

# SymCode: how does it relate to ExeVer verdicts?
print(f"\n  SymCode accuracy by ExeVer verdict group:")
for verdict in ["ALL_PASS", "REPAIRED", "SYNTAX_ERROR", "RUNTIME_ERROR"]:
    items = verdict_groups.get(verdict, [])
    if not items:
        continue
    sym_correct = sum(1 for r in items if symcode_by_id.get(r["id"], {}).get("correct", False))
    print(f"    {verdict:25s}: SymCode {sym_correct}/{len(items)} = {sym_correct/len(items):.1%}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY OF KEY FINDINGS")
print("=" * 70)

print(f"""
  1. CONFIDENCE CALIBRATION:
     - ALL_PASS accuracy: {group_accuracy(verified_items)[2]:.1%}
     - FALLBACK accuracy: {group_accuracy(fallback_items)[2]:.1%}
     - Verdict {'IS' if ap_acc > fb_acc + 0.05 else 'is NOT'} a reliable confidence signal ({(ap_acc-fb_acc)*100:+.1f}pp gap)

  2. BOOTSTRAP CIs (95%):
     - Greedy:    {bootstrap_results['greedy_cot']['observed']:.1%} [{bootstrap_results['greedy_cot']['ci_lower']:.1%}, {bootstrap_results['greedy_cot']['ci_upper']:.1%}]
     - ExeVer:    {bootstrap_results['exever']['observed']:.1%} [{bootstrap_results['exever']['ci_lower']:.1%}, {bootstrap_results['exever']['ci_upper']:.1%}]
     - Majority@4: {bootstrap_results['majority_4']['observed']:.1%} [{bootstrap_results['majority_4']['ci_lower']:.1%}, {bootstrap_results['majority_4']['ci_upper']:.1%}]
     - Best-of-4: {bootstrap_results['best_of_4']['observed']:.1%} [{bootstrap_results['best_of_4']['ci_lower']:.1%}, {bootstrap_results['best_of_4']['ci_upper']:.1%}]
     - ExeVer vs Greedy: {bootstrap_results['exever_minus_greedy']['observed_diff']*100:+.2f}pp, CI [{bootstrap_results['exever_minus_greedy']['ci_lower']*100:.2f}, {bootstrap_results['exever_minus_greedy']['ci_upper']*100:+.2f}]pp
     - Difference is {'NOT ' if not bootstrap_results['exever_minus_greedy']['significant'] else ''}statistically significant

  3. COVERAGE-CONDITIONED:
     - Verified subset (n={coverage_cond['verified']['n']}): ExeVer {coverage_cond['verified']['exever_accuracy']:.1%}, Greedy {coverage_cond['verified']['greedy_accuracy']:.1%}
     - Fallback subset (n={coverage_cond['fallback']['n']}): ExeVer {coverage_cond['fallback']['exever_accuracy']:.1%}, Greedy {coverage_cond['fallback']['greedy_accuracy']:.1%}
     - Selection bias: {coverage_cond['greedy_diff_pp']:+.1f}pp (verified problems are {'easier' if coverage_cond['selection_bias'] else 'NOT systematically easier'})

  4. ERROR PATTERNS:
     - SYNTAX_ERROR: {error_types.get('SYNTAX_ERROR', {}).get('n', 0)} problems (worst: precalculus, geometry)
     - RUNTIME_ERROR: {error_types.get('RUNTIME_ERROR', {}).get('n', 0)} problems (worst: intermediate_algebra, precalculus)

  5. ECHO CHAMBER:
     - Echo accuracy: {echo_acc:.1%} vs Non-echo accuracy: {non_echo_acc:.1%}
     - Echo is {'MORE' if echo_analysis.get('echo_rate_given_incorrect_allpass', 0) > echo_analysis.get('echo_rate_given_correct_allpass', 0) else 'LESS'} common among incorrect ALL_PASS answers
""")

# Save
output["self_correction_changes"] = {
    "n_changed": len(changed),
    "improved": changed_improved,
    "degraded": changed_degraded,
    "same": changed_same,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {OUTPUT_PATH}")
print("Done.")
