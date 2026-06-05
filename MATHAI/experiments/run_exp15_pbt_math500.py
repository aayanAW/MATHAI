"""Experiment 15: PBT measurement on MATH-500.

Runs specification-grounded randomized checking on existing ExeVer
solutions (exp5 data). Measures: coverage, FPVR, independence rate,
per-claim-type metrics.

This is the Phase 1 go/no-go experiment.
Gate: PBT FPVR on fully-independent tests < ExeVer's 13.8%.

Usage:
    python3 experiments/run_exp15_pbt_math500.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.pbt.pipeline import run_pbt, PBTResult
    from src.pbt.claim_classifier import ClaimType

    # Load MATH-500 problems
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        problems = json.load(f)

    # Load exp5 solutions (reuse existing greedy solutions)
    with open(RESULTS_DIR / "exp5_math500_full.json") as f:
        exp5 = json.load(f)

    solutions = {r["id"]: r.get("nl_solution", "") for r in exp5["exever_results"]}

    print(f"Loaded {len(problems)} problems, {len(solutions)} solutions")
    print("=" * 60)
    print("EXPERIMENT 15: PBT on MATH-500")
    print("=" * 60)

    # Run PBT on each problem
    all_results = []
    for i, prob in enumerate(problems):
        sol = solutions.get(prob["id"], "")
        if not sol:
            print(f"  [{i+1}/{len(problems)}] {prob['id']}: NO SOLUTION, skipping")
            continue

        result = run_pbt(
            problem=prob["problem"],
            solution=sol,
            gold_answer=prob["answer"],
            problem_id=prob["id"],
        )
        all_results.append(result)

        if (i + 1) % 50 == 0:
            n_testable = sum(r.n_testable for r in all_results)
            n_total_steps = sum(r.n_steps for r in all_results)
            print(f"  [{i+1}/{len(problems)}] testable: {n_testable}/{n_total_steps} steps")

    print(f"\nProcessed {len(all_results)} problems")

    # === Compute Metrics ===

    # 1. Coverage
    total_steps = sum(r.n_steps for r in all_results)
    total_testable = sum(r.n_testable for r in all_results)
    total_fully_indep = sum(r.n_fully_independent for r in all_results)
    total_partially = sum(r.n_partially_independent for r in all_results)
    total_untestable = sum(r.n_untestable for r in all_results)

    step_coverage = total_testable / total_steps if total_steps > 0 else 0
    problem_coverage = sum(1 for r in all_results if r.n_testable > 0) / len(all_results)

    print(f"\n--- Coverage ---")
    print(f"  Step coverage: {total_testable}/{total_steps} ({step_coverage:.1%})")
    print(f"  Problem coverage: {problem_coverage:.1%}")
    print(f"  Fully independent: {total_fully_indep}")
    print(f"  Partially independent: {total_partially}")
    print(f"  Untestable: {total_untestable}")
    print(f"  Independence rate: {total_fully_indep/max(total_testable,1):.1%}")

    # 2. FPVR
    all_pass_results = [r for r in all_results if r.all_tested_pass and r.n_testable > 0]
    n_all_pass = len(all_pass_results)
    n_false_positive = sum(1 for r in all_pass_results if r.fpvr is True)
    fpvr = n_false_positive / n_all_pass if n_all_pass > 0 else 0

    # FPVR on fully-independent-only
    fully_indep_pass = [
        r for r in all_results
        if r.n_fully_independent > 0
        and all(
            sr.label == "PASS"
            for sr in r.step_results
            if sr.independence == "fully"
        )
    ]
    n_fi_pass = len(fully_indep_pass)
    n_fi_fp = sum(1 for r in fully_indep_pass if not r.answer_correct)
    fpvr_fi = n_fi_fp / n_fi_pass if n_fi_pass > 0 else 0

    print(f"\n--- FPVR ---")
    print(f"  Overall FPVR (all tested): {n_false_positive}/{n_all_pass} ({fpvr:.1%})")
    print(f"  Fully-independent FPVR: {n_fi_fp}/{n_fi_pass} ({fpvr_fi:.1%}) [PRIMARY METRIC]")
    print(f"  ExeVer FPVR (comparison): 13.8%")
    print(f"  NOTE: Partially-independent tests share operands with model state")
    print(f"        and should NOT be counted in headline FPVR")

    # 3. GO/NO-GO GATE (uses fully-independent FPVR per specification)
    print(f"\n--- GO/NO-GO GATE ---")
    gate_fpvr = fpvr_fi  # Use fully-independent FPVR, not overall
    if gate_fpvr < 0.138:
        print(f"  PASS: Fully-independent FPVR ({gate_fpvr:.1%}) < ExeVer FPVR (13.8%)")
    else:
        print(f"  FAIL: Fully-independent FPVR ({gate_fpvr:.1%}) >= ExeVer FPVR (13.8%)")
        print(f"  Project should pivot to diagnostic study")

    # 4. Per-claim-type breakdown
    claim_counts = Counter()
    claim_pass = Counter()
    claim_fail = Counter()
    for r in all_results:
        for sr in r.step_results:
            ct = sr.claim_type.value
            claim_counts[ct] += 1
            if sr.label == "PASS":
                claim_pass[ct] += 1
            elif sr.label == "FAIL":
                claim_fail[ct] += 1

    print(f"\n--- Per-Claim-Type ---")
    print(f"  {'Type':<20} {'Count':>6} {'Pass':>6} {'Fail':>6} {'Untested':>8}")
    for ct in sorted(claim_counts.keys()):
        n = claim_counts[ct]
        p = claim_pass.get(ct, 0)
        f_ = claim_fail.get(ct, 0)
        u = n - p - f_
        print(f"  {ct:<20} {n:>6} {p:>6} {f_:>6} {u:>8}")

    # 5. FPVR by difficulty level
    print(f"\n--- FPVR by Level ---")
    for lv in [1, 2, 3, 4, 5]:
        lv_probs = {p["id"] for p in problems if p["level"] == lv}
        lv_pass = [r for r in all_pass_results if r.problem_id in lv_probs]
        lv_fp = sum(1 for r in lv_pass if r.fpvr is True)
        lv_fpvr = lv_fp / len(lv_pass) if lv_pass else 0
        print(f"  L{lv}: {lv_fp}/{len(lv_pass)} ({lv_fpvr:.1%})")

    # 6. Calibration
    tested_correct = sum(1 for r in all_pass_results if r.answer_correct)
    untested_results = [r for r in all_results if not r.all_tested_pass or r.n_testable == 0]
    untested_correct = sum(1 for r in untested_results if r.answer_correct)
    pass_acc = tested_correct / n_all_pass if n_all_pass > 0 else 0
    untest_acc = untested_correct / len(untested_results) if untested_results else 0

    print(f"\n--- Calibration ---")
    print(f"  ALL_PASS accuracy: {pass_acc:.1%} (n={n_all_pass})")
    print(f"  Non-pass accuracy: {untest_acc:.1%} (n={len(untested_results)})")
    print(f"  Gap: +{(pass_acc - untest_acc)*100:.1f}pp")

    # === Save Results ===
    output = {
        "experiment": "exp15_pbt_math500",
        "n_problems": len(all_results),
        "coverage": {
            "step_coverage": round(step_coverage, 4),
            "problem_coverage": round(problem_coverage, 4),
            "total_steps": total_steps,
            "total_testable": total_testable,
            "total_fully_independent": total_fully_indep,
            "total_partially_independent": total_partially,
            "total_untestable": total_untestable,
            "independence_rate": round(total_fully_indep / max(total_testable, 1), 4),
        },
        "fpvr": {
            "overall": round(fpvr, 4),
            "fully_independent": round(fpvr_fi, 4),
            "n_all_pass": n_all_pass,
            "n_false_positive": n_false_positive,
        },
        "go_no_go": fpvr < 0.138,
        "per_claim_type": {
            ct: {"count": claim_counts[ct], "pass": claim_pass.get(ct, 0), "fail": claim_fail.get(ct, 0)}
            for ct in sorted(claim_counts.keys())
        },
        "calibration": {
            "all_pass_accuracy": round(pass_acc, 4),
            "non_pass_accuracy": round(untest_acc, 4),
            "gap_pp": round((pass_acc - untest_acc) * 100, 1),
        },
        "detailed_results": [r.to_dict() for r in all_results],
    }

    out_path = RESULTS_DIR / "exp15_pbt_math500.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
