"""Stratify ProcessBench precision by generator model.

Addresses NeurIPS reviewer concern 4:
"ProcessBench precision (0.61) undermines the perfect precision story.
You need to reconcile the mismatch between MATH-500 (1.00) and ProcessBench (0.61)."

Hypothesis: the precision drop is caused by extraction failures on
out-of-distribution solution formats from different generators,
not by a fundamental method limitation.

If precision is high on Qwen-family generators (similar LaTeX
conventions to our template-training distribution) and lower on
Llama/Mistral (different conventions), the hypothesis is supported.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from scipy.stats import binomtest

RESULTS_DIR = Path(__file__).parent.parent / "results"


def ci(k, n, conf=0.95):
    if n == 0:
        return 0.0, 1.0
    result = binomtest(k, n)
    cint = result.proportion_ci(confidence_level=conf, method="exact")
    return cint.low, cint.high


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.pbt.pipeline import run_pbt

    print("Loading ProcessBench MATH split...")
    ds = load_dataset("Qwen/ProcessBench", split="math")
    print(f"Loaded {len(ds)} examples across generators: {set(ex['generator'] for ex in ds)}")

    print("=" * 60)
    print("STRATIFIED PROCESSBENCH EVALUATION")
    print("=" * 60)

    # Re-run PBT with generator tracking
    results = []
    for i, ex in enumerate(ds):
        steps = ex["steps"]
        solution = "\n\n".join(f"## Step {j+1}\n{s}" for j, s in enumerate(steps))

        gold_label = ex["label"]
        has_error = gold_label >= 0

        pbt_result = run_pbt(
            problem=ex["problem"],
            solution=solution,
            gold_answer="",
            problem_id=ex["id"],
        )

        has_fail = pbt_result.n_failed > 0
        has_test = pbt_result.n_testable > 0

        results.append({
            "id": ex["id"],
            "generator": ex["generator"],
            "has_error": has_error,
            "pbt_has_fail": has_fail,
            "pbt_has_test": has_test,
            "n_tested": pbt_result.n_testable,
            "n_failed": pbt_result.n_failed,
        })

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(ds)}]")

    # Compute per-generator metrics
    by_gen = defaultdict(list)
    for r in results:
        by_gen[r["generator"]].append(r)

    print(f"\n--- Per-Generator Metrics (testable subset) ---")
    print(f"{'Generator':<35} {'N':>5} {'Test%':>7} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("-" * 105)

    stratified = {}
    for gen in sorted(by_gen.keys()):
        gen_results = by_gen[gen]
        gen_testable = [r for r in gen_results if r["pbt_has_test"]]

        tp = sum(1 for r in gen_testable if r["pbt_has_fail"] and r["has_error"])
        fp = sum(1 for r in gen_testable if r["pbt_has_fail"] and not r["has_error"])
        fn = sum(1 for r in gen_testable if not r["pbt_has_fail"] and r["has_error"])
        tn = sum(1 for r in gen_testable if not r["pbt_has_fail"] and not r["has_error"])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        prec_low, prec_high = ci(tp, tp + fp) if (tp + fp) > 0 else (0, 0)

        test_pct = len(gen_testable) / len(gen_results) * 100

        short = gen.replace("Meta-Llama-", "Llama-").replace("-Instruct", "")[:34]
        print(f"{short:<35} {len(gen_results):>5} {test_pct:>6.1f}% "
              f"{tp:>5} {fp:>5} {fn:>5} "
              f"{prec:>8.2f} [{prec_low:.2f}-{prec_high:.2f}] {rec:>6.2f} {f1:>6.2f}")

        stratified[gen] = {
            "n_total": len(gen_results),
            "n_testable": len(gen_testable),
            "test_rate": round(test_pct / 100, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4),
            "precision_ci": [round(prec_low, 4), round(prec_high, 4)],
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    # Group generators into "similar distribution" vs "dissimilar"
    print(f"\n--- Distribution-Shift Analysis ---")
    qwen_gens = [g for g in by_gen if "Qwen" in g]
    other_gens = [g for g in by_gen if "Qwen" not in g]

    def agg(gens):
        items = []
        for g in gens:
            items.extend([r for r in by_gen[g] if r["pbt_has_test"]])
        tp = sum(1 for r in items if r["pbt_has_fail"] and r["has_error"])
        fp = sum(1 for r in items if r["pbt_has_fail"] and not r["has_error"])
        fn = sum(1 for r in items if not r["pbt_has_fail"] and r["has_error"])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return tp, fp, fn, prec, rec, f1, len(items)

    qtp, qfp, qfn, qprec, qrec, qf1, qn = agg(qwen_gens)
    otp, ofp, ofn, oprec, orec, of1, on = agg(other_gens)

    print(f"  Qwen-family (n={qn}):   precision={qprec:.3f}, recall={qrec:.3f}, F1={qf1:.3f}")
    print(f"  Other-family (n={on}):  precision={oprec:.3f}, recall={orec:.3f}, F1={of1:.3f}")
    print(f"  Precision gap: {(qprec - oprec):.3f}")
    print(f"  Interpretation: {'Distribution shift confirmed' if qprec > oprec + 0.05 else 'Uniform precision (not distribution shift)'}")

    # Save
    output = {
        "stratified": stratified,
        "distribution_shift": {
            "qwen_family": {"precision": round(qprec, 4), "recall": round(qrec, 4), "f1": round(qf1, 4), "n": qn},
            "other_family": {"precision": round(oprec, 4), "recall": round(orec, 4), "f1": round(of1, 4), "n": on},
            "precision_gap": round(qprec - oprec, 4),
        },
    }

    out_path = RESULTS_DIR / "processbench_stratified.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
