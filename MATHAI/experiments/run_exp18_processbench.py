"""Experiment 18: ProcessBench evaluation.

Runs PBT on ProcessBench MATH split (1000 examples with
human-annotated first-error-step labels). Measures step-level
precision/recall/F1 on the computationally testable subset.

Usage:
    python3 experiments/run_exp18_processbench.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

RESULTS_DIR = Path("results")


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.pbt.pipeline import run_pbt
    from src.pbt.claim_classifier import ClaimType

    print("Loading ProcessBench MATH split...")
    ds = load_dataset("Qwen/ProcessBench", split="math")
    print(f"Loaded {len(ds)} examples")

    print("=" * 60)
    print("EXPERIMENT 18: ProcessBench Evaluation")
    print("=" * 60)

    results = []
    for i, ex in enumerate(ds):
        # Reconstruct solution from steps
        steps = ex["steps"]
        solution = "\n\n".join(f"## Step {j+1}\n{s}" for j, s in enumerate(steps))

        # The gold label: index of first error step (-1 if all correct)
        # label=0 means first step is wrong, label=k means step k+1 is first error
        gold_label = ex["label"]  # -1 = all correct, else 0-indexed first error
        gold_correct = ex["final_answer_correct"]

        # Run PBT — we don't have a gold answer to test against,
        # so we use PBT to detect if ANY step fails
        # For final_answer test we'd need the gold answer which ProcessBench doesn't provide
        # So we run PBT on intermediate steps only
        pbt_result = run_pbt(
            problem=ex["problem"],
            solution=solution,
            gold_answer="",  # ProcessBench doesn't provide gold answers
            problem_id=ex["id"],
        )

        # PBT's prediction: does any tested step fail?
        has_fail = pbt_result.n_failed > 0
        has_test = pbt_result.n_testable > 0

        # Ground truth: is there an error?
        has_error = gold_label >= 0  # -1 means no error

        results.append({
            "id": ex["id"],
            "generator": ex["generator"],
            "gold_label": gold_label,
            "gold_correct": gold_correct,
            "has_error": has_error,
            "pbt_has_fail": has_fail,
            "pbt_has_test": has_test,
            "n_tested": pbt_result.n_testable,
            "n_failed": pbt_result.n_failed,
            "n_steps": pbt_result.n_steps,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(ds)}] processed")

    print(f"\nProcessed {len(results)} examples")

    # === Metrics ===

    # Split into testable (PBT generated at least one test) vs abstained
    testable = [r for r in results if r["pbt_has_test"]]
    abstained = [r for r in results if not r["pbt_has_test"]]

    print(f"\n--- Coverage ---")
    print(f"  Testable: {len(testable)}/{len(results)} ({len(testable)/len(results)*100:.1f}%)")
    print(f"  Abstained: {len(abstained)}/{len(results)} ({len(abstained)/len(results)*100:.1f}%)")

    # On testable subset: precision/recall/F1 for error detection
    # PBT predicts "error" if any step fails
    # Gold truth: has_error (label >= 0)
    tp = sum(1 for r in testable if r["pbt_has_fail"] and r["has_error"])
    fp = sum(1 for r in testable if r["pbt_has_fail"] and not r["has_error"])
    fn = sum(1 for r in testable if not r["pbt_has_fail"] and r["has_error"])
    tn = sum(1 for r in testable if not r["pbt_has_fail"] and not r["has_error"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- Error Detection (testable subset, n={len(testable)}) ---")
    print(f"  TP (PBT detects real error): {tp}")
    print(f"  FP (PBT flags correct solution): {fp}")
    print(f"  FN (PBT misses real error): {fn}")
    print(f"  TN (PBT accepts correct solution): {tn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")

    # Full benchmark with abstention
    print(f"\n--- Full Benchmark with Abstention (n={len(results)}) ---")
    n_errors = sum(1 for r in results if r["has_error"])
    n_correct = sum(1 for r in results if not r["has_error"])
    detected_errors = sum(1 for r in results if r["pbt_has_fail"] and r["has_error"])
    print(f"  Total errors in benchmark: {n_errors}")
    print(f"  Total correct in benchmark: {n_correct}")
    print(f"  Errors detected by PBT: {detected_errors}/{n_errors} ({detected_errors/n_errors*100:.1f}%)")
    print(f"  False alarms: {fp}")

    # By generator
    generators = set(r["generator"] for r in results)
    print(f"\n--- By Generator ---")
    for gen in sorted(generators):
        gen_results = [r for r in results if r["generator"] == gen]
        gen_testable = [r for r in gen_results if r["pbt_has_test"]]
        gen_tp = sum(1 for r in gen_testable if r["pbt_has_fail"] and r["has_error"])
        gen_fp = sum(1 for r in gen_testable if r["pbt_has_fail"] and not r["has_error"])
        gen_fn = sum(1 for r in gen_testable if not r["pbt_has_fail"] and r["has_error"])
        gen_prec = gen_tp / (gen_tp + gen_fp) if (gen_tp + gen_fp) > 0 else 0
        gen_rec = gen_tp / (gen_tp + gen_fn) if (gen_tp + gen_fn) > 0 else 0
        print(f"  {gen}: n={len(gen_results)}, testable={len(gen_testable)}, "
              f"TP={gen_tp}, FP={gen_fp}, FN={gen_fn}, P={gen_prec:.2f}, R={gen_rec:.2f}")

    # Save
    output = {
        "experiment": "exp18_processbench",
        "n_total": len(results),
        "n_testable": len(testable),
        "n_abstained": len(abstained),
        "coverage": len(testable) / len(results),
        "error_detection": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "full_benchmark": {
            "total_errors": n_errors,
            "detected": detected_errors,
            "detection_rate": round(detected_errors / n_errors, 4) if n_errors > 0 else 0,
        },
    }

    out_path = RESULTS_DIR / "exp18_processbench.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
