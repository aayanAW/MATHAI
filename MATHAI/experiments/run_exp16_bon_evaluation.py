"""Experiment 16: Best-of-N evaluation using PBT as a filter.

Since we have greedy solutions + PBT verdicts, we evaluate PBT's
utility as a binary accept/reject filter:
- If PBT says ALL_PASS -> accept the solution (use it)
- If PBT says FAIL -> reject (fall back to random/majority)

This measures whether PBT-filtered solutions are more accurate
than unfiltered ones, which is the core downstream utility.

We also evaluate using PBT labels to RERANK the exp5 sampled
solutions (4 per problem) if that data is recoverable.

Usage:
    python3 experiments/run_exp16_bon_evaluation.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("results")


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer

    # Load PBT results
    with open(RESULTS_DIR / "exp15_pbt_math500.json") as f:
        pbt = json.load(f)

    # Load problems
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        problems = json.load(f)
    prob_map = {p["id"]: p for p in problems}

    detailed = pbt["detailed_results"]
    print(f"Loaded {len(detailed)} PBT results")
    print("=" * 60)
    print("EXPERIMENT 16: Best-of-N Evaluation with PBT Filter")
    print("=" * 60)

    # === Evaluation 1: PBT as Accept/Reject Filter ===
    # Split problems into PBT-accepted (all tested steps pass) vs PBT-rejected
    accepted = [r for r in detailed if r["all_tested_pass"] and r["n_testable"] > 0]
    rejected = [r for r in detailed if not r["all_tested_pass"] or r["n_testable"] == 0]

    acc_correct = sum(1 for r in accepted if r["answer_correct"])
    rej_correct = sum(1 for r in rejected if r["answer_correct"])
    all_correct = sum(1 for r in detailed if r["answer_correct"])

    print(f"\n--- PBT as Accept/Reject Filter ---")
    print(f"  Total: {len(detailed)} problems, {all_correct} correct ({all_correct/len(detailed)*100:.1f}%)")
    print(f"  PBT-accepted: {len(accepted)} problems, {acc_correct} correct ({acc_correct/len(accepted)*100:.1f}%)" if accepted else "  PBT-accepted: 0")
    print(f"  PBT-rejected: {len(rejected)} problems, {rej_correct} correct ({rej_correct/len(rejected)*100:.1f}%)" if rejected else "  PBT-rejected: 0")

    if accepted:
        acc_rate = acc_correct / len(accepted) * 100
        rej_rate = rej_correct / len(rejected) * 100 if rejected else 0
        print(f"  Accuracy lift: +{acc_rate - all_correct/len(detailed)*100:.1f}pp for accepted vs overall")
        print(f"  Filtering value: accepted solutions are {acc_rate:.1f}% accurate vs {rej_rate:.1f}% for rejected")

    # === Evaluation 2: By Difficulty Level ===
    print(f"\n--- PBT Filter by Difficulty Level ---")
    print(f"  {'Level':<8} {'Accepted':>10} {'Acc. Acc%':>10} {'Rejected':>10} {'Rej. Acc%':>10} {'Lift':>8}")
    for lv in [1, 2, 3, 4, 5]:
        lv_acc = [r for r in accepted if prob_map.get(r["problem_id"], {}).get("level") == lv]
        lv_rej = [r for r in rejected if prob_map.get(r["problem_id"], {}).get("level") == lv]

        lv_acc_correct = sum(1 for r in lv_acc if r["answer_correct"])
        lv_rej_correct = sum(1 for r in lv_rej if r["answer_correct"])

        acc_pct = lv_acc_correct / len(lv_acc) * 100 if lv_acc else 0
        rej_pct = lv_rej_correct / len(lv_rej) * 100 if lv_rej else 0
        lift = acc_pct - rej_pct if lv_acc and lv_rej else 0

        print(f"  L{lv:<7} {len(lv_acc):>10} {acc_pct:>9.1f}% {len(lv_rej):>10} {rej_pct:>9.1f}% {lift:>+7.1f}pp")

    # === Evaluation 3: By Claim Type Coverage ===
    print(f"\n--- Accuracy by Number of Tested Steps ---")
    by_n_tested = {}
    for r in detailed:
        n = r["n_testable"]
        bucket = "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"
        by_n_tested.setdefault(bucket, []).append(r)

    for bucket in ["0", "1", "2-3", "4+"]:
        items = by_n_tested.get(bucket, [])
        if items:
            n_correct = sum(1 for r in items if r["answer_correct"])
            print(f"  {bucket} tested steps: {n_correct}/{len(items)} ({n_correct/len(items)*100:.1f}%) correct")

    # === Evaluation 4: Fully-Independent Filter Only ===
    print(f"\n--- Fully-Independent Filter (strictest) ---")
    fi_accepted = [
        r for r in detailed
        if r["n_fully_independent"] > 0
        and all(
            sr["label"] == "PASS"
            for sr in r.get("step_results", [])
            if sr["independence"] == "fully"
        )
    ]
    fi_rejected = [r for r in detailed if r not in fi_accepted]

    fi_acc_correct = sum(1 for r in fi_accepted if r["answer_correct"])
    fi_rej_correct = sum(1 for r in fi_rejected if r["answer_correct"])

    print(f"  FI-accepted: {len(fi_accepted)} problems, {fi_acc_correct} correct ({fi_acc_correct/len(fi_accepted)*100:.1f}%)" if fi_accepted else "  FI-accepted: 0")
    print(f"  FI-rejected: {len(fi_rejected)} problems, {fi_rej_correct} correct ({fi_rej_correct/len(fi_rejected)*100:.1f}%)" if fi_rejected else "  FI-rejected: 0")

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Greedy baseline: {all_correct/len(detailed)*100:.1f}% ({all_correct}/{len(detailed)})")
    if accepted:
        print(f"  PBT-accepted accuracy: {acc_correct/len(accepted)*100:.1f}% ({acc_correct}/{len(accepted)})")
        print(f"  PBT-rejected accuracy: {rej_correct/len(rejected)*100:.1f}% ({rej_correct}/{len(rejected)})")
    if fi_accepted:
        print(f"  FI-accepted accuracy: {fi_acc_correct/len(fi_accepted)*100:.1f}% ({fi_acc_correct}/{len(fi_accepted)})")

    # Save
    output = {
        "experiment": "exp16_bon_evaluation",
        "n_problems": len(detailed),
        "greedy_accuracy": all_correct / len(detailed),
        "pbt_filter": {
            "n_accepted": len(accepted),
            "accepted_accuracy": acc_correct / len(accepted) if accepted else None,
            "n_rejected": len(rejected),
            "rejected_accuracy": rej_correct / len(rejected) if rejected else None,
        },
        "fi_filter": {
            "n_accepted": len(fi_accepted),
            "accepted_accuracy": fi_acc_correct / len(fi_accepted) if fi_accepted else None,
            "n_rejected": len(fi_rejected),
            "rejected_accuracy": fi_rej_correct / len(fi_rejected) if fi_rejected else None,
        },
    }

    out_path = RESULTS_DIR / "exp16_bon_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
