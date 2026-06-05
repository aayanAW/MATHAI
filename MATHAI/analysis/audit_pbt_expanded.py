"""Expanded PBT audit: 300 random tested steps.

Provides more statistical power than the initial 100-step audit.
"""
import json
import random
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("results")


def main():
    with open(RESULTS_DIR / "exp15_pbt_math500.json") as f:
        pbt = json.load(f)

    detailed = pbt["detailed_results"]

    all_steps = []
    for result in detailed:
        for sr in result.get("step_results", []):
            all_steps.append({
                "problem_id": result["problem_id"],
                "answer_correct": result["answer_correct"],
                "step_index": sr["step_index"],
                "claim_type": sr["claim_type"],
                "label": sr["label"],
                "independence": sr["independence"],
            })

    tested = [s for s in all_steps if s["label"] in ("PASS", "FAIL")]
    print(f"Total tested steps: {len(tested)}")

    random.seed(42)
    sample_size = min(300, len(tested))
    sample = random.sample(tested, sample_size)

    print(f"Audit sample: {sample_size} steps")
    print("=" * 60)

    # Type distribution
    type_counts = Counter(s["claim_type"] for s in sample)
    label_counts = Counter(s["label"] for s in sample)
    indep_counts = Counter(s["independence"] for s in sample)

    print(f"\n--- Claim types ---")
    for ct, n in type_counts.most_common():
        print(f"  {ct:<20} {n:>4}")

    print(f"\n--- Labels ---")
    for l, n in label_counts.most_common():
        print(f"  {l:<10} {n:>4}")

    print(f"\n--- Independence ---")
    for i, n in indep_counts.most_common():
        print(f"  {i:<15} {n:>4}")

    # Key metrics
    answer_steps = [s for s in sample if s["claim_type"] == "answer"]
    non_answer = [s for s in sample if s["claim_type"] != "answer"]

    # Answer steps audit
    a_pass_correct = sum(1 for s in answer_steps if s["label"] == "PASS" and s["answer_correct"])
    a_pass_wrong = sum(1 for s in answer_steps if s["label"] == "PASS" and not s["answer_correct"])
    a_fail_correct = sum(1 for s in answer_steps if s["label"] == "FAIL" and s["answer_correct"])
    a_fail_wrong = sum(1 for s in answer_steps if s["label"] == "FAIL" and not s["answer_correct"])

    print(f"\n--- Answer step audit (n={len(answer_steps)}) ---")
    print(f"  PASS + correct: {a_pass_correct}")
    print(f"  PASS + wrong:   {a_pass_wrong} (FALSE PASS)")
    print(f"  FAIL + correct: {a_fail_correct} (FALSE FAIL)")
    print(f"  FAIL + wrong:   {a_fail_wrong}")
    if answer_steps:
        accuracy = (a_pass_correct + a_fail_wrong) / len(answer_steps)
        false_pass_rate = a_pass_wrong / max(a_pass_correct + a_pass_wrong, 1)
        false_fail_rate = a_fail_correct / max(a_fail_correct + a_fail_wrong, 1)
        print(f"  Label accuracy: {accuracy:.1%}")
        print(f"  False pass rate: {false_pass_rate:.1%}")
        print(f"  False fail rate: {false_fail_rate:.1%}")

    # Non-answer steps
    na_pass_correct = sum(1 for s in non_answer if s["label"] == "PASS" and s["answer_correct"])
    na_pass_wrong = sum(1 for s in non_answer if s["label"] == "PASS" and not s["answer_correct"])
    na_fail_correct = sum(1 for s in non_answer if s["label"] == "FAIL" and s["answer_correct"])
    na_fail_wrong = sum(1 for s in non_answer if s["label"] == "FAIL" and not s["answer_correct"])

    print(f"\n--- Non-answer step audit (n={len(non_answer)}) ---")
    print(f"  PASS + correct answer: {na_pass_correct}")
    print(f"  PASS + wrong answer:   {na_pass_wrong}")
    print(f"  FAIL + correct answer: {na_fail_correct}")
    print(f"  FAIL + wrong answer:   {na_fail_wrong}")

    # By claim type
    print(f"\n--- Per-type label accuracy (answer steps excluded) ---")
    for ct in sorted(set(s["claim_type"] for s in non_answer)):
        ct_steps = [s for s in non_answer if s["claim_type"] == ct]
        ct_pass = sum(1 for s in ct_steps if s["label"] == "PASS")
        ct_fail = sum(1 for s in ct_steps if s["label"] == "FAIL")
        print(f"  {ct:<20} PASS={ct_pass:>3} FAIL={ct_fail:>3} total={len(ct_steps):>3}")

    # Save
    output = {
        "sample_size": sample_size,
        "type_distribution": dict(type_counts),
        "label_distribution": dict(label_counts),
        "answer_audit": {
            "n": len(answer_steps),
            "pass_correct": a_pass_correct,
            "pass_wrong": a_pass_wrong,
            "fail_correct": a_fail_correct,
            "fail_wrong": a_fail_wrong,
            "label_accuracy": (a_pass_correct + a_fail_wrong) / max(len(answer_steps), 1),
            "false_pass_rate": a_pass_wrong / max(a_pass_correct + a_pass_wrong, 1),
            "false_fail_rate": a_fail_correct / max(a_fail_correct + a_fail_wrong, 1),
        },
        "non_answer_audit": {
            "n": len(non_answer),
            "pass_correct_answer": na_pass_correct,
            "pass_wrong_answer": na_pass_wrong,
            "fail_correct_answer": na_fail_correct,
            "fail_wrong_answer": na_fail_wrong,
        },
    }

    out_path = RESULTS_DIR / "pbt_audit_expanded.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
