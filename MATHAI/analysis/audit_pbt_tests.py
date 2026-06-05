"""Manual audit helper for PBT tests.

Samples 100 PBT test results and outputs them in a format
suitable for human verification. Computes automated audit
metrics where possible.

Usage:
    python3 analysis/audit_pbt_tests.py
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("results")


def main():
    with open(RESULTS_DIR / "exp15_pbt_math500.json") as f:
        pbt = json.load(f)

    detailed = pbt["detailed_results"]
    print(f"Loaded {len(detailed)} PBT results")

    # Collect all step-level results
    all_steps = []
    for result in detailed:
        for sr in result.get("step_results", []):
            all_steps.append({
                "problem_id": result["problem_id"],
                "answer_correct": result["answer_correct"],
                "answer_predicted": result["answer_predicted"],
                "step_index": sr["step_index"],
                "claim_type": sr["claim_type"],
                "label": sr["label"],
                "independence": sr["independence"],
                "validation_notes": sr.get("validation_notes", ""),
            })

    print(f"Total steps: {len(all_steps)}")

    # Separate tested vs untested
    tested = [s for s in all_steps if s["label"] in ("PASS", "FAIL")]
    untested = [s for s in all_steps if s["label"] == "UNTESTABLE"]
    print(f"Tested: {len(tested)}, Untested: {len(untested)}")

    # Sample 100 from tested steps for audit
    random.seed(42)
    sample_size = min(100, len(tested))
    sample = random.sample(tested, sample_size)

    print(f"\n{'='*60}")
    print(f"AUDIT SAMPLE: {sample_size} tested steps")
    print(f"{'='*60}")

    # Automated audit checks
    type_counts = Counter(s["claim_type"] for s in sample)
    label_counts = Counter(s["label"] for s in sample)
    indep_counts = Counter(s["independence"] for s in sample)

    print(f"\n--- Claim type distribution ---")
    for ct, n in type_counts.most_common():
        print(f"  {ct:<20} {n:>4}")

    print(f"\n--- Label distribution ---")
    for label, n in label_counts.most_common():
        print(f"  {label:<10} {n:>4}")

    print(f"\n--- Independence ---")
    for ind, n in indep_counts.most_common():
        print(f"  {ind:<15} {n:>4}")

    # Check for suspicious patterns
    print(f"\n--- Automated audit checks ---")

    # 1. PASS steps where answer is wrong (potential false pass)
    pass_wrong = [s for s in sample if s["label"] == "PASS" and not s["answer_correct"]]
    print(f"  PASS + answer wrong: {len(pass_wrong)} (potential false passes)")

    # 2. FAIL steps where answer is correct (potential false fail)
    fail_correct = [s for s in sample if s["label"] == "FAIL" and s["answer_correct"]]
    print(f"  FAIL + answer correct: {len(fail_correct)} (potential false fails)")

    # 3. "answer" type steps — most reliable since they compare to gold
    answer_steps = [s for s in sample if s["claim_type"] == "answer"]
    answer_pass_correct = sum(1 for s in answer_steps if s["label"] == "PASS" and s["answer_correct"])
    answer_fail_wrong = sum(1 for s in answer_steps if s["label"] == "FAIL" and not s["answer_correct"])
    answer_pass_wrong = sum(1 for s in answer_steps if s["label"] == "PASS" and not s["answer_correct"])
    answer_fail_correct = sum(1 for s in answer_steps if s["label"] == "FAIL" and s["answer_correct"])
    print(f"\n  Final answer checks (gold-oracle, should be perfect):")
    print(f"    PASS + correct: {answer_pass_correct}")
    print(f"    FAIL + wrong: {answer_fail_wrong}")
    print(f"    PASS + wrong (false pass): {answer_pass_wrong}")
    print(f"    FAIL + correct (false fail): {answer_fail_correct}")

    # Full audit for non-answer tested steps
    non_answer = [s for s in sample if s["claim_type"] != "answer"]
    na_pass_correct = sum(1 for s in non_answer if s["label"] == "PASS" and s["answer_correct"])
    na_pass_wrong = sum(1 for s in non_answer if s["label"] == "PASS" and not s["answer_correct"])
    na_fail_correct = sum(1 for s in non_answer if s["label"] == "FAIL" and s["answer_correct"])
    na_fail_wrong = sum(1 for s in non_answer if s["label"] == "FAIL" and not s["answer_correct"])

    print(f"\n  Non-answer step checks (intermediate verification):")
    print(f"    PASS + correct answer: {na_pass_correct}")
    print(f"    PASS + wrong answer: {na_pass_wrong} (may be OK — step could be correct even if answer is wrong)")
    print(f"    FAIL + correct answer: {na_fail_correct} (potential false fail)")
    print(f"    FAIL + wrong answer: {na_fail_wrong}")

    # Compute automated precision/recall estimates
    # For FINAL_ANSWER steps, we have ground truth
    if answer_steps:
        answer_precision = (answer_pass_correct + answer_fail_wrong) / len(answer_steps) if answer_steps else 0
        print(f"\n  Final answer label accuracy: {answer_precision:.1%} ({answer_pass_correct + answer_fail_wrong}/{len(answer_steps)})")

    # Save audit
    audit_output = {
        "sample_size": sample_size,
        "type_distribution": dict(type_counts),
        "label_distribution": dict(label_counts),
        "independence_distribution": dict(indep_counts),
        "pass_wrong_answer": len(pass_wrong),
        "fail_correct_answer": len(fail_correct),
        "answer_step_accuracy": (answer_pass_correct + answer_fail_wrong) / max(len(answer_steps), 1),
        "sample": sample[:20],  # First 20 for review
    }

    out_path = RESULTS_DIR / "pbt_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(audit_output, f, indent=2)
    print(f"\nSaved audit to {out_path}")


if __name__ == "__main__":
    main()
