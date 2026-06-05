"""Generate PRM training data from PBT results.

Reads exp15 PBT results and converts to Math-Shepherd format
for both PBT-only and outcome-conditioned variants.

Usage:
    python3 experiments/generate_prm_data.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pbt.pipeline import PBTResult, PBTStepResult
from src.pbt.claim_classifier import ClaimType
from src.pbt.prm_data import generate_training_dataset, save_training_data, dataset_stats

RESULTS_DIR = Path("results")


def reconstruct_pbt_results(exp15_data: dict) -> list:
    """Reconstruct PBTResult objects from saved JSON."""
    results = []
    for d in exp15_data["detailed_results"]:
        step_results = []
        for sr in d.get("step_results", []):
            step_results.append(PBTStepResult(
                step_index=sr["step_index"],
                step_text=f"Step {sr['step_index'] + 1}",  # Placeholder
                claim_type=ClaimType(sr["claim_type"]),
                label=sr["label"],
                independence=sr["independence"],
                validation_notes=sr.get("validation_notes", ""),
            ))

        result = PBTResult(
            problem_id=d["problem_id"],
            problem_text="",
            solution_text="",
            gold_answer="",
            n_steps=d["n_steps"],
            step_results=step_results,
            n_testable=d["n_testable"],
            n_fully_independent=d["n_fully_independent"],
            n_partially_independent=d["n_partially_independent"],
            n_untestable=d["n_untestable"],
            n_passed=d["n_passed"],
            n_failed=d["n_failed"],
            answer_predicted=d["answer_predicted"],
            answer_correct=d["answer_correct"],
            all_tested_pass=d["all_tested_pass"],
            fpvr=d["fpvr"],
        )
        results.append(result)
    return results


def main():
    print("=" * 60)
    print("Generating PRM Training Data from PBT Results")
    print("=" * 60)

    # Load exp15 results
    with open(RESULTS_DIR / "exp15_pbt_math500.json") as f:
        exp15 = json.load(f)

    pbt_results = reconstruct_pbt_results(exp15)
    print(f"Loaded {len(pbt_results)} PBT results")

    # Generate PBT-only variant
    print("\n--- PBT-Only Variant ---")
    pbt_only = generate_training_dataset(pbt_results, "pbt_only")
    stats_only = dataset_stats(pbt_only)
    print(f"  Examples: {stats_only['n_examples']}")
    print(f"  Correct answers: {stats_only['n_correct_answer']}")
    print(f"  Wrong answers: {stats_only['n_wrong_answer']}")
    print(f"  Label +: {stats_only['label_plus']}, -: {stats_only['label_minus']}, masked: {stats_only['label_masked']}")

    save_training_data(pbt_only, str(RESULTS_DIR / "prm_train_pbt_only.json"))
    print(f"  Saved to results/prm_train_pbt_only.json")

    # Generate outcome-conditioned variant
    print("\n--- Outcome-Conditioned Variant ---")
    outcome = generate_training_dataset(pbt_results, "outcome_conditioned")
    stats_oc = dataset_stats(outcome)
    print(f"  Examples: {stats_oc['n_examples']}")
    print(f"  Correct answers: {stats_oc['n_correct_answer']}")
    print(f"  Wrong answers: {stats_oc['n_wrong_answer']}")
    print(f"  Label +: {stats_oc['label_plus']}, -: {stats_oc['label_minus']}, masked: {stats_oc['label_masked']}")

    save_training_data(outcome, str(RESULTS_DIR / "prm_train_outcome_cond.json"))
    print(f"  Saved to results/prm_train_outcome_cond.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
