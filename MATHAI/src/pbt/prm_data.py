"""Convert PBT results to PRM training data.

Two variants:
1. PBT-only: Only tested steps in loss. UNTESTABLE masked.
2. PBT + outcome-conditioned: Tested steps get PBT labels,
   untestable steps get outcome-based labels (correct answer -> +, wrong -> -).
"""
import json
from typing import Dict, List, Optional

from .pipeline import PBTResult


STEP_DELIMITER = " ки"  # Math-Shepherd format delimiter


def pbt_result_to_math_shepherd(
    result: PBTResult,
    variant: str = "pbt_only",  # "pbt_only" or "outcome_conditioned"
) -> Optional[Dict]:
    """Convert a single PBT result to Math-Shepherd training format.

    Math-Shepherd format: solution text with ки delimiters,
    each followed by + (correct) or - (incorrect).

    Args:
        result: PBTResult from run_pbt.
        variant: "pbt_only" masks untestable steps,
                 "outcome_conditioned" labels them based on answer correctness.

    Returns:
        Dict with 'input' (unlabeled), 'label' (labeled), 'task' keys.
        None if no useful labels.
    """
    if not result.step_results:
        return None

    input_parts = []
    label_parts = []
    has_useful_label = False

    for sr in result.step_results:
        step_text = sr.step_text
        input_parts.append(step_text + STEP_DELIMITER)

        if sr.label == "PASS":
            label_parts.append(step_text + STEP_DELIMITER + "+")
            has_useful_label = True
        elif sr.label == "FAIL":
            label_parts.append(step_text + STEP_DELIMITER + "-")
            has_useful_label = True
        elif sr.label == "UNTESTABLE":
            if variant == "pbt_only":
                # EXCLUDE untestable steps entirely from training text
                # (SFT computes loss on all tokens — including '?' would
                # train the model to generate '?' which is not useful)
                label_parts.append(step_text)  # No delimiter, no label
            elif variant == "outcome_conditioned":
                # Label based on final answer correctness
                label_char = "+" if result.answer_correct else "-"
                label_parts.append(step_text + STEP_DELIMITER + label_char)
            else:
                label_parts.append(step_text)  # No label

    if not has_useful_label:
        return None

    return {
        "input": "\n".join(input_parts),
        "label": "\n".join(label_parts),
        "task": "MATH",
        "problem_id": result.problem_id,
        "answer_correct": result.answer_correct,
        "n_tested": result.n_testable,
        "n_untested": result.n_untestable,
        "variant": variant,
    }


def generate_training_dataset(
    pbt_results: List[PBTResult],
    variant: str = "pbt_only",
) -> List[Dict]:
    """Convert a list of PBT results to training data.

    Args:
        pbt_results: List of PBTResult from run_pbt.
        variant: "pbt_only" or "outcome_conditioned".

    Returns:
        List of training examples in Math-Shepherd format.
    """
    dataset = []
    for result in pbt_results:
        example = pbt_result_to_math_shepherd(result, variant)
        if example is not None:
            dataset.append(example)
    return dataset


def save_training_data(dataset: List[Dict], path: str) -> None:
    """Save training data to JSON."""
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


def dataset_stats(dataset: List[Dict]) -> Dict:
    """Compute statistics on the training dataset."""
    n = len(dataset)
    n_correct = sum(1 for d in dataset if d["answer_correct"])
    total_tested = sum(d["n_tested"] for d in dataset)
    total_untested = sum(d["n_untested"] for d in dataset)

    # Count label distribution by splitting on step delimiter
    n_plus = 0
    n_minus = 0
    n_mask = 0
    for d in dataset:
        # Split by step delimiter and check the label char at the end of each segment
        segments = d["label"].split(STEP_DELIMITER)
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            last_char = seg[-1] if seg else ""
            if last_char == "+":
                n_plus += 1
            elif last_char == "-":
                n_minus += 1
            elif last_char == "?":
                n_mask += 1

    return {
        "n_examples": n,
        "n_correct_answer": n_correct,
        "n_wrong_answer": n - n_correct,
        "total_tested_steps": total_tested,
        "total_untested_steps": total_untested,
        "label_plus": n_plus,
        "label_minus": n_minus,
        "label_masked": n_mask,
    }
