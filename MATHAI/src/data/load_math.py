"""Load and sample from the MATH dataset (Hendrycks et al.)."""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset


# MATH has 7 subjects
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# MATH has 5 difficulty levels
MATH_LEVELS = [1, 2, 3, 4, 5]


def load_math_test() -> List[Dict]:
    """Load the full MATH test set from HuggingFace.

    The dataset is split by subject on EleutherAI/hendrycks_math.
    We load all 7 subjects and merge them.

    Returns list of dicts with keys:
        problem, solution, answer, level, type (subject)
    """
    problems = []
    hf_subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    for subject in hf_subjects:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
            for item in ds:
                answer = extract_boxed_answer(item["solution"])
                level_str = item["level"]
                # Parse "Level 3" -> 3
                level = int(level_str.replace("Level ", "")) if "Level" in level_str else 0
                problems.append({
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "answer": answer,
                    "level": level,
                    "type": subject,
                    "id": f"math_{len(problems)}",
                })
        except Exception as e:
            print(f"Warning: Failed to load subject '{subject}': {e}")

    return problems


def extract_boxed_answer(solution: str) -> str:
    """Extract the \\boxed{...} answer from a MATH solution string.

    Handles nested braces correctly.
    """
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        # Try \\boxed without braces (rare)
        idx = solution.rfind("\\boxed ")
        if idx != -1:
            return solution[idx + 7:].strip().split()[0]
        return ""

    # Find matching closing brace
    start = idx + 7  # len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1

    return solution[start:i - 1]


def stratified_sample(
    problems: List[Dict],
    n_per_level: int = 60,
    seed: int = 42,
) -> List[Dict]:
    """Sample problems stratified by difficulty level.

    Args:
        problems: Full problem list.
        n_per_level: Number of problems per difficulty level.
        seed: Random seed for reproducibility.

    Returns:
        Stratified sample of n_per_level * 5 problems.
    """
    rng = random.Random(seed)
    by_level: Dict[int, List[Dict]] = {lv: [] for lv in MATH_LEVELS}
    for p in problems:
        by_level[p["level"]].append(p)

    sampled = []
    for lv in MATH_LEVELS:
        pool = by_level[lv]
        n = min(n_per_level, len(pool))
        sampled.extend(rng.sample(pool, n))

    return sampled


def stratified_sample_by_subject_and_level(
    problems: List[Dict],
    n_per_cell: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """Sample problems stratified by BOTH subject and difficulty level.

    Creates a balanced grid: 7 subjects × 5 levels × n_per_cell.
    If a cell has fewer than n_per_cell, takes all available.

    Returns sampled problems with balanced representation.
    """
    rng = random.Random(seed)
    grid: Dict[Tuple[str, int], List[Dict]] = {}
    for p in problems:
        key = (p["type"], p["level"])
        grid.setdefault(key, []).append(p)

    sampled = []
    for subj in MATH_SUBJECTS:
        for lv in MATH_LEVELS:
            pool = grid.get((subj, lv), [])
            n = min(n_per_cell, len(pool))
            if n > 0:
                sampled.extend(rng.sample(pool, n))

    return sampled


def save_problems(problems: List[Dict], path: str) -> None:
    """Save problems to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(problems, f, indent=2)


def load_problems(path: str) -> List[Dict]:
    """Load problems from JSON."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Loading MATH test set...")
    problems = load_math_test()
    print(f"Loaded {len(problems)} problems")

    # Print distribution
    from collections import Counter
    level_counts = Counter(p["level"] for p in problems)
    subject_counts = Counter(p["type"] for p in problems)
    print("\nBy level:", dict(sorted(level_counts.items())))
    print("\nBy subject:", dict(sorted(subject_counts.items())))

    # Create stratified sample
    sample = stratified_sample(problems, n_per_level=60)
    print(f"\nStratified sample: {len(sample)} problems")
    sample_levels = Counter(p["level"] for p in sample)
    print("Sample by level:", dict(sorted(sample_levels.items())))

    # Save
    save_problems(problems, "results/math_test_full.json")
    save_problems(sample, "results/math_test_sample_300.json")
    print("\nSaved to results/")
