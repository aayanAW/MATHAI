"""Standard Chain-of-Thought baseline.

Generates a single CoT solution and extracts the answer.
Also implements best-of-N and majority voting.
"""
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple

from ..eval.answer_check import answers_equivalent, extract_model_answer

GenerateFn = Callable[[str], str]

COT_PROMPT = """Solve the following math problem step by step. Show your work clearly and state your final answer as \\boxed{{answer}}.

Problem: {problem}"""

SELF_CORRECT_PROMPT = """You previously solved this problem but may have made an error. Review your solution carefully, identify any mistakes, and provide a corrected solution.

Problem: {problem}

Your previous solution:
{previous_solution}

Carefully review each step. If you find an error, correct it. If the solution is correct, confirm it. State your final answer as \\boxed{{answer}}."""


def run_cot(
    problem: str,
    gold_answer: str,
    solver_fn: GenerateFn,
) -> Dict:
    """Run standard CoT (pass@1).

    Returns dict with answer, correctness, and solution.
    """
    prompt = COT_PROMPT.format(problem=problem)
    response = solver_fn(prompt)
    predicted = extract_model_answer(response)
    correct, method = answers_equivalent(predicted, gold_answer)

    return {
        "answer_predicted": predicted,
        "answer_correct": correct,
        "check_method": method,
        "solution": response,
    }


def run_best_of_n(
    problem: str,
    gold_answer: str,
    solver_fn: GenerateFn,
    n: int = 8,
) -> Dict:
    """Run best-of-N sampling.

    Generates N solutions, returns the one with correct answer
    (or first if none correct).
    """
    solutions = []
    for _ in range(n):
        prompt = COT_PROMPT.format(problem=problem)
        response = solver_fn(prompt)
        predicted = extract_model_answer(response)
        correct, method = answers_equivalent(predicted, gold_answer)
        solutions.append({
            "answer": predicted,
            "correct": correct,
            "solution": response,
        })

    # Count correct
    n_correct = sum(1 for s in solutions if s["correct"])

    # Return best correct solution (or first)
    correct_solutions = [s for s in solutions if s["correct"]]
    best = correct_solutions[0] if correct_solutions else solutions[0]

    return {
        "answer_predicted": best["answer"],
        "answer_correct": best["correct"],
        "n_samples": n,
        "n_correct": n_correct,
        "all_answers": [s["answer"] for s in solutions],
    }


def run_majority_vote(
    problem: str,
    gold_answer: str,
    solver_fn: GenerateFn,
    n: int = 8,
) -> Dict:
    """Run majority voting over N samples.

    Generates N solutions, takes the most common answer.
    """
    answers = []
    solutions = []
    for _ in range(n):
        prompt = COT_PROMPT.format(problem=problem)
        response = solver_fn(prompt)
        predicted = extract_model_answer(response)
        answers.append(predicted)
        solutions.append(response)

    # Majority vote with symbolic equivalence grouping
    # Group answers by symbolic equivalence to avoid vote-splitting
    # e.g., "1/2", "0.5", "\frac{1}{2}" should all be in the same group
    if not answers:
        return {"answer_predicted": "", "answer_correct": False}

    groups: List[Tuple[str, int]] = []  # (canonical_answer, count)
    for ans in answers:
        merged = False
        for i, (canonical, count) in enumerate(groups):
            eq, _ = answers_equivalent(ans, canonical)
            if eq:
                groups[i] = (canonical, count + 1)
                merged = True
                break
        if not merged:
            groups.append((ans, 1))

    # Pick the group with the most votes
    groups.sort(key=lambda x: x[1], reverse=True)
    majority_answer = groups[0][0]
    correct, method = answers_equivalent(majority_answer, gold_answer)

    n_correct = sum(
        1 for a in answers
        if answers_equivalent(a, gold_answer)[0]
    )

    return {
        "answer_predicted": majority_answer,
        "answer_correct": correct,
        "n_samples": n,
        "n_correct": n_correct,
        "n_unique_answers": len(counter),
        "majority_count": counter.most_common(1)[0][1],
        "all_answers": answers,
    }


def run_self_correction(
    problem: str,
    gold_answer: str,
    solver_fn: GenerateFn,
    n_rounds: int = 1,
) -> Dict:
    """Run self-correction baseline.

    Model generates a solution, then reviews and corrects it.
    """
    # Initial solution
    prompt = COT_PROMPT.format(problem=problem)
    solution = solver_fn(prompt)

    for round_num in range(n_rounds):
        # Self-correct
        correct_prompt = SELF_CORRECT_PROMPT.format(
            problem=problem,
            previous_solution=solution,
        )
        solution = solver_fn(correct_prompt)

    predicted = extract_model_answer(solution)
    correct, method = answers_equivalent(predicted, gold_answer)

    return {
        "answer_predicted": predicted,
        "answer_correct": correct,
        "n_correction_rounds": n_rounds,
        "solution": solution,
    }
