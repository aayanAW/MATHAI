"""Experiment 8: Self-Correction Baseline.

The model reviews its own solution and corrects errors WITHOUT code execution.
This is baseline B4 — tests whether code execution is actually necessary.

Setup: Generate solution → ask model to review → generate corrected solution.
Total: 2 model calls (same compute budget as ExeVer without repair).
"""
import json
import re
import sys
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp8-selfcorrect")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.49.0", "vllm==0.7.3", "numpy<2")
)
model_volume = modal.Volume.from_name("exever-models", create_if_missing=True)

SOLVE_PROMPT = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

## Step 2: [brief title]
[reasoning and computation for this step]

...continue for all steps...

At the end, state your final answer as: The answer is \\boxed{{answer}}.

Problem: {problem}"""

REVIEW_PROMPT = """Review the following solution to a math problem. Check each step carefully for errors.

Problem: {problem}

Solution:
{solution}

If you find any errors:
1. Identify which step has the error and explain what's wrong
2. Provide a corrected solution with all steps

If the solution is correct, say "The solution is correct" and restate the answer.

State your final answer as: The answer is \\boxed{{answer}}."""


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=7200,
    scaledown_window=120,
)
def run_selfcorrect(problems_json: str) -> str:
    """Generate solutions + self-correction reviews."""
    import json as _json
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    print("Loading Qwen2.5-Math-7B...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    )
    from modal import Volume
    Volume.from_name("exever-models").commit()

    params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)

    # Phase 1: Initial solutions
    solve_prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in problems]
    print(f"Generating {len(solve_prompts)} initial solutions...")
    t0 = __import__("time").time()
    solve_outputs = llm.generate(solve_prompts, params)
    t1 = __import__("time").time()
    solutions = [o.outputs[0].text for o in solve_outputs]
    print(f"Solutions done in {t1-t0:.1f}s")

    # Phase 2: Self-correction reviews
    review_prompts = [
        REVIEW_PROMPT.format(problem=p["problem"], solution=sol)
        for p, sol in zip(problems, solutions)
    ]
    print(f"Generating {len(review_prompts)} self-correction reviews...")
    t0 = __import__("time").time()
    review_outputs = llm.generate(review_prompts, params)
    t1 = __import__("time").time()
    reviews = [o.outputs[0].text for o in review_outputs]
    print(f"Reviews done in {t1-t0:.1f}s")

    return _json.dumps({
        "solutions": solutions,
        "reviews": reviews,
    })


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer

    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems")

    raw = run_selfcorrect.remote(json.dumps(problems))
    data = json.loads(raw)

    # Evaluate initial solutions
    initial_correct = 0
    for prob, sol in zip(problems, data["solutions"]):
        pred = extract_model_answer(sol)
        correct, _ = answers_equivalent(pred, prob["answer"])
        initial_correct += int(correct)
    initial_acc = initial_correct / len(problems)

    # Evaluate self-corrected solutions
    corrected_correct = 0
    changed = 0
    improved = 0
    degraded = 0
    results = []
    for prob, sol, review in zip(problems, data["solutions"], data["reviews"]):
        init_pred = extract_model_answer(sol)
        init_correct, _ = answers_equivalent(init_pred, prob["answer"])

        corr_pred = extract_model_answer(review)
        corr_correct, _ = answers_equivalent(corr_pred, prob["answer"])
        corrected_correct += int(corr_correct)

        if init_pred != corr_pred:
            changed += 1
            if corr_correct and not init_correct:
                improved += 1
            elif init_correct and not corr_correct:
                degraded += 1

        results.append({
            "id": prob["id"], "level": prob["level"], "type": prob["type"],
            "initial_correct": init_correct, "corrected_correct": corr_correct,
            "changed": init_pred != corr_pred,
        })

    corrected_acc = corrected_correct / len(problems)

    print(f"\n{'='*60}")
    print(f"SELF-CORRECTION BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Initial CoT: {initial_acc:.3f}")
    print(f"After self-correction: {corrected_acc:.3f} ({corrected_acc - initial_acc:+.3f})")
    print(f"Changed: {changed}, Improved: {improved}, Degraded: {degraded}")

    # By level
    by_level = {}
    for lv in [1, 2, 3, 4, 5]:
        lv_r = [r for r in results if r["level"] == lv]
        if not lv_r:
            continue
        n = len(lv_r)
        init = sum(1 for r in lv_r if r["initial_correct"]) / n
        corr = sum(1 for r in lv_r if r["corrected_correct"]) / n
        by_level[lv] = {"initial": init, "corrected": corr, "n": n}
        print(f"  L{lv}: {init:.3f} -> {corr:.3f} ({corr-init:+.3f})")

    output = {
        "experiment": "exp8_selfcorrect",
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "n_problems": len(problems),
        "initial_accuracy": initial_acc,
        "corrected_accuracy": corrected_acc,
        "changed": changed, "improved": improved, "degraded": degraded,
        "by_level": {str(k): v for k, v in by_level.items()},
        "results": results,
    }
    with open(RESULTS_DIR / "exp8_selfcorrect.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'exp8_selfcorrect.json'}")
