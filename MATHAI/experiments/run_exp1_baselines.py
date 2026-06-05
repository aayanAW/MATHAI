"""Experiment 1: Baseline CoT on 300 MATH problems via Modal H100.

Self-contained Modal app that loads the model, runs all problems, returns results.
"""
import json
import sys
import time
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp1")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.49.0", "vllm==0.7.3", "numpy<2")
)
model_volume = modal.Volume.from_name("exever-models", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=3600,
    scaledown_window=60,
)
def run_cot_batch(problems_json: str) -> str:
    """Run CoT on all problems inside Modal. Returns JSON results."""
    import json as _json
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    # Load model
    print(f"Loading Qwen2.5-Math-7B-Instruct...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    from modal import Volume
    Volume.from_name("exever-models").commit()

    params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)

    # Build prompts
    SOLVE_TMPL = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

At the end, state your final answer as: The answer is \\boxed{{answer}}.

Problem: {problem}"""

    prompts = [SOLVE_TMPL.format(problem=p["problem"]) for p in problems]

    # Run inference
    print(f"Running inference on {len(prompts)} problems...")
    t0 = __import__("time").time()
    outputs = llm.generate(prompts, params)
    elapsed = __import__("time").time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    # Build results
    results = []
    for i, (p, out) in enumerate(zip(problems, outputs)):
        response = out.outputs[0].text
        results.append({
            "id": p["id"],
            "level": p["level"],
            "type": p["type"],
            "gold_answer": p["answer"],
            "response": response,
        })

    return _json.dumps({"results": results, "inference_time": elapsed})


@app.local_entrypoint()
def main():
    # Import local modules only in the local entrypoint (not in the remote function)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer

    # Load problems
    with open(RESULTS_DIR / "math_test_sample_300.json") as f:
        problems = json.load(f)

    print(f"Sending {len(problems)} problems to Modal H100...")
    raw_result = run_cot_batch.remote(json.dumps(problems))
    data = json.loads(raw_result)
    inference_time = data["inference_time"]

    print(f"Inference completed in {inference_time:.1f}s")
    print("Evaluating answers locally...")

    # Evaluate locally (answer checking is CPU-only)
    results = data["results"]
    correct_count = 0
    for r in results:
        predicted = extract_model_answer(r["response"])
        is_correct, method = answers_equivalent(predicted, r["gold_answer"])
        r["predicted_answer"] = predicted
        r["correct"] = is_correct
        r["check_method"] = method
        if is_correct:
            correct_count += 1

    pass_at_1 = correct_count / len(results)
    print(f"\n{'='*50}")
    print(f"EXPERIMENT 1 RESULTS: Qwen2.5-Math-7B CoT")
    print(f"{'='*50}")
    print(f"Overall pass@1: {pass_at_1:.1%} ({correct_count}/{len(results)})")

    # By level
    from collections import defaultdict
    by_level = defaultdict(list)
    by_subject = defaultdict(list)
    for r in results:
        by_level[r["level"]].append(r["correct"])
        by_subject[r["type"]].append(r["correct"])

    print("\nBy difficulty level:")
    for lv in sorted(by_level):
        acc = sum(by_level[lv]) / len(by_level[lv])
        n = sum(by_level[lv])
        print(f"  Level {lv}: {acc:.1%} ({n}/{len(by_level[lv])})")

    print("\nBy subject:")
    for subj in sorted(by_subject):
        acc = sum(by_subject[subj]) / len(by_subject[subj])
        n = sum(by_subject[subj])
        print(f"  {subj}: {acc:.1%} ({n}/{len(by_subject[subj])})")

    # Save
    output = {
        "experiment": "exp1_baseline",
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "n_problems": len(results),
        "pass_at_1": pass_at_1,
        "inference_time_s": inference_time,
        "by_level": {str(lv): sum(v)/len(v) for lv, v in sorted(by_level.items())},
        "by_subject": {s: sum(v)/len(v) for s, v in sorted(by_subject.items())},
        "results": results,
    }
    out_path = RESULTS_DIR / "exp1_baseline_qwen7b.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
