"""Experiment 6: MATH-500 with DeepSeek-R1-Distill-Qwen-7B as SOLVER.

Tests generalization: does ExeVer work with a different solver model?
DeepSeek-R1-Distill generates long-CoT reasoning traces.
Uses Qwen-7B as verifier (cross-model verification).
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp6-deepseek-solver")

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

VERIFY_PROMPT = """Below is a step-by-step math solution. Write a Python/SymPy script that checks each step.

IMPORTANT: Your script MUST use assert statements to verify each step.

Example of what I want:
```python
from sympy import *
x = symbols('x')
# === STEP 1 ===
eq = x**2 - 5*x + 6
# === STEP 2 ===
claimed = (x-2)*(x-3)
assert expand(claimed - eq) == 0, "FAIL:Step 2: factoring"
# === STEP 3 ===
assert eq.subs(x, 2) == 0, "FAIL:Step 3: x=2 not root"
assert eq.subs(x, 3) == 0, "FAIL:Step 3: x=3 not root"
# === STEP 4 ===
assert 2 + 3 == 5, "FAIL:Step 4: sum"
print("ANSWER:", 5)
```

Rules:
- Start with: from sympy import *
- For each step write: # === STEP k === then assert statements
- Every step MUST have at least one assert
- Use expand() not simplify()
- End with: print("ANSWER:", answer)
- Output ONLY the Python code, nothing else.

Solution to verify:
{solution}

```python"""


def extract_script(response: str) -> str:
    """Extract Python script from model response."""
    if "```" not in response:
        lines = response.split("\n")
        code_lines = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not started:
                if stripped.startswith(("from ", "import ", "#", "x ", "x=",
                                        "def ", "print(")) or stripped == "":
                    started = True
                    code_lines.append(line)
            else:
                code_lines.append(line)
        return "\n".join(code_lines).strip()

    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    return ""


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=7200,
    scaledown_window=120,
)
def run_deepseek_solve(problems_json: str) -> str:
    """Generate solutions with DeepSeek-R1-Distill-Qwen-7B."""
    import json as _json
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    print("Loading DeepSeek-R1-Distill-Qwen-7B as solver...")
    llm = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    )
    from modal import Volume
    Volume.from_name("exever-models").commit()

    # Greedy CoT
    greedy_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)
    prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in problems]

    print(f"Generating {len(prompts)} greedy solutions...")
    t0 = __import__("time").time()
    outputs = llm.generate(prompts, greedy_params)
    elapsed = __import__("time").time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Sampled (4 per problem for baselines)
    sample_params = SamplingParams(max_tokens=2048, temperature=0.7, top_p=0.95, n=4)
    print(f"Generating 4 sampled solutions per problem...")
    t0 = __import__("time").time()
    sample_outputs = llm.generate(prompts, sample_params)
    sample_elapsed = __import__("time").time() - t0
    print(f"Done in {sample_elapsed:.1f}s")

    return _json.dumps({
        "greedy": [o.outputs[0].text for o in outputs],
        "sampled": [[s.text for s in o.outputs] for o in sample_outputs],
        "greedy_time": elapsed,
        "sampled_time": sample_elapsed,
    })


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=3600,
    scaledown_window=120,
)
def run_qwen_verify(solutions_json: str) -> str:
    """Generate verification scripts with Qwen-7B (cross-model)."""
    import json as _json
    from vllm import LLM, SamplingParams

    solutions = _json.loads(solutions_json)

    print("Loading Qwen2.5-Math-7B for verification...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    )

    params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, stop=["```"])
    prompts = [VERIFY_PROMPT.format(solution=sol) for sol in solutions]

    print(f"Generating {len(prompts)} verification scripts...")
    t0 = __import__("time").time()
    outputs = llm.generate(prompts, params)
    elapsed = __import__("time").time() - t0
    print(f"Done in {elapsed:.1f}s")

    return _json.dumps({
        "verify_responses": [o.outputs[0].text for o in outputs],
        "time": elapsed,
    })


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.exever.executor import execute_verification_script
    from src.exever.step_parser import extract_assertions

    # Load MATH-500
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} MATH-500 problems")

    # Phase 1: Generate DeepSeek solutions
    print("\n[Phase 1] Generating DeepSeek solutions...")
    solve_raw = run_deepseek_solve.remote(json.dumps(problems))
    solve_data = json.loads(solve_raw)
    greedy_solutions = solve_data["greedy"]
    sampled_solutions = solve_data["sampled"]

    # Evaluate DeepSeek baseline
    greedy_correct = 0
    greedy_results = []
    for prob, sol in zip(problems, greedy_solutions):
        pred = extract_model_answer(sol)
        correct, _ = answers_equivalent(pred, prob["answer"])
        greedy_correct += int(correct)
        greedy_results.append({"correct": correct, "pred": pred})
    print(f"DeepSeek greedy CoT: {greedy_correct/len(problems):.3f}")

    # Majority@4 and best-of-4
    maj4_correct = 0
    best4_correct = 0
    for prob, samples in zip(problems, sampled_solutions):
        preds = []
        any_correct = False
        for sol in samples:
            pred = extract_model_answer(sol)
            correct, _ = answers_equivalent(pred, prob["answer"])
            preds.append({"pred": pred, "correct": correct})
            if correct:
                any_correct = True
        if any_correct:
            best4_correct += 1
        # Majority vote
        groups = []
        for sp in preds:
            merged = False
            for gi, (can, cnt, cr) in enumerate(groups):
                eq, _ = answers_equivalent(sp["pred"], can)
                if eq:
                    groups[gi] = (can, cnt + 1, cr or sp["correct"])
                    merged = True
                    break
            if not merged:
                groups.append((sp["pred"], 1, sp["correct"]))
        groups.sort(key=lambda x: x[1], reverse=True)
        if groups and groups[0][2]:
            maj4_correct += 1

    print(f"DeepSeek majority@4: {maj4_correct/len(problems):.3f}")
    print(f"DeepSeek best-of-4: {best4_correct/len(problems):.3f}")

    # Phase 2: Verify with Qwen (cross-model)
    print("\n[Phase 2] Generating Qwen verification scripts for DeepSeek solutions...")
    verify_raw = run_qwen_verify.remote(json.dumps(greedy_solutions))
    verify_data = json.loads(verify_raw)
    verify_responses = verify_data["verify_responses"]

    # Phase 3: Execute and evaluate ExeVer
    exever_results = []
    n_pass = 0
    n_fail = 0
    echo_total = 0
    echo_pos = 0

    for prob, sol, ver_resp, gr in zip(problems, greedy_solutions, verify_responses, greedy_results):
        script = extract_script(ver_resp)
        if not script.strip():
            exever_results.append({"verdict": "NO_SCRIPT", "correct": gr["correct"]})
            continue

        try:
            compile(script, "<v>", "exec")
        except SyntaxError:
            exever_results.append({"verdict": "SYNTAX_ERROR", "correct": gr["correct"]})
            continue

        exec_result = execute_verification_script(script, timeout=30)

        if exec_result.success:
            n_pass += 1
            ans = exec_result.answer_extracted or ""
            if ans:
                ans_ok, _ = answers_equivalent(ans, prob["answer"])
                echo_total += 1
                if not ans_ok:
                    echo_pos += 1
            correct = gr["correct"]
            exever_results.append({"verdict": "ALL_PASS", "correct": correct})
        elif exec_result.assertion_error:
            n_fail += 1
            exever_results.append({"verdict": "FAIL", "correct": gr["correct"]})
        else:
            exever_results.append({"verdict": "ERROR", "correct": gr["correct"]})

    exever_acc = sum(1 for r in exever_results if r["correct"]) / len(problems)
    echo_rate = echo_pos / echo_total if echo_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"DEEPSEEK SOLVER RESULTS")
    print(f"{'='*60}")
    print(f"Greedy CoT: {greedy_correct/len(problems):.3f}")
    print(f"Majority@4: {maj4_correct/len(problems):.3f}")
    print(f"Best-of-4: {best4_correct/len(problems):.3f}")
    print(f"ExeVer (Qwen verifier): {exever_acc:.3f}")
    print(f"Echo chamber (cross-model): {echo_rate:.3f} ({echo_pos}/{echo_total})")

    # By level
    by_level = {}
    for lv in [1, 2, 3, 4, 5]:
        idxs = [i for i, p in enumerate(problems) if p["level"] == lv]
        n = len(idxs)
        if n == 0:
            continue
        g = sum(1 for i in idxs if greedy_results[i]["correct"]) / n
        e = sum(1 for i in idxs if exever_results[i]["correct"]) / n
        by_level[lv] = {"greedy": g, "exever": e, "n": n}
        print(f"  L{lv}: greedy={g:.3f} exever={e:.3f}")

    # Save
    output = {
        "experiment": "exp6_deepseek_solver",
        "solver": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "verifier": "Qwen/Qwen2.5-Math-7B-Instruct",
        "n_problems": len(problems),
        "accuracy": {
            "greedy": greedy_correct / len(problems),
            "majority_4": maj4_correct / len(problems),
            "best_of_4": best4_correct / len(problems),
            "exever": exever_acc,
        },
        "echo_chamber_rate": echo_rate,
        "by_level": {str(k): v for k, v in by_level.items()},
    }
    out_path = RESULTS_DIR / "exp6_deepseek_solver.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
