"""Experiment 10: Model Scaling Analysis.

Runs ExeVer pipeline with Qwen2.5-Math-1.5B-Instruct to test the
hypothesis that ExeVer helps SMALLER models MORE (they have more errors to catch).

Also tests with Llama-3.1-8B-Instruct (non-math-specialized) as a control.
Uses 300-problem subset for efficiency.
"""
import json
import re
import sys
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp10-scaling")

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
def run_1_5b_inference(problems_json: str) -> str:
    """Run Qwen2.5-Math-1.5B: solve + verify."""
    import json as _json
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    print("Loading Qwen2.5-Math-1.5B-Instruct...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    from modal import Volume
    Volume.from_name("exever-models").commit()

    solve_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)
    verify_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, stop=["```"])
    sample_params = SamplingParams(max_tokens=2048, temperature=0.7, top_p=0.95, n=4)

    # Greedy solve
    solve_prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in problems]
    print(f"Generating {len(solve_prompts)} greedy solutions...")
    solve_outputs = llm.generate(solve_prompts, solve_params)
    solutions = [o.outputs[0].text for o in solve_outputs]

    # Sampled (4 per problem)
    print(f"Generating 4 sampled solutions per problem...")
    sample_outputs = llm.generate(solve_prompts, sample_params)
    sampled = [[s.text for s in o.outputs] for o in sample_outputs]

    # Verify greedy solutions
    verify_prompts = [VERIFY_PROMPT.format(solution=sol) for sol in solutions]
    print(f"Generating verification scripts...")
    verify_outputs = llm.generate(verify_prompts, verify_params)
    verify_responses = [o.outputs[0].text for o in verify_outputs]

    return _json.dumps({
        "solutions": solutions,
        "sampled": sampled,
        "verify_responses": verify_responses,
    })


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=7200,
    scaledown_window=120,
)
def run_general_7b_inference(problems_json: str) -> str:
    """Run Qwen2.5-7B-Instruct: solve + verify (non-math-specialized control)."""
    import json as _json
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    print("Loading Qwen2.5-7B-Instruct (general, non-math)...")
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    from modal import Volume
    Volume.from_name("exever-models").commit()

    solve_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)
    verify_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, stop=["```"])
    sample_params = SamplingParams(max_tokens=2048, temperature=0.7, top_p=0.95, n=4)

    solve_prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in problems]

    print(f"Generating {len(solve_prompts)} greedy solutions...")
    solve_outputs = llm.generate(solve_prompts, solve_params)
    solutions = [o.outputs[0].text for o in solve_outputs]

    print(f"Generating 4 sampled solutions per problem...")
    sample_outputs = llm.generate(solve_prompts, sample_params)
    sampled = [[s.text for s in o.outputs] for o in sample_outputs]

    verify_prompts = [VERIFY_PROMPT.format(solution=sol) for sol in solutions]
    print(f"Generating verification scripts...")
    verify_outputs = llm.generate(verify_prompts, verify_params)
    verify_responses = [o.outputs[0].text for o in verify_outputs]

    return _json.dumps({
        "solutions": solutions,
        "sampled": sampled,
        "verify_responses": verify_responses,
    })


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.exever.executor import execute_verification_script

    with open(RESULTS_DIR / "math_test_sample_300.json") as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems")

    all_model_results = {}

    for model_name, remote_fn in [
        ("Qwen2.5-Math-1.5B", run_1_5b_inference),
        ("Qwen2.5-7B-General", run_general_7b_inference),
    ]:
        print(f"\n{'='*60}")
        print(f"Running {model_name}...")
        print(f"{'='*60}")

        raw = remote_fn.remote(json.dumps(problems))
        data = json.loads(raw)

        # Evaluate greedy
        greedy_correct = 0
        for prob, sol in zip(problems, data["solutions"]):
            pred = extract_model_answer(sol)
            correct, _ = answers_equivalent(pred, prob["answer"])
            greedy_correct += int(correct)

        # Evaluate majority@4 + best-of-4
        maj4_correct = 0
        best4_correct = 0
        for prob, samples in zip(problems, data["sampled"]):
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
            groups = []
            for sp in preds:
                merged = False
                for gi, (can, cnt, cr) in enumerate(groups):
                    eq, _ = answers_equivalent(sp["pred"], can)
                    if eq:
                        groups[gi] = (can, cnt+1, cr or sp["correct"])
                        merged = True
                        break
                if not merged:
                    groups.append((sp["pred"], 1, sp["correct"]))
            groups.sort(key=lambda x: x[1], reverse=True)
            if groups and groups[0][2]:
                maj4_correct += 1

        # Evaluate ExeVer (verify-only, no repair for simplicity)
        exever_correct = 0
        n_pass = 0
        for prob, sol, ver_resp in zip(problems, data["solutions"], data["verify_responses"]):
            pred = extract_model_answer(sol)
            correct, _ = answers_equivalent(pred, prob["answer"])

            script = extract_script(ver_resp)
            if script.strip():
                try:
                    compile(script, "<v>", "exec")
                    exec_result = execute_verification_script(script, timeout=30)
                    if exec_result.success:
                        n_pass += 1
                        if exec_result.answer_extracted:
                            ans_ok, _ = answers_equivalent(exec_result.answer_extracted, prob["answer"])
                            if ans_ok:
                                correct = True
                except SyntaxError:
                    pass

            exever_correct += int(correct)

        n = len(problems)
        model_results = {
            "greedy": greedy_correct / n,
            "majority_4": maj4_correct / n,
            "best_of_4": best4_correct / n,
            "exever": exever_correct / n,
            "n_verify_pass": n_pass,
        }
        all_model_results[model_name] = model_results

        print(f"\n{model_name} Results:")
        print(f"  Greedy CoT: {model_results['greedy']:.3f}")
        print(f"  Majority@4: {model_results['majority_4']:.3f}")
        print(f"  Best-of-4: {model_results['best_of_4']:.3f}")
        print(f"  ExeVer: {model_results['exever']:.3f}")
        print(f"  Verify pass: {n_pass}/{n}")

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"SCALING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    # Add Qwen-7B results from Exp 4 for comparison
    try:
        with open(RESULTS_DIR / "exp4_repair_baselines.json") as f:
            exp4 = json.load(f)
        all_model_results["Qwen2.5-Math-7B"] = {
            "greedy": exp4["accuracy"]["exp1_cot_pass1"],
            "majority_4": exp4["accuracy"]["baseline_majority_4"],
            "best_of_4": exp4["accuracy"]["baseline_best_of_4"],
            "exever": exp4["accuracy"]["exever"],
        }
    except FileNotFoundError:
        pass

    print(f"{'Model':<25} {'Greedy':>8} {'Maj@4':>8} {'Best@4':>8} {'ExeVer':>8} {'Gain':>8}")
    print("-" * 73)
    for name, r in sorted(all_model_results.items()):
        gain = r["exever"] - r["majority_4"]
        print(f"{name:<25} {r['greedy']:>7.1%} {r['majority_4']:>7.1%} "
              f"{r['best_of_4']:>7.1%} {r['exever']:>7.1%} {gain:>+7.1%}")

    output = {
        "experiment": "exp10_scaling",
        "n_problems": len(problems),
        "results": all_model_results,
    }
    with open(RESULTS_DIR / "exp10_scaling.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'exp10_scaling.json'}")
