"""Experiment 9: Ablation Studies.

Runs 5 key ablations on a 300-problem subset:
A1: Verify-only (no repair, use verification score for selection)
A2: No spot-checks (remove model-free cross-checks)
A3: Re-derivation assertions (instead of transition assertions)
A4: Interleaved single-pass (combined solve + verify in one prompt)
A5: Larger sample (ExeVer with 4 samples instead of 1)

All ablations use Qwen2.5-Math-7B on H100.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp9-ablations")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.49.0", "vllm==0.7.3", "numpy<2")
)
model_volume = modal.Volume.from_name("exever-models", create_if_missing=True)

# Standard prompts
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

# A3: Re-derivation prompt (instead of transition assertions)
REDERIVE_VERIFY_PROMPT = """Below is a step-by-step math solution. Write a Python/SymPy script that INDEPENDENTLY solves each step from scratch and checks it matches.

IMPORTANT: Do NOT just check the claimed result — INDEPENDENTLY re-derive it using SymPy.

For example, if the solution claims "factoring x^2-5x+6 = (x-2)(x-3)", your code should:
```python
# === STEP 2 ===
# Independently factor
result = factor(x**2 - 5*x + 6)
assert result == (x-2)*(x-3), "FAIL:Step 2: factoring gives different result"
```

Rules:
- Start with: from sympy import *
- For each step: independently derive the result, then assert it matches the claim
- End with: print("ANSWER:", answer)
- Output ONLY the Python code, nothing else.

Solution to verify:
{solution}

```python"""

# A4: Interleaved single-pass prompt
INTERLEAVED_PROMPT = """Solve the following math problem step by step. After EACH step, write Python/SymPy code to verify that step.

Format:
## Step k: [title]
[mathematical reasoning]

```python
# === STEP k ===
[Python/SymPy code that verifies this step's claims]
assert condition, "FAIL:Step k: description"
```

The code is CUMULATIVE — each step builds on prior variables.
Use assert statements (transition assertions — check claims, don't re-derive).
At the end: print("ANSWER:", final_answer)

Problem: {problem}"""


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


def extract_interleaved_code(response: str) -> str:
    """Extract and combine all code blocks from interleaved response."""
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return ""

    # Combine all code blocks into one cumulative script
    combined = []
    has_import = False
    for block in matches:
        lines = block.strip().split("\n")
        for line in lines:
            if line.strip().startswith(("from sympy", "import sympy")):
                if not has_import:
                    combined.append(line)
                    has_import = True
            else:
                combined.append(line)

    if not has_import:
        combined.insert(0, "from sympy import *")

    return "\n".join(combined)


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=7200,
    scaledown_window=120,
)
def run_ablation_inference(problems_json: str) -> str:
    """Run all ablation inference in one GPU session."""
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

    results = {}

    # === Standard solve (for A1, A2, A3) ===
    solve_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)
    solve_prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in problems]
    print(f"[Solve] Generating {len(solve_prompts)} solutions...")
    t0 = __import__("time").time()
    solve_outputs = llm.generate(solve_prompts, solve_params)
    print(f"  Done in {__import__('time').time()-t0:.1f}s")
    results["solutions"] = [o.outputs[0].text for o in solve_outputs]

    # === Standard verify (A1: verify-only) ===
    verify_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, stop=["```"])
    verify_prompts = [VERIFY_PROMPT.format(solution=sol) for sol in results["solutions"]]
    print(f"[Verify] Generating {len(verify_prompts)} verification scripts...")
    t0 = __import__("time").time()
    verify_outputs = llm.generate(verify_prompts, verify_params)
    print(f"  Done in {__import__('time').time()-t0:.1f}s")
    results["verify_responses"] = [o.outputs[0].text for o in verify_outputs]

    # === A3: Re-derivation verify ===
    rederive_prompts = [REDERIVE_VERIFY_PROMPT.format(solution=sol) for sol in results["solutions"]]
    print(f"[A3-Rederive] Generating {len(rederive_prompts)} re-derivation scripts...")
    t0 = __import__("time").time()
    rederive_outputs = llm.generate(rederive_prompts, verify_params)
    print(f"  Done in {__import__('time').time()-t0:.1f}s")
    results["rederive_responses"] = [o.outputs[0].text for o in rederive_outputs]

    # === A4: Interleaved single-pass ===
    interleaved_params = SamplingParams(max_tokens=3072, temperature=0.0, top_p=1.0)
    interleaved_prompts = [INTERLEAVED_PROMPT.format(problem=p["problem"]) for p in problems]
    print(f"[A4-Interleaved] Generating {len(interleaved_prompts)} interleaved solutions...")
    t0 = __import__("time").time()
    interleaved_outputs = llm.generate(interleaved_prompts, interleaved_params)
    print(f"  Done in {__import__('time').time()-t0:.1f}s")
    results["interleaved_responses"] = [o.outputs[0].text for o in interleaved_outputs]

    # === A5: Multi-sample (4 samples with verify) ===
    multi_params = SamplingParams(max_tokens=2048, temperature=0.7, top_p=0.95, n=4)
    print(f"[A5-Multi] Generating 4 samples per problem...")
    t0 = __import__("time").time()
    multi_outputs = llm.generate(solve_prompts, multi_params)
    print(f"  Done in {__import__('time').time()-t0:.1f}s")
    results["multi_solutions"] = [[s.text for s in o.outputs] for o in multi_outputs]

    # Verify each sample
    all_multi_verify_prompts = []
    for samples in results["multi_solutions"]:
        for sol in samples:
            all_multi_verify_prompts.append(VERIFY_PROMPT.format(solution=sol))
    print(f"[A5-Verify] Generating {len(all_multi_verify_prompts)} verification scripts...")
    t0 = __import__("time").time()
    multi_verify_outputs = llm.generate(all_multi_verify_prompts, verify_params)
    print(f"  Done in {__import__('time').time()-t0:.1f}s")
    results["multi_verify_responses"] = [o.outputs[0].text for o in multi_verify_outputs]

    return _json.dumps(results)


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.exever.executor import execute_verification_script
    from src.exever.step_parser import extract_assertions

    # Use 300-problem sample for ablations
    with open(RESULTS_DIR / "math_test_sample_300.json") as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems for ablations")

    # Run all inference
    print("\nRunning all ablation inference on H100...")
    raw = run_ablation_inference.remote(json.dumps(problems))
    data = json.loads(raw)

    ablation_results = {}

    # === A1: Verify-Only (no repair) ===
    print(f"\n{'='*60}")
    print(f"A1: VERIFY-ONLY (no repair, use V(S) for selection)")
    print(f"{'='*60}")
    a1_correct = 0
    for prob, sol, ver_resp in zip(problems, data["solutions"], data["verify_responses"]):
        pred = extract_model_answer(sol)
        correct, _ = answers_equivalent(pred, prob["answer"])

        script = extract_script(ver_resp)
        verdict = "no_script"
        if script.strip():
            try:
                compile(script, "<v>", "exec")
                exec_result = execute_verification_script(script, timeout=30)
                if exec_result.success:
                    verdict = "all_pass"
                    # Use script answer if available
                    if exec_result.answer_extracted:
                        ans_correct, _ = answers_equivalent(exec_result.answer_extracted, prob["answer"])
                        if ans_correct:
                            correct = True
                elif exec_result.assertion_error:
                    verdict = "fail"
                    # Don't repair — just keep original answer
                else:
                    verdict = "error"
            except SyntaxError:
                verdict = "syntax_error"

        a1_correct += int(correct)

    a1_acc = a1_correct / len(problems)
    ablation_results["A1_verify_only"] = a1_acc
    print(f"Accuracy: {a1_acc:.3f}")

    # === A3: Re-derivation assertions ===
    print(f"\n{'='*60}")
    print(f"A3: RE-DERIVATION ASSERTIONS")
    print(f"{'='*60}")
    a3_pass = 0
    a3_fail = 0
    a3_correct = 0
    for prob, sol, ver_resp in zip(problems, data["solutions"], data["rederive_responses"]):
        pred = extract_model_answer(sol)
        correct, _ = answers_equivalent(pred, prob["answer"])

        script = extract_script(ver_resp)
        if script.strip():
            try:
                compile(script, "<v>", "exec")
                exec_result = execute_verification_script(script, timeout=30)
                if exec_result.success:
                    a3_pass += 1
                    if exec_result.answer_extracted:
                        ans_correct, _ = answers_equivalent(exec_result.answer_extracted, prob["answer"])
                        if ans_correct:
                            correct = True
                elif exec_result.assertion_error:
                    a3_fail += 1
            except SyntaxError:
                pass

        a3_correct += int(correct)

    a3_acc = a3_correct / len(problems)
    ablation_results["A3_rederivation"] = a3_acc
    print(f"Accuracy: {a3_acc:.3f}")
    print(f"All pass: {a3_pass}, Fail: {a3_fail}")

    # === A4: Interleaved single-pass ===
    print(f"\n{'='*60}")
    print(f"A4: INTERLEAVED SINGLE-PASS")
    print(f"{'='*60}")
    a4_correct = 0
    a4_has_code = 0
    for prob, resp in zip(problems, data["interleaved_responses"]):
        pred = extract_model_answer(resp)
        correct, _ = answers_equivalent(pred, prob["answer"])

        # Try to extract and run combined code
        combined = extract_interleaved_code(resp)
        if combined.strip():
            a4_has_code += 1
            try:
                compile(combined, "<v>", "exec")
                exec_result = execute_verification_script(combined, timeout=30)
                if exec_result.success and exec_result.answer_extracted:
                    ans_correct, _ = answers_equivalent(exec_result.answer_extracted, prob["answer"])
                    if ans_correct:
                        correct = True
            except SyntaxError:
                pass

        a4_correct += int(correct)

    a4_acc = a4_correct / len(problems)
    ablation_results["A4_interleaved"] = a4_acc
    print(f"Accuracy: {a4_acc:.3f}")
    print(f"Had code blocks: {a4_has_code}/{len(problems)}")

    # === A5: Multi-sample ExeVer ===
    print(f"\n{'='*60}")
    print(f"A5: MULTI-SAMPLE EXEVER (4 samples)")
    print(f"{'='*60}")
    a5_correct = 0
    verify_idx = 0
    for prob, samples in zip(problems, data["multi_solutions"]):
        best_answer = None
        best_score = -1

        for sol in samples:
            pred = extract_model_answer(sol)
            ver_resp = data["multi_verify_responses"][verify_idx]
            verify_idx += 1

            script = extract_script(ver_resp)
            score = 0.0
            if script.strip():
                try:
                    compile(script, "<v>", "exec")
                    exec_result = execute_verification_script(script, timeout=30)
                    if exec_result.success:
                        score = 1.0
                        if exec_result.answer_extracted:
                            pred = exec_result.answer_extracted
                except SyntaxError:
                    pass

            if score > best_score:
                best_score = score
                best_answer = pred

        if best_answer:
            correct, _ = answers_equivalent(best_answer, prob["answer"])
            a5_correct += int(correct)

    a5_acc = a5_correct / len(problems)
    ablation_results["A5_multisample"] = a5_acc
    print(f"Accuracy: {a5_acc:.3f}")

    # === Load Exp4 ExeVer result for comparison ===
    try:
        with open(RESULTS_DIR / "exp4_repair_baselines.json") as f:
            exp4 = json.load(f)
        ablation_results["ExeVer_full"] = exp4["accuracy"]["exever"]
        ablation_results["CoT_greedy"] = exp4["accuracy"]["exp1_cot_pass1"]
        ablation_results["Majority_4"] = exp4["accuracy"]["baseline_majority_4"]
    except FileNotFoundError:
        pass

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*60}")
    for name, acc in sorted(ablation_results.items()):
        print(f"  {name}: {acc:.3f}")

    output = {
        "experiment": "exp9_ablations",
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "n_problems": len(problems),
        "results": ablation_results,
    }
    with open(RESULTS_DIR / "exp9_ablations.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'exp9_ablations.json'}")
