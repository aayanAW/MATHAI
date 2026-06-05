"""Experiment 7: SymCode Baseline.

Implements the SymCode approach: monolithic Python/SymPy solution with
self-debugging. The model writes the ENTIRE solution as Python code
(no NL reasoning), then debugs if execution fails.

This is baseline B5 from the plan — the strongest code-based baseline.
"""
import json
import re
import sys
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp7-symcode")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.49.0", "vllm==0.7.3", "numpy<2")
)
model_volume = modal.Volume.from_name("exever-models", create_if_missing=True)

SYMCODE_PROMPT = """Solve the following math problem by writing a Python/SymPy script.

Your script should:
1. Import sympy at the top
2. Set up the problem using symbolic computation
3. Solve it step by step using SymPy functions
4. Print the final answer with: print("ANSWER:", answer)

Write ONLY the Python code, nothing else.

Problem: {problem}

```python"""

SYMCODE_DEBUG_PROMPT = """The following Python/SymPy script was supposed to solve a math problem but failed with an error.

Problem: {problem}

Script:
```python
{script}
```

Error:
{error}

Fix the script. Write ONLY the corrected Python code, nothing else.

```python"""


def extract_script(response: str) -> str:
    """Extract Python code from response."""
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
def run_symcode_inference(problems_json: str) -> str:
    """Generate SymCode solutions + self-debug."""
    import json as _json
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    print("Loading Qwen2.5-Math-7B for SymCode...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    )
    from modal import Volume
    Volume.from_name("exever-models").commit()

    # Phase 1: Initial code generation
    params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, stop=["```"])
    prompts = [SYMCODE_PROMPT.format(problem=p["problem"]) for p in problems]

    print(f"Generating {len(prompts)} SymCode solutions...")
    t0 = __import__("time").time()
    outputs = llm.generate(prompts, params)
    elapsed = __import__("time").time() - t0
    print(f"Done in {elapsed:.1f}s")

    initial_scripts = [o.outputs[0].text for o in outputs]

    return _json.dumps({
        "scripts": initial_scripts,
        "time": elapsed,
    })


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=3600,
    scaledown_window=120,
)
def run_symcode_debug(debug_inputs_json: str) -> str:
    """Self-debug failed SymCode scripts."""
    import json as _json
    from vllm import LLM, SamplingParams

    inputs = _json.loads(debug_inputs_json)
    if not inputs:
        return _json.dumps({"debugged": []})

    print(f"Loading model for SymCode debug ({len(inputs)} scripts)...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    )

    params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, stop=["```"])
    prompts = [inp["debug_prompt"] for inp in inputs]

    print(f"Generating {len(prompts)} debug fixes...")
    outputs = llm.generate(prompts, params)

    return _json.dumps({
        "debugged": [o.outputs[0].text for o in outputs],
    })


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.exever.executor import execute_verification_script

    # Load MATH-500
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems")

    # Phase 1: Generate initial SymCode scripts
    print("\n[Phase 1] Generating SymCode solutions...")
    raw = run_symcode_inference.remote(json.dumps(problems))
    data = json.loads(raw)
    scripts = [extract_script(s) for s in data["scripts"]]

    # Phase 2: Execute initial scripts
    print(f"\n[Phase 2] Executing {len(scripts)} scripts...")
    results = []
    debug_needed = []
    n_pass = 0
    n_fail = 0

    for i, (prob, script) in enumerate(zip(problems, scripts)):
        if not script.strip():
            results.append({
                "id": prob["id"], "verdict": "EMPTY",
                "answer": "", "correct": False,
            })
            continue

        exec_result = execute_verification_script(script, timeout=30)

        # Extract answer from stdout (handle multiple formats)
        ans = exec_result.answer_extracted or ""
        if not ans and exec_result.stdout:
            # Try additional patterns the model might use
            import re as _re
            for pattern in [r"ANS:\s*(.+?)(?:\n|$)", r"ans:\s*(.+?)(?:\n|$)",
                           r"Result:\s*(.+?)(?:\n|$)", r"answer\s*[:=]\s*(.+?)(?:\n|$)"]:
                m = _re.search(pattern, exec_result.stdout, _re.IGNORECASE)
                if m:
                    ans = m.group(1).strip()
                    break
            if not ans:
                # Last resort: take last non-empty line of stdout
                lines = [l.strip() for l in exec_result.stdout.strip().split('\n') if l.strip()]
                if lines:
                    ans = lines[-1]

        if exec_result.success and ans:
            correct, _ = answers_equivalent(ans, prob["answer"])
            results.append({
                "id": prob["id"], "verdict": "PASS",
                "answer": ans, "correct": correct,
            })
            n_pass += 1
        else:
            error = exec_result.stderr[:500] if exec_result.stderr else "No output"
            debug_needed.append({
                "idx": i,
                "debug_prompt": SYMCODE_DEBUG_PROMPT.format(
                    problem=prob["problem"],
                    script=script[:1500],
                    error=error[:300],
                ),
            })
            results.append({
                "id": prob["id"], "verdict": "FAIL_INITIAL",
                "answer": "", "correct": False,
            })
            n_fail += 1

    print(f"Initial: {n_pass} pass, {n_fail} fail, {len(results)-n_pass-n_fail} empty")

    # Phase 3: Self-debug (2 rounds)
    for debug_round in range(2):
        if not debug_needed:
            break

        print(f"\n[Debug Round {debug_round+1}] Debugging {len(debug_needed)} scripts...")
        debug_raw = run_symcode_debug.remote(json.dumps(debug_needed))
        debug_data = json.loads(debug_raw)

        next_debug = []
        n_fixed = 0

        for di, (info, debugged_resp) in enumerate(zip(debug_needed, debug_data["debugged"])):
            idx = info["idx"]
            prob = problems[idx]
            debugged_script = extract_script(debugged_resp)

            if not debugged_script.strip():
                next_debug.append(info)
                continue

            exec_result = execute_verification_script(debugged_script, timeout=30)

            # Extract answer (handle multiple formats)
            ans = exec_result.answer_extracted or ""
            if not ans and exec_result.stdout:
                import re as _re
                for pattern in [r"ANS:\s*(.+?)(?:\n|$)", r"ans:\s*(.+?)(?:\n|$)",
                               r"Result:\s*(.+?)(?:\n|$)"]:
                    m = _re.search(pattern, exec_result.stdout, _re.IGNORECASE)
                    if m:
                        ans = m.group(1).strip()
                        break
                if not ans:
                    lines = [l.strip() for l in exec_result.stdout.strip().split('\n') if l.strip()]
                    if lines:
                        ans = lines[-1]

            if exec_result.success and ans:
                correct, _ = answers_equivalent(ans, prob["answer"])
                results[idx] = {
                    "id": prob["id"], "verdict": f"FIXED_ROUND_{debug_round+1}",
                    "answer": ans, "correct": correct,
                }
                n_fixed += 1
            else:
                error = exec_result.stderr[:500] if exec_result.stderr else "No output"
                next_debug.append({
                    "idx": idx,
                    "debug_prompt": SYMCODE_DEBUG_PROMPT.format(
                        problem=prob["problem"],
                        script=debugged_script[:1500],
                        error=error[:300],
                    ),
                })

        print(f"  Fixed: {n_fixed}/{len(debug_needed)}")
        debug_needed = next_debug

    # === Results ===
    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / len(problems)
    verdicts = {}
    for r in results:
        v = r["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    print(f"\n{'='*60}")
    print(f"SYMCODE BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Overall accuracy: {accuracy:.3f} ({n_correct}/{len(problems)})")
    print(f"Verdicts: {verdicts}")

    # By level
    by_level = {}
    for lv in [1, 2, 3, 4, 5]:
        lv_results = [(i, r) for i, r in enumerate(results) if problems[i]["level"] == lv]
        if not lv_results:
            continue
        n = len(lv_results)
        correct = sum(1 for _, r in lv_results if r["correct"])
        by_level[lv] = {"accuracy": correct / n, "n": n}
        print(f"  L{lv}: {correct/n:.3f} ({correct}/{n})")

    # By subject
    by_subject = {}
    subjects = sorted(set(p["type"] for p in problems))
    for subj in subjects:
        s_results = [(i, r) for i, r in enumerate(results) if problems[i]["type"] == subj]
        if not s_results:
            continue
        n = len(s_results)
        correct = sum(1 for _, r in s_results if r["correct"])
        by_subject[subj] = {"accuracy": correct / n, "n": n}
        print(f"  {subj}: {correct/n:.3f} ({correct}/{n})")

    # Save
    output = {
        "experiment": "exp7_symcode_baseline",
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "n_problems": len(problems),
        "accuracy": accuracy,
        "verdicts": verdicts,
        "by_level": {str(k): v for k, v in by_level.items()},
        "by_subject": by_subject,
        "results": results,
    }
    with open(RESULTS_DIR / "exp7_symcode_baseline.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'exp7_symcode_baseline.json'}")
