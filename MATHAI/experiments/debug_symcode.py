"""Quick debug: test what SymCode scripts look like and why they fail."""
import json
import sys
from pathlib import Path

import modal

app = modal.App("debug-symcode")

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


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=600,
    scaledown_window=60,
)
def debug_symcode(problems_json: str) -> str:
    import json as _json
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )

    params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, stop=["```"])
    prompts = [SYMCODE_PROMPT.format(problem=p["problem"]) for p in problems]

    outputs = llm.generate(prompts, params)
    responses = [o.outputs[0].text for o in outputs]

    return _json.dumps({"responses": responses})


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.exever.executor import execute_verification_script

    with open("results/math_test_sample_500.json") as f:
        problems = json.load(f)

    # Test 5 problems
    test_problems = problems[:5]
    raw = debug_symcode.remote(json.dumps(test_problems))
    data = json.loads(raw)

    for i, (prob, resp) in enumerate(zip(test_problems, data["responses"])):
        print(f"\n{'='*60}")
        print(f"Problem {i}: {prob['problem'][:100]}...")
        print(f"Gold answer: {prob['answer']}")
        print(f"\nRaw response ({len(resp)} chars):")
        print(resp[:500])
        print(f"\n--- Execution ---")

        if resp.strip():
            result = execute_verification_script(resp.strip(), timeout=15)
            print(f"Success: {result.success}")
            print(f"Answer: '{result.answer_extracted}'")
            print(f"Stdout: {result.stdout[:200]}")
            print(f"Stderr: {result.stderr[:200]}")
        else:
            print("Empty response!")
