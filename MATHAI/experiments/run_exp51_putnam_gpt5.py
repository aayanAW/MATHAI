"""Putnam B-problem proof-mode X-SGRV via Claude Sonnet 4.6.

Extends X-SGRV from find-style to prove-style problems. For each Putnam B
problem we ask Claude to emit a Python SymPy script that either:
  (a) evaluates numerical/symbolic claims at random inputs
  (b) constructs the existence-claimed object and verifies its properties
  (c) outputs UNVERIFIABLE if the problem needs a real proof.

Budget: ~100 problems * ~4k tokens * ($3/M in + $15/M out) ~ $6-8.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import execute_verifier  # type: ignore

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp51_putnam_gpt5.json"
MODEL = "gpt-5-mini"

if "OPENAI_API_KEY" not in os.environ:
    raise SystemExit("ERROR: set OPENAI_API_KEY")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


PROOFMODE_PROMPT = """You are reading a Putnam competition problem that asks for a proof or for a quantity to be determined and justified.

Produce a Python function verify(answer) -> bool that takes a string answer (either a final numeric or symbolic value OR a short natural-language claim like "n=7") and returns True iff that candidate is consistent with the problem's constraints.

Strategies:
- Numeric or symbolic final answer: extract from the input, test the problem's stated constraints. For constraints with free variables, evaluate at 5 random substitutions via `import random; random.seed(42)`.
- Existence problem: try to construct the claimed object in SymPy and verify its properties.
- Inequality: enumerate a bounded range and check.
- If the problem genuinely requires a proof beyond computational checking, output the token UNVERIFIABLE as your entire response (no code block).

OUTPUT FORMAT (one of):
```python
def verify(answer):
    # your check
    return True_or_False
```
or just the word UNVERIFIABLE on its own line.

Problem:
{problem}

Your response:"""


def load_putnam_problems(n_target=100):
    """Use amitayusht/PutnamBench (522 problems, informal + Lean statements)."""
    from datasets import load_dataset
    ds = load_dataset("amitayusht/PutnamBench", split="train")
    print(f"Loaded amitayusht/PutnamBench: {len(ds)} problems")
    # Sample first N that have an informal solution (many do)
    probs = []
    for ex in ds:
        q = ex.get("informal_statement", "") or ""
        a = ex.get("informal_solution", "") or ""
        if q:
            # Extract a simple string answer heuristically
            answer_short = a[:150]
            probs.append({"id": ex.get("name") or f"putnam_{len(probs)}",
                          "problem": q, "answer": answer_short,
                          "tags": ex.get("tags", [])})
            if len(probs) >= n_target:
                break
    return probs, "amitayusht/PutnamBench"


def extract_verifier_claude(problem_text, max_retries=2):
    prompt = PROOFMODE_PROMPT.format(problem=problem_text)
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(model=MODEL, max_completion_tokens=2048,
                                        messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2.0)
                continue
            return ""
    return ""


def parse_script(text):
    import re
    if "UNVERIFIABLE" in text[:200]:
        return None, "unverifiable"
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip(), "ok"
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip(), "ok"
    if "def verify" in text:
        return text.strip(), "ok"
    return None, "no_script"


def main():
    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {"model": MODEL, "results": []}
    done = {r["id"] for r in out["results"]}
    probs, source = load_putnam_problems(n_target=100)
    if not probs:
        print("No Putnam problems available.")
        return
    out["source"] = source
    todo = [p for p in probs if p["id"] not in done]
    print(f"Putnam: {len(probs)}, todo: {len(todo)}", flush=True)

    for i, p in enumerate(todo):
        t0 = time.time()
        text = extract_verifier_claude(p["problem"])
        script, status = parse_script(text)
        row = {"id": p["id"], "status": status, "gold_answer": p["answer"], "elapsed": time.time() - t0}
        if script:
            row["script"] = script[:3000]
            try:
                v = execute_verifier(script, p["answer"], timeout=10.0)
                row["gold_verdict"] = v.verdict
                row["exec_error"] = v.error
            except Exception as e:
                row["gold_verdict"] = None
                row["exec_error"] = str(e)[:200]
        out["results"].append(row)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            print(f"  [{i+1}/{len(todo)}] {p['id']}: {status} gold_v={row.get('gold_verdict','-')} ({row['elapsed']:.1f}s)", flush=True)

    n = len(out["results"])
    uver = sum(1 for r in out["results"] if r["status"] == "unverifiable")
    produced = sum(1 for r in out["results"] if r["status"] == "ok")
    accepts_gold = sum(1 for r in out["results"] if r.get("gold_verdict") is True)
    print("\n" + "=" * 60)
    print(f"Putnam proof-mode via Claude Sonnet 4.6: n={n}")
    print(f"  UNVERIFIABLE (refused):   {uver} ({uver/n:.1%})")
    print(f"  Verifier produced:        {produced} ({produced/n:.1%})")
    print(f"  Accepts gold answer:      {accepts_gold} ({accepts_gold/n:.1%})")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
