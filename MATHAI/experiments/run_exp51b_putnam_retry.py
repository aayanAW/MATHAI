"""Retry Putnam no_script rows with higher token budget + save raw text."""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import execute_verifier  # type: ignore

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
IN = RESULTS / "exp51_putnam_gpt5.json"
OUT = RESULTS / "exp51_putnam_gpt5.json"  # in-place update
MODEL = "gpt-5-mini"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT = """You are reading a Putnam competition problem.

Produce a Python function `verify(answer) -> bool` that returns True iff the candidate answer satisfies the problem's constraints. Use SymPy and `import random; random.seed(42)` for random-input tests.

If the problem needs a real proof beyond computational checking, output UNVERIFIABLE as the entire response.

OUTPUT FORMAT: either a raw Python `def verify(answer):` at the top of your response, or the word UNVERIFIABLE alone.

Problem:
{problem}

Your response:"""


def parse_script(text):
    if not text or not text.strip():
        return None, "empty_response"
    if "UNVERIFIABLE" in text[:300]:
        return None, "unverifiable"
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip(), "ok"
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip(), "ok"
    if "def verify" in text:
        # Extract from 'def verify' onwards, trimming trailing prose
        start = text.index("def verify")
        body = text[start:]
        return body.strip(), "ok"
    return None, "no_script"


def call_gpt5(problem_text, max_tokens=6000):
    """Higher token budget to accommodate GPT-5-mini's reasoning."""
    for attempt in range(2):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": PROMPT.format(problem=problem_text)}],
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            if attempt < 1:
                time.sleep(2)
                continue
            return ""
    return ""


def main():
    from datasets import load_dataset
    ds = load_dataset("amitayusht/PutnamBench", split="train")
    by_name = {ex["name"]: ex for ex in ds}

    d = json.load(open(IN))
    results = d["results"]
    retry_ids = [r["id"] for r in results if r.get("status") in ("no_script", "empty_response")]
    print(f"Rows to retry: {len(retry_ids)}", flush=True)

    by_id = {r["id"]: r for r in results}
    for i, pid in enumerate(retry_ids):
        ex = by_name.get(pid)
        if not ex:
            continue
        t0 = time.time()
        text = call_gpt5(ex["informal_statement"])
        script, status = parse_script(text)
        r = by_id[pid]
        r["status"] = status
        r["raw_text_head"] = text[:300]
        r["elapsed_retry"] = time.time() - t0
        if script:
            r["script"] = script[:3000]
            try:
                v = execute_verifier(script, r.get("gold_answer", ""), timeout=10.0)
                r["gold_verdict"] = v.verdict
            except Exception as e:
                r["gold_verdict"] = None
                r["exec_error"] = str(e)[:200]

        with open(OUT, "w") as f:
            json.dump(d, f, indent=2, default=str)

        if (i + 1) % 10 == 0 or i == len(retry_ids) - 1:
            print(f"  [{i+1}/{len(retry_ids)}] {pid}: {status} ({time.time() - t0:.1f}s)", flush=True)

    # Summary
    from collections import Counter
    s = Counter(r.get("status", "?") for r in results)
    print("\n== Final status distribution ==")
    for k, v in sorted(s.items()):
        print(f"  {k}: {v}")
    ok = sum(1 for r in results if r.get("status") == "ok")
    gold_true = sum(1 for r in results if r.get("gold_verdict") is True)
    print(f"  -> verifier produced: {ok}/{len(results)}")
    print(f"  -> accepts gold:      {gold_true}/{len(results)}")


if __name__ == "__main__":
    main()
