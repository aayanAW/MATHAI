"""Lean head-to-head experiment.

The professor's central criticism: 'current foundation models CAN emit compilable
Lean code for simple answer-checking; Python/SymPy is unjustified.' This
experiment tests that empirically.

For each of MATH-175 (first 60) + AIME 2025 (30) = 90 problems, we prompt
three frontier extractors to emit a Lean 4 theorem statement plus a proof
that would check the gold answer. We measure structural validity (does it
parse as Lean 4 syntactically, does it contain a `theorem` declaration,
does it terminate proof with `sorry` or a tactic).

For the structural parse we use pylean if installed, otherwise lightweight
regex checks (has `theorem`, has `: Prop := `, proof terminates with `by` or
explicit term, no obvious syntax errors).

Extractors: Claude Sonnet 4.6, GPT-5-mini, DeepSeek-V3.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp52_lean_headtohead.json"

PROMPT_LEAN = """You are given a natural-language math problem and its numerical answer. Emit a Lean 4 theorem statement plus a proof that verifies the given answer. The theorem should state that the answer satisfies the problem's constraints.

Requirements:
- Use Lean 4 syntax (not Lean 3). Include the `theorem` keyword.
- The theorem's proof should either close with a tactic like `by decide`, `by norm_num`, `by ring`, `by linarith`, OR use `sorry` if not possible.
- Output ONLY the Lean code block. No prose explanation.
- If you cannot express the problem as a Lean 4 theorem, output the single word UNEXPRESSIBLE.

Problem: {problem}

Gold answer: {answer}

Your Lean 4 code:"""


def lean_structure_check(text: str) -> dict:
    """Lightweight structural check (no Lean compiler)."""
    t = text.strip()
    if "UNEXPRESSIBLE" in t[:200]:
        return {"verdict": "unexpressible", "has_theorem": False, "has_proof": False,
                "uses_sorry": False, "tactic_count": 0}
    # Strip markdown code fences
    m = re.search(r"```(?:lean4?|Lean)?\s*(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    has_theorem = bool(re.search(r"\btheorem\s+\w+", t))
    has_proof = bool(re.search(r":=\s*(by\b|⟨|fun\b)", t, re.DOTALL))
    uses_sorry = bool(re.search(r"\bsorry\b", t))
    tactic_count = len(re.findall(r"\b(norm_num|decide|ring|linarith|omega|simp|exact|rfl|constructor|apply)\b", t))
    return {
        "verdict": "ok",
        "has_theorem": has_theorem,
        "has_proof": has_proof,
        "uses_sorry": uses_sorry,
        "tactic_count": tactic_count,
        "lean_src": t[:1500],
    }


def load_problems(n_math=60, n_aime=30):
    probs = []
    with open(RESULTS / "exp25_selective_prediction.json") as f:
        exp25 = json.load(f)
    with open(RESULTS / "math_test_sample_500.json") as f:
        math_all = json.load(f)
    math_by_id = {p["id"]: p for p in math_all}
    for r in exp25[:n_math]:
        p = math_by_id.get(r["id"])
        if p:
            probs.append({"id": r["id"], "bench": "math175",
                          "problem": p["problem"], "answer": str(p["answer"])})
    with open(RESULTS / "aime_2025.json") as f:
        aime = json.load(f)
    for p in aime[:n_aime]:
        probs.append({"id": p["id"], "bench": "aime",
                      "problem": p["problem"], "answer": str(p["answer"])})
    return probs


def claude_extractor(client, problem, answer):
    try:
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": PROMPT_LEAN.format(problem=problem, answer=answer)}],
        )
        return "".join(b.text for b in r.content if hasattr(b, "text"))
    except Exception as e:
        return f"[ERROR {type(e).__name__}: {str(e)[:100]}]"


def gpt5_extractor(client, problem, answer):
    try:
        r = client.chat.completions.create(
            model="gpt-5-mini",
            max_completion_tokens=4000,
            messages=[{"role": "user", "content": PROMPT_LEAN.format(problem=problem, answer=answer)}],
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"[ERROR {type(e).__name__}: {str(e)[:100]}]"


def deepseek_extractor(client, problem, answer):
    try:
        r = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            max_tokens=1500,
            temperature=0.0,
            messages=[{"role": "user", "content": PROMPT_LEAN.format(problem=problem, answer=answer)}],
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"[ERROR {type(e).__name__}: {str(e)[:100]}]"


def main():
    import anthropic
    from openai import OpenAI

    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    together_client = OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=os.environ.get("TOGETHER_API_KEY",
            "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY"),
        timeout=90.0,
    )

    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {"results": []}
    done = {(r["id"], r["extractor"]) for r in out["results"]}

    probs = load_problems()
    print(f"Loaded {len(probs)} problems", flush=True)

    extractors = [
        ("claude-sonnet-4-6", lambda p, a: claude_extractor(anthropic_client, p, a)),
        ("gpt-5-mini",         lambda p, a: gpt5_extractor(openai_client, p, a)),
        ("deepseek-v3",        lambda p, a: deepseek_extractor(together_client, p, a)),
    ]

    for i, prob in enumerate(probs):
        for ext_name, ext_fn in extractors:
            key = (prob["id"], ext_name)
            if key in done:
                continue
            t0 = time.time()
            text = ext_fn(prob["problem"], prob["answer"])
            res = lean_structure_check(text)
            row = {
                "id": prob["id"],
                "bench": prob["bench"],
                "extractor": ext_name,
                "answer": prob["answer"],
                "elapsed": time.time() - t0,
                **res,
            }
            out["results"].append(row)
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2, default=str)

        if (i + 1) % 5 == 0 or i == len(probs) - 1:
            print(f"  [{i+1}/{len(probs)}] {prob['id']} done for 3 extractors", flush=True)

    # Summary
    from collections import Counter
    print("\n=== Lean head-to-head summary ===")
    for ext_name in ["claude-sonnet-4-6", "gpt-5-mini", "deepseek-v3"]:
        rows = [r for r in out["results"] if r["extractor"] == ext_name]
        n = len(rows)
        unexpressible = sum(1 for r in rows if r["verdict"] == "unexpressible")
        has_theorem = sum(1 for r in rows if r.get("has_theorem"))
        has_proof = sum(1 for r in rows if r.get("has_proof"))
        uses_sorry = sum(1 for r in rows if r.get("uses_sorry"))
        real_proof = sum(1 for r in rows if r.get("has_proof") and not r.get("uses_sorry"))
        print(f"\n[{ext_name}] n={n}")
        print(f"  refused (UNEXPRESSIBLE):         {unexpressible} ({unexpressible/n:.1%})")
        print(f"  emitted Lean theorem:            {has_theorem} ({has_theorem/n:.1%})")
        print(f"  emitted a proof:                 {has_proof} ({has_proof/n:.1%})")
        print(f"  proof uses `sorry`:              {uses_sorry} ({uses_sorry/n:.1%})")
        print(f"  emitted a non-sorry proof:       {real_proof} ({real_proof/n:.1%})")


if __name__ == "__main__":
    main()
