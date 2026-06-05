"""p(True) / Kadavath baseline.

For each problem, take the plurality-vote answer from the cached 10 samples,
ask Qwen2.5-7B-Instruct-Turbo "Is this answer correct? True/False" with logprobs,
and use the True-token probability as a continuous confidence score.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval.answer_check import answers_equivalent  # type: ignore
from openai import OpenAI

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp36_ptrue.json"
SOLVER = "Qwen/Qwen2.5-7B-Instruct-Turbo"

API_KEY = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")

client = OpenAI(base_url="https://api.together.xyz/v1", api_key=API_KEY, timeout=60.0)


def _equiv(a: str, b: str) -> bool:
    r = answers_equivalent(a, b)
    if isinstance(r, tuple):
        return bool(r[0])
    return bool(r)


def plurality_answer(samples: list[dict]) -> str:
    vote = Counter(s.get("answer", "") for s in samples if s.get("answer"))
    if not vote:
        return ""
    return vote.most_common(1)[0][0]


def load_problems(bench: str) -> list[dict]:
    if bench == "math175":
        with open(RESULTS / "exp25_selective_prediction.json") as f:
            exp25 = json.load(f)
        with open(RESULTS / "math_test_sample_500.json") as f:
            math_all = json.load(f)
        by_id = {p["id"]: p for p in math_all}
        return [{"id": r["id"], "problem": by_id[r["id"]]["problem"], "gold": str(by_id[r["id"]]["answer"])}
                for r in exp25 if r["id"] in by_id]
    if bench == "aime":
        with open(RESULTS / "aime_2025.json") as f:
            aime = json.load(f)
        return [{"id": p["id"], "problem": p["problem"], "gold": str(p["answer"])} for p in aime]
    if bench == "cleanmath":
        with open(RESULTS / "cleanmath_combo.json") as f:
            cm = json.load(f)
        return [{"id": p["id"], "problem": p["problem"], "gold": str(p["answer"])} for p in cm]
    raise ValueError(bench)


PROMPT_TEMPLATE = """Consider the following math problem and a proposed answer.

Problem: {problem}

Proposed answer: {answer}

Is this proposed answer correct? Reply with exactly one word: either \"True\" or \"False\"."""


def ptrue_call(problem: str, answer: str, max_retries: int = 2) -> tuple[float | None, str | None]:
    """Return (p_true, raw_text). p_true is softmax prob of True over {True, False}."""
    prompt = PROMPT_TEMPLATE.format(problem=problem, answer=answer)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=SOLVER,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
            )
            choice = resp.choices[0]
            raw = (choice.message.content or "").strip()
            # Find per-token logprobs on the first token.
            lp_obj = choice.logprobs
            if lp_obj is None or not getattr(lp_obj, "content", None):
                # Fallback: judge by raw text
                if raw.lower().startswith("true"):
                    return 0.95, raw
                elif raw.lower().startswith("false"):
                    return 0.05, raw
                return 0.5, raw
            first_tok = lp_obj.content[0]
            top = getattr(first_tok, "top_logprobs", []) or []
            tb = {}
            import math
            for entry in top:
                t = entry.token.lower().strip()
                tb[t] = math.exp(entry.logprob)
            # Also include the chosen first token
            chosen_tok = first_tok.token.lower().strip()
            chosen_lp = first_tok.logprob
            tb.setdefault(chosen_tok, math.exp(chosen_lp))
            p_true_raw = tb.get("true", 0.0) + tb.get("tr", 0.0)
            p_false_raw = tb.get("false", 0.0) + tb.get("fa", 0.0) + tb.get("fals", 0.0)
            if p_true_raw + p_false_raw < 1e-6:
                if raw.lower().startswith("true"):
                    return 0.95, raw
                if raw.lower().startswith("false"):
                    return 0.05, raw
                return 0.5, raw
            return p_true_raw / (p_true_raw + p_false_raw), raw
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"  retry exhausted: {type(e).__name__}: {e}")
            return None, None


def main():
    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {"solver": SOLVER, "rows": []}
    done_ids = {r["id"] for r in out["rows"]}

    for bench in ["math175", "aime", "cleanmath"]:
        print(f"\n>>> Phase p(True) [{bench}]")
        probs = load_problems(bench)
        samples_path = RESULTS / f"se_samples_{bench}.json"
        cache = json.load(open(samples_path))
        for i, p in enumerate(probs):
            pid = p["id"]
            c = cache.get(pid)
            if not c:
                continue
            samples = c.get("samples", [])
            plur = plurality_answer(samples)
            if (pid, bench) in {(r["id"], r["bench"]) for r in out["rows"]}:
                continue
            correct = _equiv(plur, c.get("gold", p["gold"]))
            p_true, raw = ptrue_call(p["problem"], plur)
            row = {
                "id": pid,
                "bench": bench,
                "plurality_answer": plur,
                "gold": c.get("gold", p["gold"]),
                "plurality_correct": correct,
                "p_true": p_true,
                "raw_reply": raw,
            }
            out["rows"].append(row)
            if (i + 1) % 10 == 0 or i == len(probs) - 1:
                with open(OUT, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"  [{bench}] {i+1}/{len(probs)}  p_true={p_true}  plur_correct={correct}", flush=True)

    # Final summary: AUROC of 1-p_true vs plurality_correct wrong
    print("\n" + "=" * 50)
    print("p(True) Summary")
    print("=" * 50)
    for bench in ["math175", "aime", "cleanmath"]:
        rows = [r for r in out["rows"] if r["bench"] == bench and r.get("p_true") is not None]
        if not rows:
            continue
        y = [not r["plurality_correct"] for r in rows]
        s = [1 - r["p_true"] for r in rows]  # higher = more likely wrong
        try:
            auroc = roc_auc_score(y, s)
        except Exception:
            auroc = None
        print(f"  [{bench}] n={len(rows)}  AUROC(wrong vs 1-p_true)={auroc}")

    summary = {}
    for bench in ["math175", "aime", "cleanmath"]:
        rows = [r for r in out["rows"] if r["bench"] == bench and r.get("p_true") is not None]
        if not rows:
            continue
        y = [not r["plurality_correct"] for r in rows]
        s = [1 - r["p_true"] for r in rows]
        try:
            auroc = roc_auc_score(y, s)
        except Exception:
            auroc = None
        summary[bench] = {"n": len(rows), "auroc_wrong_vs_1mptrue": auroc}
    out["summary"] = summary
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
