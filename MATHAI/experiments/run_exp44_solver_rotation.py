"""Solver rotation: DeepSeek-V3 as a second solver on MATH-175 + AIME + CleanMath.

Motivation: the paper's primary solver (Qwen2.5-7B-Instruct-Turbo) has low
baseline accuracy on CleanMath (12%), so the X-SGRV top tier is necessarily
small (4/4 = 100% at 3.2% coverage with CI [0.40, 1.00]). 4/125 is too narrow
to be a useful headline.

This experiment swaps the solver to DeepSeek-V3 while keeping the X-SGRV
extractor scripts identical (already cached in exp31/exp32/exp34). For each
problem we:

  1. Query DeepSeek-V3 greedy for a candidate answer.
  2. Re-execute the EXISTING cached Llama-3.3-70B extractor script against it.
  3. Same for the cached DeepSeek-V3 extractor script.
  4. Recompute top-tier precision and CI per benchmark.

Budget: ~$0.50 (330 DeepSeek-V3 greedy solver calls). No new extractor calls.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import execute_verifier  # type: ignore
from src.eval.answer_check import answers_equivalent  # type: ignore

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp44_solver_rotation.json"
SOLVER = "deepseek-ai/DeepSeek-V3"

API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY",
)
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=API_KEY, timeout=60.0, max_retries=1)


def _call_timeout(fn, timeout: float = 60.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return None


def _equiv(a, b):
    r = answers_equivalent(a, b)
    return bool(r[0]) if isinstance(r, tuple) else bool(r)


def _extract_boxed(text: str) -> str:
    i = text.rfind("\\boxed{")
    if i < 0:
        return ""
    start = i + len("\\boxed{")
    depth = 1
    j = start
    while j < len(text) and depth > 0:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j].strip()
        j += 1
    return text[start:].strip()


def solve_deepseek(problem: str) -> str:
    def _do():
        resp = client.chat.completions.create(
            model=SOLVER,
            messages=[{"role": "user", "content":
                       f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{problem}"}],
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    text = _call_timeout(_do, timeout=50.0) or ""
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def load_benchmarks() -> dict:
    out: dict = {}
    with open(RESULTS / "exp25_selective_prediction.json") as f:
        exp25_ids = [r["id"] for r in json.load(f)]
    with open(RESULTS / "math_test_sample_500.json") as f:
        math_all = json.load(f)
    math_by_id = {p["id"]: p for p in math_all}
    out["math175"] = [
        {"id": pid, "problem": math_by_id[pid]["problem"], "gold": str(math_by_id[pid]["answer"])}
        for pid in exp25_ids if pid in math_by_id
    ]
    with open(RESULTS / "aime_2025.json") as f:
        aime = json.load(f)
    out["aime"] = [{"id": p["id"], "problem": p["problem"], "gold": str(p["answer"])} for p in aime]
    with open(RESULTS / "cleanmath_combo.json") as f:
        cm = json.load(f)
    out["cleanmath"] = [{"id": p["id"], "problem": p["problem"], "gold": str(p["answer"])} for p in cm]
    return out


def load_cached_scripts(bench: str) -> dict:
    llama_map = {}
    ds_map = {}
    if bench == "math175":
        e31 = json.load(open(RESULTS / "exp31_xsgrv_math175_llama70b.json"))
        e32 = json.load(open(RESULTS / "exp32_math175_deepseek.json"))
        for r in e31["results"]:
            llama_map[r["id"]] = {
                "script": r.get("script"),
                "classification": r.get("classification"),
                "gold_verdict": r.get("gold_verdict"),
                "adv_fp_count": r.get("adv_fp_count"),
            }
        for r in e32["results"]:
            ds_map[r["id"]] = {
                "script": r.get("script"),
                "classification": r.get("classification"),
                "gold_verdict": r.get("gold_verdict"),
                "adv_fp_count": r.get("adv_fp_count"),
            }
    elif bench == "aime":
        try:
            e30 = json.load(open(RESULTS / "exp30_aime_llama70b.json"))
            for r in e30.get("results", []):
                llama_map[r["id"]] = {
                    "script": r.get("script"),
                    "classification": r.get("classification"),
                    "gold_verdict": r.get("gold_verdict"),
                    "adv_fp_count": r.get("adv_fp_count"),
                }
        except Exception:
            pass
        try:
            e32a = json.load(open(RESULTS / "exp32_aime_deepseek.json"))
            for r in e32a["results"]:
                ds_map[r["id"]] = {
                    "script": r.get("script"),
                    "classification": r.get("classification"),
                    "gold_verdict": r.get("gold_verdict"),
                    "adv_fp_count": r.get("adv_fp_count"),
                }
        except Exception:
            pass
    elif bench == "cleanmath":
        e34 = json.load(open(RESULTS / "exp34_cleanmath_llama70b.json"))
        for r in e34["results"]:
            llama_map[r["id"]] = {
                "script": r.get("script"),
                "classification": r.get("classification"),
                "gold_verdict": r.get("gold_verdict"),
                "adv_fp_count": r.get("adv_fp_count"),
            }
    return {"llama": llama_map, "deepseek": ds_map}


def run_verifier(script, candidate):
    if not script:
        return None
    try:
        return execute_verifier(script, candidate, timeout=10.0).verdict
    except Exception:
        return None


def main():
    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {"solver": SOLVER, "rows": []}
    done = {r["id"] for r in out["rows"]}

    benches = load_benchmarks()
    script_cache = {b: load_cached_scripts(b) for b in benches}

    all_probs = []
    for b, probs in benches.items():
        for p in probs:
            all_probs.append({**p, "bench": b})
    todo = [p for p in all_probs if p["id"] not in done]
    print(f"Total: {len(all_probs)}, todo: {len(todo)}")

    for i, p in enumerate(todo):
        pid = p["id"]
        bench = p["bench"]
        t0 = time.time()
        cand = solve_deepseek(p["problem"])
        if not cand:
            out["rows"].append({"id": pid, "bench": bench, "status": "solver_error"})
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2, default=str)
            continue
        correct = _equiv(cand, p["gold"])

        l_entry = script_cache[bench]["llama"].get(pid, {})
        d_entry = script_cache[bench]["deepseek"].get(pid, {})
        l_cand_v = run_verifier(l_entry.get("script"), cand)
        d_cand_v = run_verifier(d_entry.get("script"), cand)

        row = {
            "id": pid,
            "bench": bench,
            "candidate": cand,
            "gold": p["gold"],
            "solver_correct": correct,
            "llama": {
                "classification": l_entry.get("classification"),
                "gold_verdict": l_entry.get("gold_verdict"),
                "candidate_verdict": l_cand_v,
                "adv_fp_count": l_entry.get("adv_fp_count", 0),
                "has_script": bool(l_entry.get("script")),
            },
            "deepseek": {
                "classification": d_entry.get("classification"),
                "gold_verdict": d_entry.get("gold_verdict"),
                "candidate_verdict": d_cand_v,
                "adv_fp_count": d_entry.get("adv_fp_count", 0),
                "has_script": bool(d_entry.get("script")),
            },
            "elapsed": time.time() - t0,
        }
        out["rows"].append(row)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)

        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            print(f"  [{i+1}/{len(todo)}] {bench}/{pid}: cand={cand[:25]} correct={correct} "
                  f"L={l_cand_v} D={d_cand_v} ({row['elapsed']:.1f}s)", flush=True)

    print("\n" + "=" * 70)
    print("Solver rotation summary: DeepSeek-V3 solver + cached X-SGRV")
    print("=" * 70)
    from scipy.stats import binomtest
    for bench in ["math175", "aime", "cleanmath"]:
        rows = [r for r in out["rows"] if r.get("bench") == bench]
        if not rows:
            continue
        n = len(rows)
        solver_acc = sum(1 for r in rows if r.get("solver_correct")) / n
        print(f"\n[{bench}] n={n}, solver accuracy: {solver_acc:.1%}")
        for label, key in [("Llama", "llama"), ("DeepSeek", "deepseek")]:
            tier = [r for r in rows
                    if r[key].get("classification") == "working"
                    and r[key].get("candidate_verdict") is True]
            correct = sum(1 for r in tier if r.get("solver_correct"))
            if tier:
                ci = binomtest(correct, len(tier)).proportion_ci(0.95, "exact")
                print(f"  {label} top-tier: {correct}/{len(tier)} = {correct/len(tier):.3f} "
                      f"[{ci.low:.3f}, {ci.high:.3f}] (cov {len(tier)/n:.1%})")
            else:
                print(f"  {label} top-tier: empty")
        cons = [r for r in rows
                if r["llama"].get("classification") == "working"
                and r["deepseek"].get("classification") == "working"
                and r["llama"].get("candidate_verdict") is True
                and r["deepseek"].get("candidate_verdict") is True]
        cons_c = sum(1 for r in cons if r.get("solver_correct"))
        if cons:
            ci = binomtest(cons_c, len(cons)).proportion_ci(0.95, "exact")
            print(f"  Consensus strict: {cons_c}/{len(cons)} = {cons_c/len(cons):.3f} "
                  f"[{ci.low:.3f}, {ci.high:.3f}] (cov {len(cons)/n:.1%})")
    print("\nSaved ->", OUT)


if __name__ == "__main__":
    main()
