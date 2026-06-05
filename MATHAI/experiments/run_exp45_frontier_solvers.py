"""Frontier solver rotation: DeepSeek-R1, DeepSeek-V3.1, gpt-oss-120b as solvers.

For each problem, query each frontier solver for a greedy candidate, then
re-execute the EXISTING cached Llama-3.3-70B and DeepSeek-V3 X-SGRV extractor
scripts against the new candidate. Reports top-tier precision per (solver,
extractor, benchmark) cell.

Target models (Together serverless, probed available 2026-04-21):
  - deepseek-ai/DeepSeek-R1 (reasoning frontier)
  - deepseek-ai/DeepSeek-V3.1
  - openai/gpt-oss-120b (OpenAI's open-weights 120B)

Benchmarks: MATH-175 + AIME 2025 + CleanMath combo (330 problems).

Budget: ~$20-40. Time: ~2 hours wall with rate-limit padding.
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
OUT = RESULTS / "exp45_frontier_solvers.json"

SOLVERS = [
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3.1",
    "openai/gpt-oss-120b",
]

API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY",
)
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=API_KEY, timeout=120.0, max_retries=1)


def _call_timeout(fn, timeout=110.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return None


def _equiv(a, b):
    r = answers_equivalent(a, b)
    return bool(r[0]) if isinstance(r, tuple) else bool(r)


def _extract_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return ""
    start = i + len("\\boxed{")
    depth, j = 1, start
    while j < len(text) and depth > 0:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j].strip()
        j += 1
    return text[start:].strip()


def solve_with(solver_id, problem):
    def _do():
        kwargs = {
            "model": solver_id,
            "messages": [{"role": "user", "content":
                          f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{problem}"}],
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        # DeepSeek-R1 is a reasoning model; allow longer outputs
        if "R1" in solver_id:
            kwargs["max_tokens"] = 8192
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    text = _call_timeout(_do, timeout=150.0) or ""
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    # Fallback: last numeric in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def load_benchmarks():
    out = {}
    with open(RESULTS / "exp25_selective_prediction.json") as f:
        exp25_ids = [r["id"] for r in json.load(f)]
    with open(RESULTS / "math_test_sample_500.json") as f:
        math_all = json.load(f)
    math_by_id = {p["id"]: p for p in math_all}
    out["math175"] = [{"id": pid, "problem": math_by_id[pid]["problem"], "gold": str(math_by_id[pid]["answer"])}
                      for pid in exp25_ids if pid in math_by_id]
    with open(RESULTS / "aime_2025.json") as f:
        out["aime"] = [{"id": p["id"], "problem": p["problem"], "gold": str(p["answer"])} for p in json.load(f)]
    with open(RESULTS / "cleanmath_combo.json") as f:
        out["cleanmath"] = [{"id": p["id"], "problem": p["problem"], "gold": str(p["answer"])} for p in json.load(f)]
    return out


def load_cached_scripts(bench):
    llama_map, ds_map = {}, {}
    if bench == "math175":
        e31 = json.load(open(RESULTS / "exp31_xsgrv_math175_llama70b.json"))
        e32 = json.load(open(RESULTS / "exp32_math175_deepseek.json"))
        for r in e31["results"]:
            llama_map[r["id"]] = r.get("script")
        for r in e32["results"]:
            ds_map[r["id"]] = r.get("script")
    elif bench == "aime":
        try:
            e30 = json.load(open(RESULTS / "exp30_aime_llama70b.json"))
            for r in e30.get("results", []):
                llama_map[r["id"]] = r.get("script")
        except Exception:
            pass
        try:
            e32a = json.load(open(RESULTS / "exp32_aime_deepseek.json"))
            for r in e32a["results"]:
                ds_map[r["id"]] = r.get("script")
        except Exception:
            pass
    elif bench == "cleanmath":
        e34 = json.load(open(RESULTS / "exp34_cleanmath_llama70b.json"))
        for r in e34["results"]:
            llama_map[r["id"]] = r.get("script")
    return llama_map, ds_map


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
        out = {"solvers": SOLVERS, "rows": []}
    done = {(r["solver"], r["id"]) for r in out["rows"]}

    benches = load_benchmarks()
    script_cache = {b: load_cached_scripts(b) for b in benches}

    todo = []
    for solver_id in SOLVERS:
        for b, probs in benches.items():
            for p in probs:
                if (solver_id, p["id"]) in done:
                    continue
                todo.append({**p, "bench": b, "solver": solver_id})
    print(f"Total cells: {len(SOLVERS) * 330}, todo: {len(todo)}", flush=True)

    for i, p in enumerate(todo):
        pid = p["id"]
        bench = p["bench"]
        solver_id = p["solver"]
        t0 = time.time()
        cand = solve_with(solver_id, p["problem"])
        correct = _equiv(cand, p["gold"]) if cand else False

        llama_map, ds_map = script_cache[bench]
        l_script = llama_map.get(pid)
        d_script = ds_map.get(pid)
        l_v = run_verifier(l_script, cand) if cand else None
        d_v = run_verifier(d_script, cand) if cand else None

        row = {
            "solver": solver_id,
            "id": pid,
            "bench": bench,
            "candidate": cand,
            "gold": p["gold"],
            "solver_correct": correct,
            "llama_verdict": l_v,
            "deepseek_verdict": d_v,
            "elapsed": time.time() - t0,
        }
        out["rows"].append(row)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)

        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            short = solver_id.split("/")[-1][:25]
            print(f"  [{i+1}/{len(todo)}] {short}/{bench}/{pid}: correct={correct} L={l_v} D={d_v} ({row['elapsed']:.1f}s)", flush=True)

    # Summary
    from scipy.stats import binomtest
    print("\n" + "=" * 70)
    print("Frontier solver rotation summary")
    print("=" * 70)
    for solver_id in SOLVERS:
        for bench in ["math175", "aime", "cleanmath"]:
            rows = [r for r in out["rows"] if r["solver"] == solver_id and r["bench"] == bench]
            if not rows:
                continue
            n = len(rows)
            sa = sum(1 for r in rows if r["solver_correct"]) / n
            print(f"\n[{solver_id.split('/')[-1]} / {bench}] n={n} solver acc={sa:.1%}")
            for label, k in [("Llama", "llama_verdict"), ("DeepSeek", "deepseek_verdict")]:
                tier = [r for r in rows if r[k] is True]
                correct = sum(1 for r in tier if r["solver_correct"])
                if tier:
                    ci = binomtest(correct, len(tier)).proportion_ci(0.95, "exact")
                    print(f"  {label} ext: {correct}/{len(tier)} = {correct/len(tier):.3f} [{ci.low:.3f},{ci.high:.3f}] cov {len(tier)/n:.1%}")

    out.setdefault("summary_computed", True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
