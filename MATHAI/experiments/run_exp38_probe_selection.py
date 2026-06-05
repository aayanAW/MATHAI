"""Held-out probe selection for the deployment-time adversarial filter.

Addresses reviewer concern A7: the probe set {candidate +/- 1, +7, x2, 42} was
chosen ad hoc. This script:

1. Draws a 30-problem dev split from MATH-500 (stratified, disjoint from the 175).
2. Runs Llama-3.3-70B X-SGRV extraction on the dev problems.
3. Evaluates six candidate probe sets on the dev split.
4. Picks the set that maximizes {precision gain} minus {coverage loss}.
5. Saves the selection + dev split for reproducibility.

The selected probe set is then frozen for test-set evaluation.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier  # type: ignore

EXTRACTOR = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp38_probe_selection.json"

API_KEY = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=API_KEY, timeout=120.0, max_retries=2)


def make_extractor():
    def _call(prompt):
        resp = client.chat.completions.create(
            model=EXTRACTOR,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content
    return _call


def make_probes(candidate: str, probe_set_name: str) -> list[str]:
    """Return a list of probe values to test against the candidate."""
    try:
        c = int(str(candidate).strip())
    except Exception:
        # Non-integer candidate: fall back to a small fixed set.
        return ["0", "1", "100", "-1"]
    table = {
        "pm1": [c - 1, c + 1],
        "pm1_p7": [c - 1, c + 1, c + 7],
        "pm1_p7_x2": [c - 1, c + 1, c + 7, c * 2 if c != 0 else 1],
        "pm1_p7_x2_42": [c - 1, c + 1, c + 7, c * 2 if c != 0 else 1, 42 if c != 42 else 41],
        "x2_p7": [c * 2 if c != 0 else 1, c + 7],
        "random_pmk": None,  # handled below
    }
    if probe_set_name == "random_pmk":
        rng = np.random.default_rng(hash(str(candidate)) % (2**32))
        ks = rng.integers(1, 11, size=5)
        out = []
        for k in ks:
            out.extend([c + k, c - k])
        return [str(x) for x in out if x != c]
    return [str(x) for x in table[probe_set_name] if str(x) != str(candidate)]


def load_math500() -> list[dict]:
    with open(RESULTS / "math_test_sample_500.json") as f:
        all_math = json.load(f)
    return all_math


def load_exp25_ids() -> set[str]:
    with open(RESULTS / "exp25_selective_prediction.json") as f:
        return {r["id"] for r in json.load(f)}


def sample_dev_split(n: int = 30, seed: int = 0) -> list[dict]:
    """Draw a stratified 30-problem dev split, disjoint from exp25."""
    math_all = load_math500()
    used = load_exp25_ids()
    pool = [p for p in math_all if p["id"] not in used]
    # Stratify by level if available.
    by_level: dict[int, list] = {}
    for p in pool:
        lvl = int(str(p.get("level", 3)).replace("Level ", ""))
        by_level.setdefault(lvl, []).append(p)
    rng = random.Random(seed)
    dev = []
    per_level = max(1, n // max(1, len(by_level)))
    for lvl in sorted(by_level):
        rng.shuffle(by_level[lvl])
        dev.extend(by_level[lvl][:per_level])
    rng.shuffle(dev)
    return dev[:n]


def solve_with_qwen(problem: str) -> str:
    """Greedy Qwen-7B-Instruct-Turbo candidate answer."""
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        messages=[{"role": "user", "content":
            f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{problem}"}],
        max_tokens=1024,
        temperature=0.0,
    )
    text = resp.choices[0].message.content or ""
    # Extract \boxed{...}
    import re
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    # Fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def eval_probe_set(rows: list[dict], probe_name: str) -> dict:
    """Given rows {id, script, gold, candidate, solver_correct, gold_verdict, candidate_verdict},
    apply a probe set to each working verifier and report accept/reject."""
    n_total = len(rows)
    n_top_tier = 0  # after filter
    n_top_correct = 0
    n_raw_top = 0
    n_raw_correct = 0
    for r in rows:
        if r.get("script") is None or r.get("candidate_verdict") is not True:
            continue
        n_raw_top += 1
        if r.get("solver_correct"):
            n_raw_correct += 1
        probes = make_probes(r["candidate"], probe_name)
        any_accept = False
        for p in probes:
            try:
                v = execute_verifier(r["script"], p, timeout=5.0)
                if v.verdict is True:
                    any_accept = True
                    break
            except Exception:
                continue
        if not any_accept:
            n_top_tier += 1
            if r.get("solver_correct"):
                n_top_correct += 1
    raw_prec = n_raw_correct / n_raw_top if n_raw_top else 0
    filt_prec = n_top_correct / n_top_tier if n_top_tier else 0
    filt_cov = n_top_tier / n_total if n_total else 0
    return {
        "raw_top_tier": n_raw_top,
        "raw_precision": raw_prec,
        "filter_top_tier": n_top_tier,
        "filter_coverage": filt_cov,
        "filter_precision": filt_prec,
        "precision_gain": filt_prec - raw_prec,
        "coverage_loss": (n_raw_top - n_top_tier) / n_total if n_total else 0,
        "objective": (filt_prec - raw_prec) - ((n_raw_top - n_top_tier) / n_total if n_total else 0),
    }


def process_dev_problem(prob: dict, extractor_fn) -> dict | None:
    """Solve + extract + verify for a single dev problem."""
    pid = prob["id"]
    try:
        candidate = solve_with_qwen(prob["problem"])
    except Exception as e:
        print(f"  [{pid}] solver err {type(e).__name__}: {e}")
        return None
    try:
        ext = extract_verifier(prob["problem"], extractor_fn)
    except Exception as e:
        print(f"  [{pid}] extractor err {type(e).__name__}: {e}")
        return None
    if ext.unverifiable or ext.error or ext.script is None:
        return {
            "id": pid,
            "problem": prob["problem"],
            "gold": str(prob["answer"]),
            "candidate": candidate,
            "script": None,
            "candidate_verdict": None,
            "gold_verdict": None,
            "solver_correct": False,
        }
    # Evaluate gold, candidate
    gold_ver = execute_verifier(ext.script, str(prob["answer"]), timeout=10.0)
    cand_ver = execute_verifier(ext.script, candidate, timeout=10.0)
    # Crude answer-equivalence check
    from src.eval.answer_check import answers_equivalent  # type: ignore
    rr = answers_equivalent(candidate, str(prob["answer"]))
    correct = bool(rr[0]) if isinstance(rr, tuple) else bool(rr)
    return {
        "id": pid,
        "problem": prob["problem"],
        "gold": str(prob["answer"]),
        "candidate": candidate,
        "script": ext.script,
        "candidate_verdict": cand_ver.verdict,
        "gold_verdict": gold_ver.verdict,
        "solver_correct": correct,
    }


def main():
    dev = sample_dev_split(n=30, seed=0)
    print(f"Dev split: n={len(dev)}  ids: {[p['id'] for p in dev]}")

    # Load or resume
    if OUT.exists():
        out = json.load(open(OUT))
        rows = out.get("rows", [])
        done_ids = {r["id"] for r in rows}
    else:
        rows = []
        done_ids = set()
        out = {"extractor": EXTRACTOR, "dev_ids": [p["id"] for p in dev], "rows": rows}

    extractor_fn = make_extractor()
    t0 = time.time()
    for i, prob in enumerate(dev):
        if prob["id"] in done_ids:
            continue
        r = process_dev_problem(prob, extractor_fn)
        if r is None:
            continue
        rows.append(r)
        out["rows"] = rows
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)
        dt = time.time() - t0
        print(f"  [{i+1}/{len(dev)}] {prob['id']}: cand={r['candidate']}, gold={r['gold']}, "
              f"correct={r['solver_correct']}, gold_ver={r['gold_verdict']}, "
              f"cand_ver={r['candidate_verdict']}  ({dt:.0f}s elapsed)", flush=True)

    # Evaluate probe sets
    print("\n" + "=" * 60)
    print("Probe set evaluation on dev")
    print("=" * 60)
    results = {}
    for pset in ["pm1", "pm1_p7", "pm1_p7_x2", "pm1_p7_x2_42", "x2_p7", "random_pmk"]:
        r = eval_probe_set(rows, pset)
        results[pset] = r
        print(f"  {pset:16s} raw={r['raw_precision']:.3f}  filt_prec={r['filter_precision']:.3f}  "
              f"filt_cov={r['filter_coverage']:.3f}  obj={r['objective']:.3f}")

    best = max(results.items(), key=lambda kv: kv[1]["objective"])
    print(f"\nSelected probe set: {best[0]}  (objective={best[1]['objective']:.3f})")

    out["probe_results"] = results
    out["selected_probe_set"] = best[0]
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
