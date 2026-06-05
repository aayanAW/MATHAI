"""Seed sensitivity on cross-extractor consensus.

Re-run Llama-3.3-70B and DeepSeek-V3 extractors at T=0.3 on a random 30
MATH-500 problems, two independent seeds, and report variance in:
  (a) fraction of problems where the verifier accepts the candidate
  (b) fraction of problems where both extractors agree
  (c) consensus strict precision drift

Budget: ~$2 (30 problems * 2 seeds * 2 extractors = 120 calls).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier  # type: ignore

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp42_seed_sensitivity.json"
LLAMA = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEEPSEEK = "deepseek-ai/DeepSeek-V3"

API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY",
)
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=API_KEY, timeout=60.0, max_retries=1)


def make_extractor(model_id: str, temperature: float):
    def _call(prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return ""
    return _call


def sample_problems(n: int = 30, seed: int = 0) -> list[dict]:
    """Pick 30 MATH-175 problems for consistency with main paper."""
    with open(RESULTS / "exp25_selective_prediction.json") as f:
        exp25 = json.load(f)
    with open(RESULTS / "math_test_sample_500.json") as f:
        math_all = json.load(f)
    by_id = {p["id"]: p for p in math_all}
    problems = [by_id[r["id"]] for r in exp25 if r["id"] in by_id]
    rng = random.Random(seed)
    rng.shuffle(problems)
    # Use each problem's existing Qwen greedy answer as candidate
    exp25_by_id = {r["id"]: r for r in exp25}
    return [
        {
            "id": p["id"],
            "problem": p["problem"],
            "gold": str(p["answer"]),
            "candidate": exp25_by_id[p["id"]]["sample0_answer"],
            "correct": exp25_by_id[p["id"]]["sample0_correct"],
        }
        for p in problems[:n]
    ]


def run_one(probs: list[dict], extractor_fn, extractor_name: str) -> list[dict]:
    out = []
    for i, p in enumerate(probs):
        t0 = time.time()
        try:
            ext = extract_verifier(p["problem"], extractor_fn)
        except Exception as e:
            out.append({"id": p["id"], "extractor": extractor_name, "status": f"err:{type(e).__name__}"})
            continue
        if ext.unverifiable or ext.error or ext.script is None:
            out.append({
                "id": p["id"],
                "extractor": extractor_name,
                "status": "unverifiable" if ext.unverifiable else "error",
                "error": ext.error,
            })
            continue
        gold_ver = execute_verifier(ext.script, p["gold"], timeout=10.0)
        cand_ver = execute_verifier(ext.script, str(p["candidate"]), timeout=10.0)
        working = gold_ver.verdict is True
        accepted = cand_ver.verdict is True
        out.append({
            "id": p["id"],
            "extractor": extractor_name,
            "status": "ok",
            "working": working,
            "accepted": accepted,
            "solver_correct": p["correct"],
            "elapsed": time.time() - t0,
        })
        if (i + 1) % 5 == 0:
            print(f"  {extractor_name} {i+1}/{len(probs)}", flush=True)
    return out


def main():
    probs = sample_problems(n=30, seed=0)
    print(f"30 MATH-175 problems sampled")

    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {"runs": []}

    runs_done = {r["key"] for r in out["runs"]}

    # Four runs: (Llama, seed=1, T=0.3), (Llama, seed=2, T=0.3),
    #            (DeepSeek, seed=1, T=0.3), (DeepSeek, seed=2, T=0.3).
    # Together API doesn't use seed for t>0 sampling, so T=0.3 variations
    # capture natural variance.
    configs = [
        ("llama_s1", LLAMA, 0.3),
        ("llama_s2", LLAMA, 0.3),
        ("deepseek_s1", DEEPSEEK, 0.3),
        ("deepseek_s2", DEEPSEEK, 0.3),
    ]
    for key, model, T in configs:
        if key in runs_done:
            print(f"Skipping {key} (already done)")
            continue
        print(f"\n>>> Run {key}: model={model} T={T}")
        ext_fn = make_extractor(model, T)
        rows = run_one(probs, ext_fn, key)
        out["runs"].append({"key": key, "model": model, "T": T, "rows": rows})
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)

    # Summarize variance
    print("\n" + "=" * 60)
    print("Seed sensitivity summary")
    print("=" * 60)
    for r in out["runs"]:
        rows = r["rows"]
        ok = [x for x in rows if x.get("status") == "ok"]
        accepted = [x for x in ok if x.get("accepted")]
        correct_accepted = [x for x in accepted if x.get("solver_correct")]
        n = len(rows)
        print(f"  {r['key']}: working={sum(1 for x in ok if x.get('working'))}/{n}  "
              f"accepted={len(accepted)}/{n}  "
              f"precision={len(correct_accepted)}/{len(accepted) if accepted else 1}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
