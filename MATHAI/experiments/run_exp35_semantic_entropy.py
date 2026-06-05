"""Experiment 35: Semantic Entropy baselines (Kuhn 2023 / Farquhar Nature 2024).

Runs two variants:
  (a) math   — cluster samples via SymPy symbolic equivalence (fast, math-native)
  (b) nli    — cluster via DeBERTa-large-mnli bidirectional entailment (Kuhn 2023)

On three benchmarks:
  - MATH-500 n=175 (exp25 stratified)
  - AIME 2025 n=30
  - CleanMath combo n=125 (HMMT Feb 2025 + BRUMO 2025 + SMT 2025 + APEX 2025)

Three phases:
  Phase 1 — Sample 10 generations per problem from Qwen2.5-7B-Instruct-Turbo at T=0.7
            Cached to: results/se_samples_{bench}.json
  Phase 2 — Compute entropy under each variant. Saved to: results/exp35_semantic_entropy.json
  Phase 3 — AUROC for "entropy as predictor of INcorrectness" per benchmark × variant.

Budget: ~$1 (3300 samples × 1k tokens × $0.20/M input + output)
Time:   ~40 min solver sampling + ~20 min NLI clustering

Flags:
  --skip-nli   skip NLI variant (math only, faster)
  --skip-math  skip math variant (NLI only)
  --bench {math175,aime,cleanmath,all}  restrict to one benchmark
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.baselines.semantic_entropy import Sample, SemanticEntropyScorer

SOLVER_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_PATH = RESULTS_DIR / "exp35_semantic_entropy.json"
N_SAMPLES = 10
TEMP = 0.7

api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=api_key,
    timeout=60.0,
    max_retries=1,
)

SOLVER_PROMPT = "Solve the following math problem step by step. At the end, write your final answer inside \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"


def extract_boxed(text: str) -> str:
    """Extract content of the last \\boxed{...} (nested-brace aware)."""
    matches = []
    i = 0
    while True:
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 1
        j = idx + len("\\boxed{")
        start = j
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            matches.append(text[start: j - 1].strip())
        i = j
    return matches[-1] if matches else ""


def sample_one(problem: str) -> tuple[str, str]:
    resp = client.chat.completions.create(
        model=SOLVER_MODEL,
        messages=[{"role": "user", "content": SOLVER_PROMPT.format(problem=problem)}],
        max_tokens=1536,
        temperature=TEMP,
    )
    raw = resp.choices[0].message.content or ""
    return raw, extract_boxed(raw)


_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def sample_one_safe(problem: str, timeout: float = 90.0) -> tuple[str, str]:
    """sample_one with a hard thread-level timeout to catch OS-level TCP hangs."""
    future = _THREAD_POOL.submit(sample_one, problem)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"API call hung for >{timeout}s — skipping")


def load_math175() -> list[dict]:
    with open(RESULTS_DIR / "exp25_selective_prediction.json") as f:
        exp25 = json.load(f)
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        math_all = json.load(f)
    math_by_id = {p["id"]: p for p in math_all}
    probs = []
    for r in exp25:
        p = math_by_id.get(r["id"])
        if p:
            probs.append({"id": p["id"], "problem": p["problem"], "answer": str(p["answer"])})
    return probs


def load_aime() -> list[dict]:
    with open(RESULTS_DIR / "aime_2025.json") as f:
        aime = json.load(f)
    return [{"id": p["id"], "problem": p["problem"], "answer": str(p["answer"])} for p in aime]


def load_cleanmath() -> list[dict]:
    path = RESULTS_DIR / "cleanmath_combo.json"
    if not path.exists():
        print(f"  [cleanmath] SKIP: {path.name} not found")
        return []
    with open(path) as f:
        cm = json.load(f)
    return [{"id": p["id"], "problem": p["problem"], "answer": str(p["answer"])} for p in cm]


BENCHES = {
    "math175": load_math175,
    "aime": load_aime,
    "cleanmath": load_cleanmath,
}


def phase1_sample(bench: str, problems: list[dict]) -> dict[str, dict]:
    """Cache 10 solver samples per problem."""
    cache_path = RESULTS_DIR / f"se_samples_{bench}.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cache = json.load(open(cache_path))
            print(f"  [{bench}] resumed {len(cache)} cached problems")
        except Exception:
            pass

    for prob in problems:
        existing = cache.get(prob["id"], {})
        samples = existing.get("samples", [])
        needed = N_SAMPLES - len(samples)
        if needed <= 0:
            continue
        for k in range(needed):
            try:
                raw, boxed = sample_one_safe(prob["problem"], timeout=90.0)
                samples.append({"raw": raw, "answer": boxed})
            except Exception as e:
                print(f"    [{bench}] {prob['id']} sample {k+1}/{needed}: error {type(e).__name__}: {e}")
                samples.append({"raw": None, "answer": "", "error": f"{type(e).__name__}: {e}"})

        cache[prob["id"]] = {"id": prob["id"], "samples": samples, "gold": prob["answer"]}
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2, default=str)
        boxed_vals = [s["answer"] for s in samples]
        print(f"    [{bench}] {prob['id']}: sampled {len(samples)} — uniq {len(set(boxed_vals))}", flush=True)
    return cache


def phase2_score(bench: str, problems: list[dict], cache: dict[str, dict],
                 do_math: bool, do_nli: bool) -> list[dict]:
    math_scorer = SemanticEntropyScorer.from_math() if do_math else None
    nli_scorer = None
    if do_nli:
        try:
            print(f"  [{bench}] loading DeBERTa-MNLI…")
            nli_scorer = SemanticEntropyScorer.from_nli()
            print(f"  [{bench}] NLI scorer ready")
        except Exception as e:
            print(f"  [{bench}] NLI scorer failed to load ({type(e).__name__}: {e}); skipping NLI variant")
            nli_scorer = None

    rows = []
    for prob in problems:
        c = cache.get(prob["id"])
        if not c:
            continue
        samples = [Sample(raw=s.get("raw") or "", answer=s.get("answer") or "") for s in c["samples"]]
        # Correctness: any sample whose boxed answer matches gold counts; we also
        # mark the plurality-vote correctness as the main label.
        from collections import Counter
        vote = Counter(s.answer for s in samples if s.answer)
        plurality = vote.most_common(1)[0][0] if vote else ""
        plurality_correct = _equiv(plurality, c["gold"])
        any_correct = any(_equiv(s.answer, c["gold"]) for s in samples)

        row: dict = {
            "id": prob["id"],
            "bench": bench,
            "gold": c["gold"],
            "plurality": plurality,
            "plurality_correct": plurality_correct,
            "any_correct": any_correct,
            "n_samples": len(samples),
        }
        if math_scorer:
            row["se_math"] = math_scorer.score(prob["problem"], samples)
        if nli_scorer:
            row["se_nli"] = nli_scorer.score(prob["problem"], samples)
        rows.append(row)
        tag = []
        if "se_math" in row: tag.append(f"math={row['se_math']['entropy']:.2f}")
        if "se_nli" in row:  tag.append(f"nli={row['se_nli']['entropy']:.2f}")
        print(f"    [{bench}] {prob['id']}: {'/'.join(tag) if tag else 'no-score'}  "
              f"plur_correct={plurality_correct}", flush=True)
    return rows


def _equiv(a: str, b: str) -> bool:
    """answers_equivalent returns (bool, reason) — unpack carefully."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.eval.answer_check import answers_equivalent
        result = answers_equivalent(str(a), str(b))
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)
    except Exception:
        return str(a).strip() == str(b).strip()


def compute_auroc(rows: list[dict], entropy_key: str, target_key: str) -> float | None:
    """AUROC for entropy as a predictor of incorrectness (higher entropy = wrong)."""
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    scores, labels = [], []
    for r in rows:
        if entropy_key not in r:
            continue
        entropy = r[entropy_key].get("entropy")
        target = r.get(target_key)
        if entropy is None or target is None:
            continue
        scores.append(entropy)
        labels.append(0 if target else 1)  # 1 = incorrect (target class)
    if len(set(labels)) < 2 or not scores:
        return None
    return float(roc_auc_score(labels, scores))


def summarize(all_rows: list[dict]) -> dict:
    summary: dict = {}
    benches = sorted(set(r["bench"] for r in all_rows))
    for b in benches:
        b_rows = [r for r in all_rows if r["bench"] == b]
        s: dict = {"n": len(b_rows)}
        for variant in ("se_math", "se_nli"):
            if not any(variant in r for r in b_rows):
                continue
            auroc_plur = compute_auroc(b_rows, variant, "plurality_correct")
            auroc_any = compute_auroc(b_rows, variant, "any_correct")
            s[variant] = {"auroc_plurality": auroc_plur, "auroc_any_correct": auroc_any}
        summary[b] = s

    print(f"\n{'=' * 60}\nSemantic Entropy Summary\n{'=' * 60}")
    for b, s in summary.items():
        print(f"\n[{b}]  n={s['n']}")
        for variant in ("se_math", "se_nli"):
            if variant in s:
                v = s[variant]
                print(f"  {variant:7s}  AUROC(plurality)={v['auroc_plurality']}  AUROC(any-correct)={v['auroc_any_correct']}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nli", action="store_true")
    ap.add_argument("--skip-math", action="store_true")
    ap.add_argument("--bench", choices=["math175", "aime", "cleanmath", "all"], default="all")
    args = ap.parse_args()

    do_math = not args.skip_math
    do_nli = not args.skip_nli
    if not (do_math or do_nli):
        print("Nothing to do (both variants skipped).")
        return

    benches = list(BENCHES.keys()) if args.bench == "all" else [args.bench]

    all_rows: list[dict] = []
    for b in benches:
        probs = BENCHES[b]()
        if not probs:
            continue
        print(f"\n>>> Phase 1 [{b}]: sampling {len(probs)} problems × {N_SAMPLES} × T={TEMP}")
        cache = phase1_sample(b, probs)
        print(f"\n>>> Phase 2 [{b}]: scoring with math={do_math} nli={do_nli}")
        rows = phase2_score(b, probs, cache, do_math, do_nli)
        all_rows.extend(rows)

        with open(OUT_PATH, "w") as f:
            json.dump({
                "n_samples_per_problem": N_SAMPLES,
                "temperature": TEMP,
                "solver": SOLVER_MODEL,
                "rows": all_rows,
            }, f, indent=2, default=str)

    summary = summarize(all_rows)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "n_samples_per_problem": N_SAMPLES,
            "temperature": TEMP,
            "solver": SOLVER_MODEL,
            "rows": all_rows,
            "summary": summary,
        }, f, indent=2, default=str)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
