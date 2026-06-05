"""Cross-extractor X-SGRV consensus analysis.

For each problem, we have two independent verifier verdicts on the solver's
candidate answer — one from the Llama-3.3-70B extractor (exp31/exp30) and one
from the DeepSeek-V3 extractor (exp32). Consensus = trust the top-tier label
only when BOTH extractors produce working verifiers AND both accept the
candidate.

This mechanism is meant to drive the residual false-positive rate toward zero.
Cost: strictly lower coverage (we only keep problems where both extractors
succeed).

Inputs:
    results/exp31_xsgrv_math175_llama70b.json    — Llama-70B on MATH-500 n=175
    results/exp32_math175_deepseek.json           — DeepSeek-V3 on MATH-500 n=175
    results/exp30_aime_llama70b.json              — Llama-70B on AIME 2025
    results/exp32_aime_deepseek.json               — DeepSeek-V3 on AIME 2025

Output:
    results/xsgrv_consensus.json
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_PATH = RESULTS_DIR / "xsgrv_consensus.json"

CONDITIONS = [
    {
        "label": "MATH-500 n=175",
        "llama": RESULTS_DIR / "exp31_xsgrv_math175_llama70b.json",
        "deepseek": RESULTS_DIR / "exp32_math175_deepseek.json",
    },
    {
        "label": "AIME 2025 n=30",
        "llama": RESULTS_DIR / "exp30_aime_llama70b.json",
        "deepseek": RESULTS_DIR / "exp32_aime_deepseek.json",
    },
]


def load_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["id"]: r for r in data.get("results", [])}


def analyze_condition(label: str, llama_path: Path, deepseek_path: Path) -> dict:
    llama = load_results(llama_path)
    deepseek = load_results(deepseek_path)

    if not llama or not deepseek:
        return {
            "label": label,
            "error": f"missing file(s): llama={llama_path.exists()} deepseek={deepseek_path.exists()}",
        }

    shared_ids = sorted(set(llama.keys()) & set(deepseek.keys()))
    n = len(shared_ids)

    # Single-extractor top-tier stats (pre-consensus)
    llama_top = [i for i in shared_ids if llama[i].get("candidate_verdict") is True]
    llama_top_correct = sum(1 for i in llama_top if llama[i].get("solver_correct"))
    deepseek_top = [i for i in shared_ids if deepseek[i].get("candidate_verdict") is True]
    deepseek_top_correct = sum(1 for i in deepseek_top if deepseek[i].get("solver_correct"))

    # Consensus top tier: BOTH extractors say the verifier accepts the candidate
    # AND BOTH produced a working verifier (accepts gold, rejects adversarial).
    consensus_top: list[str] = []
    for i in shared_ids:
        l, d = llama[i], deepseek[i]
        l_working = l.get("classification") == "working"
        d_working = d.get("classification") == "working"
        l_accepts = l.get("candidate_verdict") is True
        d_accepts = d.get("candidate_verdict") is True
        if l_working and d_working and l_accepts and d_accepts:
            consensus_top.append(i)

    consensus_top_correct = sum(1 for i in consensus_top if llama[i].get("solver_correct"))

    # Loose-consensus top tier: both accept the candidate, but we don't require
    # the "working verifier" sign-off from adversarial rejection. Useful as an
    # ablation: how much of the gain comes from the adversarial-filter part?
    loose_top: list[str] = []
    for i in shared_ids:
        if llama[i].get("candidate_verdict") is True and deepseek[i].get("candidate_verdict") is True:
            loose_top.append(i)
    loose_top_correct = sum(1 for i in loose_top if llama[i].get("solver_correct"))

    # Disagreement diagnostics
    disagreements = 0
    both_broken = 0
    for i in shared_ids:
        l = llama[i].get("candidate_verdict")
        d = deepseek[i].get("candidate_verdict")
        if l is not None and d is not None and l != d:
            disagreements += 1
        if llama[i].get("classification") == "logic_broken" and deepseek[i].get("classification") == "logic_broken":
            both_broken += 1

    def _precision(correct: int, total: int) -> float | None:
        return correct / total if total else None

    return {
        "label": label,
        "n_shared": n,
        "llama_solo": {
            "top_tier_size": len(llama_top),
            "top_tier_correct": llama_top_correct,
            "precision": _precision(llama_top_correct, len(llama_top)),
        },
        "deepseek_solo": {
            "top_tier_size": len(deepseek_top),
            "top_tier_correct": deepseek_top_correct,
            "precision": _precision(deepseek_top_correct, len(deepseek_top)),
        },
        "consensus_strict": {
            "top_tier_size": len(consensus_top),
            "top_tier_correct": consensus_top_correct,
            "precision": _precision(consensus_top_correct, len(consensus_top)),
            "coverage": len(consensus_top) / n if n else None,
        },
        "consensus_loose": {
            "top_tier_size": len(loose_top),
            "top_tier_correct": loose_top_correct,
            "precision": _precision(loose_top_correct, len(loose_top)),
            "coverage": len(loose_top) / n if n else None,
        },
        "disagreements": disagreements,
        "both_logic_broken": both_broken,
    }


def main() -> None:
    report: dict = {}
    for cond in CONDITIONS:
        r = analyze_condition(cond["label"], cond["llama"], cond["deepseek"])
        report[cond["label"]] = r
        print(f"\n{'=' * 60}\n{cond['label']}\n{'=' * 60}")
        if "error" in r:
            print(f"  SKIP: {r['error']}")
            continue
        print(f"n_shared={r['n_shared']}")
        l, d, cs, cl = r["llama_solo"], r["deepseek_solo"], r["consensus_strict"], r["consensus_loose"]
        print(f"Llama-70B solo    top_tier={l['top_tier_size']:3d}  "
              f"prec={l['top_tier_correct']}/{l['top_tier_size']}" +
              (f" = {l['precision']:.4f}" if l["precision"] is not None else ""))
        print(f"DeepSeek-V3 solo  top_tier={d['top_tier_size']:3d}  "
              f"prec={d['top_tier_correct']}/{d['top_tier_size']}" +
              (f" = {d['precision']:.4f}" if d["precision"] is not None else ""))
        print(f"Consensus strict  top_tier={cs['top_tier_size']:3d}  "
              f"prec={cs['top_tier_correct']}/{cs['top_tier_size']}" +
              (f" = {cs['precision']:.4f}" if cs["precision"] is not None else "") +
              (f"  coverage={cs['coverage']:.3f}" if cs["coverage"] is not None else ""))
        print(f"Consensus loose   top_tier={cl['top_tier_size']:3d}  "
              f"prec={cl['top_tier_correct']}/{cl['top_tier_size']}" +
              (f" = {cl['precision']:.4f}" if cl["precision"] is not None else "") +
              (f"  coverage={cl['coverage']:.3f}" if cl["coverage"] is not None else ""))
        print(f"Disagreements: {r['disagreements']}   both-logic-broken: {r['both_logic_broken']}")

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
