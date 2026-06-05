"""Hybrid LLM-script × executable-verdict modality experiment.

Each LLM extractor in the DAJV cache emits two distinct signals per
problem:

  (A) Structural signal:  classification == 'working'
                          --- did the LLM emit a runnable verifier script?
  (B) Executable signal:  candidate_verdict is True
                          --- does that script accept the candidate?

The default DAJV accept is the AND of (A) and (B). Treating them as
two distinct extractors per LLM yields an 8-extractor ensemble from
the same 4-LLM cache. We then measure:

  1. Within-modality cross-LLM Cohen's κ          (4×3 / 2 = 6 pairs)
  2. Within-LLM cross-modality κ                  (4 pairs)
  3. Cross-LLM cross-modality κ                   (4×3 = 12 ordered pairs)

If cross-modality κ is materially lower than within-modality κ, the
hybrid ensemble carries an axis of independence that pure LLM-jury
verification cannot exploit. This is the central claim of the
ExeVer × DAJV pivot recommended by the novelty audit.

Saves: artifacts/hybrid_modality.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.dependency.kappa import cohen_kappa
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
    "E08S_gpt_4o":           HERE.parent / "../MATHAI/results/exp55_gpt4o_extractor.json",
    "E04S_gpt_4_1":          HERE.parent / "../MATHAI/results/exp56_gpt41_extractor.json",
}
# Auto-extend with gpt-5 if its cache is complete (>= 300 records)
_GPT5_PATH = HERE.parent / "../MATHAI/results/exp57_gpt5_extractor.json"
if _GPT5_PATH.exists():
    try:
        import json as _json
        _gpt5 = _json.load(open(_GPT5_PATH))
        if len(_gpt5.get("results", [])) >= 300:
            GROUP_B["E10S_gpt_5"] = _GPT5_PATH
    except Exception:
        pass
# Auto-extend with Anthropic Opus 4.7 if its cache is complete.
_OPUS47_PATH = HERE.parent / "../MATHAI/results/exp58_opus47_extractor.json"
if _OPUS47_PATH.exists():
    try:
        import json as _json
        _opus = _json.load(open(_OPUS47_PATH))
        if len(_opus.get("results", [])) >= 300:
            GROUP_B["E01A_claude_opus_4_7"] = _OPUS47_PATH
    except Exception:
        pass
_OPUS46_PATH = HERE.parent / "../MATHAI/results/exp59_opus46_extractor.json"
if _OPUS46_PATH.exists():
    try:
        import json as _json
        if len(_json.load(open(_OPUS46_PATH)).get("results", [])) >= 300:
            GROUP_B["E02A_claude_opus_4_6"] = _OPUS46_PATH
    except Exception:
        pass
_HAIKU45_PATH = HERE.parent / "../MATHAI/results/exp61_haiku45_extractor.json"
if _HAIKU45_PATH.exists():
    try:
        import json as _json
        if len(_json.load(open(_HAIKU45_PATH)).get("results", [])) >= 300:
            GROUP_B["E03A_claude_haiku_4_5"] = _HAIKU45_PATH
    except Exception:
        pass
_LLAMA33_PATH = HERE.parent / "../MATHAI/results/exp62_llama33_70b_extractor.json"
if _LLAMA33_PATH.exists():
    try:
        import json as _json
        if len(_json.load(open(_LLAMA33_PATH)).get("results", [])) >= 300:
            GROUP_B["E13_llama_3_3_70B"] = _LLAMA33_PATH
    except Exception:
        pass
_QWEN235_PATH = HERE.parent / "../MATHAI/results/exp63_qwen3_235b_extractor.json"
if _QWEN235_PATH.exists():
    try:
        import json as _json
        if len(_json.load(open(_QWEN235_PATH)).get("results", [])) >= 300:
            GROUP_B["E14_qwen_3_235B"] = _QWEN235_PATH
    except Exception:
        pass

ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def kappa(a: list[bool], b: list[bool]) -> float:
    return cohen_kappa(a, b)


def run_on_bench(bench: str | None, tag: str) -> dict:
    aligned = align_extractor_caches(GROUP_B, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "bench": bench, "n": n, "skipped": "too few"}

    k = len(aligned["extractor_ids"])
    extractor_ids = aligned["extractor_ids"]

    # Restrict to wrong-candidate subset for joint-FP analysis
    wrong_idx = [j for j, c in enumerate(aligned["solver_correct"]) if not c]
    if len(wrong_idx) < 10:
        return {"tag": tag, "bench": bench, "n_wrong": len(wrong_idx),
                "skipped": "too few wrong candidates"}

    # Build the 4 structural + 4 executable signals on wrong subset
    structural = [[aligned["working"][i][j] for j in wrong_idx]
                  for i in range(k)]
    executable = [[(aligned["candidate_verdict"][i][j] is True)
                   for j in wrong_idx]
                  for i in range(k)]

    pairs: dict[str, list[float]] = {
        "within_modality_struct":   [],   # struct[i] vs struct[j], i<j
        "within_modality_exec":     [],   # exec[i]   vs exec[j]
        "within_llm_cross_modality": [],  # struct[i] vs exec[i]
        "cross_llm_cross_modality": [],   # struct[i] vs exec[j], i!=j
    }

    pair_details: list[dict] = []

    # within-modality struct
    for i in range(k):
        for j in range(i + 1, k):
            kij = kappa(structural[i], structural[j])
            pairs["within_modality_struct"].append(kij)
            pair_details.append({
                "type": "within_modality_struct",
                "left": extractor_ids[i], "right": extractor_ids[j],
                "kappa": kij,
            })

    # within-modality exec
    for i in range(k):
        for j in range(i + 1, k):
            kij = kappa(executable[i], executable[j])
            pairs["within_modality_exec"].append(kij)
            pair_details.append({
                "type": "within_modality_exec",
                "left": extractor_ids[i], "right": extractor_ids[j],
                "kappa": kij,
            })

    # within-LLM cross-modality
    for i in range(k):
        kij = kappa(structural[i], executable[i])
        pairs["within_llm_cross_modality"].append(kij)
        pair_details.append({
            "type": "within_llm_cross_modality",
            "left": extractor_ids[i], "right": extractor_ids[i],
            "kappa": kij,
        })

    # cross-LLM cross-modality (ordered: structural[i] vs executable[j], i!=j)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            kij = kappa(structural[i], executable[j])
            pairs["cross_llm_cross_modality"].append(kij)
            pair_details.append({
                "type": "cross_llm_cross_modality",
                "left": extractor_ids[i], "right": extractor_ids[j],
                "kappa": kij,
            })

    def summarize(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "median": None, "mean": None,
                    "min": None, "max": None}
        s = sorted(vals)
        m = s[len(s) // 2] if len(s) % 2 else (s[len(s)//2 - 1] + s[len(s)//2]) / 2
        return {
            "n": len(vals),
            "median": m,
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    summary = {
        bucket: summarize(vals) for bucket, vals in pairs.items()
    }

    # Marginal accept rates per extractor per modality on the wrong subset
    margins = {}
    for i, eid in enumerate(extractor_ids):
        margins[eid] = {
            "struct_pi": sum(structural[i]) / len(structural[i]),
            "exec_pi":   sum(executable[i]) / len(executable[i]),
        }

    out = {
        "tag": tag,
        "bench": bench,
        "n_total": n,
        "n_wrong": len(wrong_idx),
        "extractor_ids": extractor_ids,
        "marginals_wrong_subset": margins,
        "summary": summary,
        "pair_details": pair_details,
    }
    print(f"\n=== {tag} (n_wrong={len(wrong_idx)}) ===")
    print(json.dumps(summary, indent=2))
    return out


def main() -> None:
    results = []
    for bench, tag in [("math175", "math175"), (None, "B_full")]:
        try:
            results.append(run_on_bench(bench, tag))
        except Exception as e:
            print(f"  ERROR on tag={tag}: {e}")
    save_artifact(results, ARTIFACTS / "hybrid_modality.json")
    print(f"\nWrote {ARTIFACTS / 'hybrid_modality.json'}")


if __name__ == "__main__":
    main()
