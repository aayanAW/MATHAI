"""Merge exp31/exp32 (n=175) with exp37 (n=323 new) into the full n=500 analysis.

Computes:
  - Llama raw precision
  - DeepSeek raw precision
  - Deployment-time filter (candidate ±1, +7, ×2, 42)
  - Llama + filter
  - DeepSeek + filter
  - Llama ∩ DeepSeek consensus strict
  - Llama ∪ DeepSeek consensus loose

All with Clopper-Pearson 95% CIs.
"""
import json
from pathlib import Path
from scipy.stats import binomtest

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")

# Load n=175 base
exp31 = json.load(open(RESULTS / "exp31_xsgrv_math175_llama70b.json"))
exp32 = json.load(open(RESULTS / "exp32_math175_deepseek.json"))

# Load n=323 new
exp37_l = json.load(open(RESULTS / "exp37_xsgrv_math500_llama70b.json"))
exp37_d = json.load(open(RESULTS / "exp37_xsgrv_math500_deepseek.json"))

# Merge (union by problem id; new rows from exp37 shouldn't overlap with exp25 ids)
def merge(base, extension):
    out = {r["id"]: r for r in base["results"]}
    for r in extension["results"]:
        if r["id"] not in out:
            out[r["id"]] = r
    return list(out.values())


llama_all = merge(exp31, exp37_l)
deepseek_all = merge(exp32, exp37_d)
print(f"Merged: llama={len(llama_all)}, deepseek={len(deepseek_all)}")

# Ensure only intersecting IDs are used for fair comparison
l_ids = {r["id"] for r in llama_all}
d_ids = {r["id"] for r in deepseek_all}
common = l_ids & d_ids
print(f"Common ids: {len(common)}")

llama_by_id = {r["id"]: r for r in llama_all if r["id"] in common}
deepseek_by_id = {r["id"]: r for r in deepseek_all if r["id"] in common}


def _prec_ci(correct, total):
    if total == 0:
        return (None, None, None)
    p = correct / total
    ci = binomtest(correct, total).proportion_ci(0.95, "exact")
    return (p, float(ci.low), float(ci.high))


def raw_stats(data, label):
    total = len(data)
    tier = [r for r in data.values() if r.get("candidate_verdict") is True]
    correct = [r for r in tier if r.get("solver_correct")]
    p, lo, hi = _prec_ci(len(correct), len(tier))
    # working verifiers
    working = [r for r in data.values() if r.get("classification") == "working"]
    adv_tests = len(working) * 4
    adv_fps = sum(r.get("adv_fp_count", 0) or 0 for r in working)
    print(f"\n{label}")
    print(f"  n_total: {total}")
    print(f"  working: {len(working)} ({len(working)/total:.1%})")
    print(f"  top-tier (candidate accepted): {len(tier)} ({len(tier)/total:.1%})")
    print(f"  top-tier correct: {len(correct)}")
    if p is not None:
        print(f"  top-tier precision: {len(correct)}/{len(tier)} = {p:.4f} [{lo:.3f}, {hi:.3f}]")
    if adv_tests:
        print(f"  adv FPs: {adv_fps}/{adv_tests} = {adv_fps/adv_tests:.2%}")
    return {
        "n": total,
        "n_working": len(working),
        "n_tier": len(tier),
        "n_tier_correct": len(correct),
        "precision": p,
        "precision_ci": (lo, hi),
        "adv_fps": adv_fps,
        "adv_tests": adv_tests,
    }


print("=" * 70)
print("MATH-500 n=500 scale-up (merged exp31+exp37_llama, exp32+exp37_deepseek)")
print("=" * 70)
llama_raw = raw_stats(llama_by_id, "Llama-3.3-70B raw")
deepseek_raw = raw_stats(deepseek_by_id, "DeepSeek-V3 raw")


# Deployment-time filter: verifier must reject adversarial perturbations of the candidate.
# We don't have candidate-relative adversarial probes cached for exp37; instead we use the
# gold-relative adv_verdicts to infer "broken" status: if adv_fp_count > 0, reject this verifier.
# This is a stricter filter than exp33's deployment-time protocol (which uses candidate-relative)
# but serves as a conservative approximation.
def filter_stats(data, label):
    total = len(data)
    tier = [r for r in data.values()
            if r.get("candidate_verdict") is True
            and (r.get("adv_fp_count", 0) or 0) == 0]
    correct = [r for r in tier if r.get("solver_correct")]
    p, lo, hi = _prec_ci(len(correct), len(tier))
    print(f"\n{label}")
    print(f"  top-tier: {len(tier)}")
    if p is not None:
        print(f"  precision: {len(correct)}/{len(tier)} = {p:.4f} [{lo:.3f}, {hi:.3f}]")
    return {"n_tier": len(tier), "n_tier_correct": len(correct), "precision": p}


llama_filt = filter_stats(llama_by_id, "Llama-3.3-70B + gold-adv filter")
deepseek_filt = filter_stats(deepseek_by_id, "DeepSeek-V3 + gold-adv filter")


# Consensus strict: both extractors produce working verifiers AND both accept the candidate
def consensus_strict():
    accept_both_working = []
    for pid in common:
        l = llama_by_id[pid]
        d = deepseek_by_id[pid]
        l_working = l.get("classification") == "working"
        d_working = d.get("classification") == "working"
        l_accept = l.get("candidate_verdict") is True
        d_accept = d.get("candidate_verdict") is True
        if l_working and d_working and l_accept and d_accept:
            accept_both_working.append(pid)
    correct = [pid for pid in accept_both_working if llama_by_id[pid].get("solver_correct")]
    p, lo, hi = _prec_ci(len(correct), len(accept_both_working))
    print(f"\nConsensus strict (Llama ∩ DeepSeek, both working, both accept)")
    print(f"  top-tier: {len(accept_both_working)}")
    if p is not None:
        print(f"  precision: {len(correct)}/{len(accept_both_working)} = {p:.4f} [{lo:.3f}, {hi:.3f}]")
    return {"n_tier": len(accept_both_working), "n_tier_correct": len(correct), "precision": p}


cons_strict = consensus_strict()


# Consensus loose: at least one extractor produces a working verifier and both accept
def consensus_loose():
    accept_both = []
    for pid in common:
        l = llama_by_id[pid]
        d = deepseek_by_id[pid]
        l_accept = l.get("candidate_verdict") is True
        d_accept = d.get("candidate_verdict") is True
        l_working = l.get("classification") == "working"
        d_working = d.get("classification") == "working"
        if (l_working or d_working) and l_accept and d_accept:
            accept_both.append(pid)
    correct = [pid for pid in accept_both if llama_by_id[pid].get("solver_correct")]
    p, lo, hi = _prec_ci(len(correct), len(accept_both))
    print(f"\nConsensus loose (at least one working, both accept)")
    print(f"  top-tier: {len(accept_both)}")
    if p is not None:
        print(f"  precision: {len(correct)}/{len(accept_both)} = {p:.4f} [{lo:.3f}, {hi:.3f}]")
    return {"n_tier": len(accept_both), "n_tier_correct": len(correct), "precision": p}


cons_loose = consensus_loose()

# Save
out = {
    "n": len(common),
    "llama_raw": llama_raw,
    "deepseek_raw": deepseek_raw,
    "llama_filt": llama_filt,
    "deepseek_filt": deepseek_filt,
    "consensus_strict": cons_strict,
    "consensus_loose": cons_loose,
}
with open(RESULTS / "exp37_n500_analysis.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nSaved → exp37_n500_analysis.json")
