"""Deployment-time adversarial filter.

Probes a verifier against perturbations of the candidate answer WITHOUT
ever using the gold answer. Allows the pipeline to abstain on broken /
tautological verifiers at deployment time.

Probe set (locked, pre-registered):
    For integer candidates ``c``:
        {c-1, c+1, c+7, c*2 if c != 0 else 100, c+100} \\ {c}
    For non-integer candidates:
        {0, 1, -1, 42, 100} \\ {candidate}

Rationale: each probe targets a distinct degenerate-verifier class
(trivial-True, additive-bias, modular residue, multiplicative scaling).
"""
from __future__ import annotations

from typing import List

from verifyensemble.sandbox.executor import execute_verifier


def deployment_time_probes(candidate_answer: str) -> List[str]:
    """Generate gold-free wrong-answer probes near the candidate.

    All probes are guaranteed distinct from the candidate. Order is
    deterministic for reproducibility.
    """
    cand_str = str(candidate_answer).strip()
    try:
        c = int(cand_str)
        probes_int = {c - 1, c + 1, c + 7, c * 2 if c != 0 else 100, c + 100}
        probes_int.discard(c)
        return [str(p) for p in sorted(probes_int)]
    except (ValueError, TypeError):
        fallbacks = ["0", "1", "-1", "42", "100"]
        return [p for p in fallbacks if p != cand_str]


def deployment_time_filter(
    script: str,
    candidate_answer: str,
    timeout: float = 10.0,
) -> tuple[bool, list[dict]]:
    """Probe a verifier against gold-free perturbations of the candidate.

    Returns:
        broken: True if the verifier accepts any perturbation (and is
            therefore unreliable; the caller should abstain).
        probe_results: per-probe verdicts for diagnostics.
    """
    probes = deployment_time_probes(candidate_answer)
    probe_results: list[dict] = []
    broken = False
    for probe in probes:
        if probe == str(candidate_answer).strip():
            continue
        ver = execute_verifier(script, probe, timeout=timeout)
        probe_results.append({
            "probe": probe,
            "verdict": ver.verdict,
            "error": ver.error,
        })
        if ver.verdict is True:
            broken = True
    return broken, probe_results
