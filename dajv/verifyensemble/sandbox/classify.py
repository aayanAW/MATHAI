"""Classify a verifier as working / wrong-spec / trivial-or-broken.

A verifier is **working** iff:
  (a) accepts the gold answer
  (b) rejects each of four gold-relative adversarial wrong candidates
        {gold - 1, gold + 1, gold + 7, 42 if 42 != gold else 0}
  (c) halts within ``timeout`` seconds
  (d) raises no unhandled exception

A verifier is **trivial-or-broken** iff it accepts the gold AND accepts
at least one adversarial wrong candidate (i.e. it always accepts, or has
modular/additive bias).

A verifier is **wrong-spec** iff it fails to accept the gold.

This function uses the GOLD answer and is therefore for offline
classification only; the deployment-time filter in adversarial.py uses
only the candidate.
"""
from __future__ import annotations

from typing import Literal

from verifyensemble.sandbox.executor import execute_verifier

Classification = Literal["working", "wrong_spec", "trivial_or_broken", "exec_error"]


def classify(script: str, gold: str, timeout: float = 10.0) -> Classification:
    """Classify a verifier given the gold answer."""
    gold_ver = execute_verifier(script, gold, timeout=timeout)
    if gold_ver.verdict is None:
        return "exec_error"
    if gold_ver.verdict is False:
        return "wrong_spec"

    # gold accepted; run adversarial wrong candidates
    try:
        g = int(str(gold).strip())
        adv_candidates = [str(g - 1), str(g + 1), str(g + 7),
                          "42" if g != 42 else "0"]
    except (ValueError, TypeError):
        adv_candidates = ["0", "1", "-1", "42", "100"]
        adv_candidates = [a for a in adv_candidates if a != str(gold).strip()]

    for adv in adv_candidates:
        adv_ver = execute_verifier(script, adv, timeout=timeout)
        if adv_ver.verdict is True:
            return "trivial_or_broken"
    return "working"
