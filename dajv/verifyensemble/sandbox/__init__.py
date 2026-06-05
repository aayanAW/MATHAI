"""Sandbox subpackage: execute extractor-generated SymPy verifiers safely.

The sandbox runs each emitted ``verify(answer) -> bool`` function in a
Python subprocess with a ``SIGALRM``-based wall-clock timeout. The
adversarial filter probes the verifier against perturbations of the
candidate answer without ever using the gold answer (deployment-safe).

Public API:
    execute_verifier(script, candidate, timeout=10.0) -> VerificationResult
    deployment_time_filter(script, candidate, timeout=10.0) -> (broken, probes)
    classify(script, gold, timeout=10.0) -> ('working' | 'wrong_spec' | 'trivial')
"""
from verifyensemble.sandbox.adversarial import (
    deployment_time_filter,
    deployment_time_probes,
)
from verifyensemble.sandbox.classify import classify
from verifyensemble.sandbox.executor import (
    VerificationResult,
    execute_verifier,
)

__all__ = [
    "VerificationResult",
    "execute_verifier",
    "deployment_time_filter",
    "deployment_time_probes",
    "classify",
]
