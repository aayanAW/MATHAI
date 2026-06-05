"""verifyensemble: Dependency-Aware Jury Verification for LLM Math Reasoning.

Top-level package. See README.md for usage.

Modules:
    sandbox       - SymPy verifier execution + adversarial probe filter
    extractors    - LLM extractor wrappers (Together AI / OpenAI / Anthropic)
    dependency    - pairwise dependency estimators (kappa, joint-FP, CIG)
    aggregate     - vote aggregation rules (naive consensus, CARE, DAJV)
    theory        - numerical validation of Theorem 1 / Theorem 2
    evaluation    - risk-coverage curves, ECE, Brier, bootstrap CIs
    utils         - I/O, X-SGRV cache loader

Project: DAJV (Dependency-Aware Jury Verification)
Paper:   "Two Modalities Are Better Than One: Cross-Modality
          Independence in Executable Spec-Grounded LLM-Jury Verification
          for Math Reasoning"
"""

__version__ = "0.1.15"
__all__ = [
    "sandbox",
    "extractors",
    "dependency",
    "aggregate",
    "theory",
    "evaluation",
    "utils",
]
