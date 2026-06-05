# Condensed abstract (NeurIPS-length, ~280 words)

Same-model LLM self-verification on math suffers from **correlated-error collapse**: when the solver writes both a reasoning chain and the checks that validate it, errors propagate into the checks. We measure a 13.8% false-acceptance rate on MATH-500 with Qwen2.5-Math-7B that scales monotonically with difficulty (1.4% at L1, 30.8% at L5). The problem is compounded by benchmark contamination: on AIME 2025 and a 125-problem CleanMath combo of post-cutoff olympiad problems, every template- and agreement-based selective-prediction signal (template SGRV, self-consistency 4/4, ExeVer, Semantic Entropy in SymPy and NLI-clustering variants, p(True)) collapses, with p(True) AUROC *worse than random* on contamination-clean benchmarks (0.46 AIME, 0.47 CleanMath).

We propose **X-SGRV** (cross-family symbolic verification): a large cross-family LLM extractor reads only the problem statement and emits a Python SymPy `verify(answer)` function executed against the solver's candidate. Two deployment-safe mechanisms harden the signal: a gold-free adversarial filter probing perturbations of the candidate, and cross-extractor consensus requiring two independent extractors to both accept. A **pre-registered scale-up to the full MATH-500 (n=498)** yields consensus precision **171/171 = 1.000 [0.979, 1.000] at 34.3% coverage** — fulfilling the pre-commitment's Branch A (≥99%). A solver-rotation experiment swapping the 7B base solver for DeepSeek-V3 yields 9/9 on CleanMath (95% lower bound 66%, up from 40% with the 7B solver) and 4/4 on AIME 2025, confirming selective-prediction behavior is solver-independent.

At matched coverage, X-SGRV ties Skywork-o1-Open-PRM-7B on MATH-500 (0.938) and CleanMath (1.000), and doubles Qwen2.5-Math-PRM-7B on CleanMath (0.500→1.000). The **honest framing**: X-SGRV correctly collapses its coverage on contamination-clean benchmarks rather than claiming broad high-precision coverage — a deployment-safe abstention signal, not a precision-maximizer. All code, prompts, probe sets, and pre-registration released.

---

## Even shorter version (~180 words, for venues with tighter abstracts)

LLM self-verification on math suffers from *correlated-error collapse*: same-model SymPy verification has a 13.8% false-acceptance rate on MATH-500 that scales with difficulty. Every agreement-based signal (self-consistency, Semantic Entropy, p(True)) further collapses on contamination-clean benchmarks (AIME 2025, CleanMath). We propose **X-SGRV**: a cross-family LLM extractor reads only the problem statement and emits an executable SymPy `verify(answer)` function. A pre-registered scale-up yields consensus precision **171/171 = 1.000 [0.979, 1.000]** on full MATH-500 (n=498) — Branch A of the pre-commitment. A DeepSeek-V3 solver rotation raises CleanMath's 95% lower bound from 40% (4/4) to 66% (9/9). X-SGRV ties Skywork-o1-Open-PRM-7B on MATH and CleanMath at matched coverage and doubles Qwen2.5-Math-PRM-7B on CleanMath. X-SGRV correctly collapses coverage on hard clean benchmarks rather than claiming broad precision — a deployment-safe abstention signal with no training, no GPU infrastructure, and interpretable SymPy verifiers.

---

## One-liner (for Twitter/social/Regeneron STS essay)

X-SGRV has a cross-family LLM read only the problem statement and write a Python SymPy verifier that the base model's answer must pass — achieving 171/171 = 100% precision on full MATH-500 via cross-extractor consensus, while correctly collapsing to single-digit coverage on contamination-clean olympiad benchmarks where it is structurally uncertain.
