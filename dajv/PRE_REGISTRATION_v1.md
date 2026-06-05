# DAJV Pre-Registration v1

**Commit timestamp:** 2026-05-25 (initial commit before any new data collection)
**Project:** DAJV — Dependency-Aware Jury Verification for LLM Math Reasoning
**Library codename:** `verifyensemble`
**Pre-registration scope:** Hypotheses H1–H6 and their decision branches.

This document is the locked, time-stamped version of all hypotheses,
splits, baselines, metrics, and pass/fail thresholds for the DAJV
project. Any change to this document after the commit timestamp must be
recorded as an explicit `PRE_REGISTRATION_v<N>.md` with a clear
diff and rationale.

---

## Hypotheses

### H1 (theory)

There exists a concentration inequality bounding joint false-positive
rate as a function of measured pairwise dependency that is strictly
tighter than the trivial product-of-marginals bound and strictly looser
than the union bound; the gap shrinks with $\rho \to 0$.

**Test:** Prove Theorem 1 in [paper/sections/theory.tex](paper/sections/theory.tex)
and validate numerically via the Monte Carlo procedure in
[scripts/run_synthetic_validation.py](scripts/run_synthetic_validation.py).

**Pass criterion:** Theorem 1 proven; in $\geq 95\%$ of 45 Monte Carlo
trials spanning $k \in \{2, 5, 12\}$, $\rho \in \{0, 0.1, 0.2, 0.4, 0.6\}$,
$\pi \in \{0.05, 0.10, 0.20\}$, the empirical joint-acceptance
probability is bounded above by the DAJV bound to within $2\sigma$ MC
noise.

**Status as of 2026-05-25:** PASS — 45/45 trials satisfy the bound.

### H2 (empirical, calibration)

Across $k \geq 4$ frontier LLM verifiers, the empirical joint
false-positive rate on a held-out set deviates from the independence
prediction by $\geq 3\times$ on at least 50% of pairs.

**Test:** Run [scripts/run_dependency_mapping.py](scripts/run_dependency_mapping.py)
on the cached X-SGRV extractor outputs.

**Pass criterion:** Median off-diagonal ratio (joint-FP /
independence-FP) $\geq 3$ on at least one cross-family benchmark of
$n \geq 30$ wrong candidates.

**Status as of 2026-05-25:** PASS — median ratio is $15.7\times$ on
math175 ($n_{\text{wrong}} = 47$); median Cohen's $\kappa = 0.79$.

### H3 (empirical, deployment)

A dependency-aware aggregation rule reduces expected calibration error
(ECE) by $\geq 30\%$ versus naive consensus on a held-out test set
drawn from the same problem distribution as the calibration set.

**Test:** Run [scripts/run_aggregation_comparison.py](scripts/run_aggregation_comparison.py)
with a 70/30 calibration/test split (seed=42).

**Pass criterion:** ECE reduction (naive unanimous $\to$ DAJV) is
$\geq 30\%$ on at least one cross-family benchmark of $n_{\text{test}} \geq 50$.

**Status as of 2026-05-25:** PASS — ECE reduction is $47\%$ on
math175 test split ($n_{\text{test}} = 53$).

### H4 (empirical, generalization)

Pairwise dependency $\rho_{ij}$ estimated on a calibration set
(MATH-500) transfers to held-out benchmarks (AIME 2025, CleanMath,
OlympiadBench) within $\pm 0.10$ of the calibration value on $\geq 70\%$
of pairs.

**Test:** Cross-benchmark transfer experiment. Estimate $\rho$ on
math175 calibration; deploy on AIME / CleanMath; compare per-pair
$|\rho_{\text{calibration}} - \rho_{\text{deployment}}|$.

**Pass criterion:** $\geq 70\%$ of pairs satisfy $|\Delta\rho| \leq 0.10$.

**Status as of 2026-05-25:** PENDING — requires new extractor calls on
the AIME / CleanMath wrong-candidate subset; deferred to the budget-approved
empirical run.

### H5 (empirical, Pareto)

At any matched coverage $\in [5\%, 90\%]$ on the held-out test set, the
dependency-aware consensus achieves precision $\geq$ Qwen-PRM,
Skywork-PRM, GenPRM, naive consensus, and CARE.

**Test:** Risk-coverage curve comparison
([scripts/run_aggregation_comparison.py](scripts/run_aggregation_comparison.py)).

**Pass criterion:** DAJV's risk-coverage curve is below (better than)
every baseline curve at all coverages where both methods provide a
probabilistic output.

**Status as of 2026-05-25:** PARTIAL PASS — DAJV Pareto-dominates
naive unanimous and CARE on math175 at all coverages; risk-coverage AUC
not yet computed against PRMs (PRMs require new evaluation runs).

### H6 (mechanism, optional)

Measured dependency $\rho_{ij}$ correlates with shared training-data
overlap, shared base-architecture family, and shared instruction-tuning
recipe.

**Test:** Stratify the 6-pair $\rho_{ij}$ values by (a) shared lab,
(b) shared base-model family, (c) shared instruction-tuning data.

**Pass criterion:** Within-lab pairs have systematically higher $\rho$
than cross-lab pairs (one-sided test, $p < 0.05$).

**Status as of 2026-05-25:** PENDING — requires within-lab pairs which
are absent from the current 4-extractor cache (all 4 are cross-lab).
Deferred to extended extractor set (E01--E12 in
[ARCHITECTURE_PIVOT_A.md](../ARCHITECTURE_PIVOT_A.md)).

---

## Falsification branches

| Branch | Condition | Action |
|---|---|---|
| A | All five primary hypotheses (H1–H5) hold | Submit to ICML 2027 main |
| B | H1 + H3 hold, H4 fails | Reframe around in-distribution calibration only; submit to workshop |
| C | H1 fails | Drop theory; pivot to mechanism study |
| D | H2 fails | Flip paper to "Cross-family LLM verifiers on math are empirically independent" |
| E | H5 fails | Abandon main-track submission; release library + benchmark as workshop paper |

**Branch reached as of 2026-05-25:** A-partial (H1, H2, H3 confirmed
on cached data; H4, H5, H6 pending new data collection).

---

## Locked parameters

- **Adversarial probe set:** $\{\hat a - 1, \hat a + 1, \hat a + 7,
  \hat a \times 2 \text{ if } \hat a \ne 0 \text{ else } 100\} \setminus
  \{\hat a\}$ for integer $\hat a$; fallback $\{0, 1, -1, 42, 100\}
  \setminus \{\hat a\}$ for non-integer.
- **Sandbox timeout:** 10 seconds wall-clock.
- **Calibration / test split:** 70 / 30 with fixed RNG seed = 42.
- **Working-verifier definition:** identical to the cached X-SGRV
  classification (`classification == "working"` AND `candidate_verdict
  is True`).
- **Default decision thresholds:** $\tau_{\text{commit}} = 0.95$,
  $\tau_{\text{abstain}} = 0.50$.
- **Bootstrap settings:** 500 resamples for dependency-matrix CIs,
  1000 resamples for any metric reported with CI in the paper.
- **Multiple-comparison correction:** Holm-Bonferroni across the four
  primary hypothesis tests reported in the main paper.

## Frozen extractor set

The cached extractor outputs used in the current empirical work:

- $E_5$ = `gpt-oss-120B` (OpenAI; Together AI endpoint)
- $E_6$ = `gpt-5-mini` (OpenAI)
- $E_7$ = `claude-sonnet-4-6` (Anthropic)
- $E_9$ = `Qwen3-Coder-480B-A35B-Instruct` (Alibaba; Together AI)

The extended set committed for the camera-ready empirical run is the
12-extractor list in
[ARCHITECTURE_PIVOT_A.md](../ARCHITECTURE_PIVOT_A.md) §5.2 (Workstream B).

## Frozen prompt

The extractor prompt is the frozen X-SGRV prompt
([verifyensemble/extractors/prompt.py](verifyensemble/extractors/prompt.py)),
unchanged.

## Frozen seeds

All Python RNG seeds = 42. Bootstrap seed = 42. Calibration/test
split seed = 42.
