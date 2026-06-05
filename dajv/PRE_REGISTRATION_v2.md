# DAJV Pre-Registration v2 — Pivot to Cross-Modality Framing

**Commit timestamp:** 2026-05-25 (post-novelty audit, post-7-extractor scale-up)
**Supersedes:** `PRE_REGISTRATION_v1.md` (kept on file for diff)
**Project:** DAJV — Dependency-Aware Jury Verification for LLM Math Reasoning
**Library codename:** `verifyensemble` (v0.1.5)

## Rationale for v2

The novelty-verification protocol (`NOVELTY_REPORT_2026-05-25.md`)
identified three concurrent 2026 manuscripts that subsume the
abstract problem of dependency-aware aggregation:

- Kuai et al. 2026 (arXiv:2604.07650) — joint-error bound on $\kappa$.
- Zhao et al. 2026 CARE (arXiv:2603.00039) — confounder-aware aggregation.
- Balasubramanian, Podkopaev, Kasiviswanathan 2026 (arXiv:2601.22336) — Ising-model dependence-aware label aggregation; strictly more general than DAJV's second-order copula.

A direct submission of the v1 framing would invite reviewer pushback
on novelty grounds. The pivot is to re-headline the paper around an
empirical finding *orthogonal* to all three: in math reasoning, the
LLM juror emits a deterministic Python verifier executed by an oracle,
and the script-writing modality (LLM self-classification) is essentially
independent ($\kappa \approx 0.02$--$0.06$) of the script-execution
modality (Python verdict), across $42$ cross-LLM pairs in a
$7$-extractor frontier ensemble. Pure LLM-jury aggregators in the
concurrent work cannot exploit this axis by construction.

The library, theorems, calibration, and empirical pipeline from v1
are preserved unchanged. Only the framing is repositioned: cross-modality
independence is the headline, dependency-aware aggregation theory is
demoted to "specialization of Balasubramanian's Ising framework to
binary executable votes."

## Locked hypotheses (v2)

**H1' (measurement, replaces v1 H1):** Across a frontier 7-extractor
ensemble, within-modality (LLM-to-LLM, same modality) median pairwise
$\kappa$ on the wrong-candidate subset exceeds $0.5$, while cross-LLM
cross-modality median pairwise $\kappa$ is below $0.15$. The gap is
robust to subset choice (math175 vs full Group B).

- **Pass criterion:** within-modality median $\kappa \geq 0.5$ AND cross-modality median $\kappa \leq 0.15$ on the wrong subset of the full Group B benchmark.
- **Status (v2 commit):** PASS. Within-modality executable median $0.720$; cross-LLM cross-modality $0.021$.

**H2', H3', H5' (calibration, unchanged from v1 H2/H3/H5):**

- H2': pairwise $\kappa$ on wrong-candidate subset exceeds $0.5$ across at least $50\%$ of pairs. PASS.
- H3': DAJV ECE reduction $\geq 30\%$ vs naive unanimous. PASS (47%).
- H5': DAJV Pareto-dominates Qwen-PRM and Skywork-PRM at matched coverage on contamination-clean splits. PASS (35--88× risk-coverage AUC gap).

**H4' (cross-benchmark transfer; unchanged from v1 H4):** $|\Delta \kappa| \leq 0.10$ across $\geq 70\%$ of pairs between math175 and AIME/CleanMath.

- **Status:** FAIL. Pre-registered Branch B activated; per-benchmark recalibration recommended.

**H6' (lab-pairing dependency, unchanged from v1 H6):** Within-lab median $\kappa$ exceeds cross-lab median $\kappa$.

- **Status (k=7 scale-up):** PARTIAL. Direction supported (within $0.664$ vs cross $0.565$ on B-full); single-sided permutation $p = 0.243$, statistically inconclusive.

**H7' (new for v2): operational exploitation of cross-modality independence.**

A naive $2k$-signal DAJV that pools structural and executable signals does NOT improve the calibrated operating point at the default threshold $\tau = 0.95$.

- **Status:** CONFIRMED NEGATIVE (Appendix B.10 of the paper).
- **Future-work direction:** design an aggregator that exploits cross-modality independence at a per-problem level (e.g.\ gating per-LLM accept on cross-modality agreement). Pre-registered for follow-up.

## Pre-registered follow-up experiments

1. **Operational-exploitation experiment.** Design at least one
   aggregator that converts cross-modality independence into a
   calibrated gain over default DAJV without sacrificing precision.
   Candidates: (a) gated-acceptance variant requiring per-LLM
   cross-modality agreement; (b) hierarchical posterior with
   modality-specific latent factors; (c) executable-as-prior
   formulation. Success criterion: $\geq 5$pp coverage gain at the
   default operating threshold on math175, holding precision at
   $\geq 0.97$.
2. **Lab-rotation test with $k_{\text{within}} \geq 8$ per lab.**
   Requires user-side Together AI / Anthropic credit top-up. Pass
   criterion: single-sided permutation test $p < 0.05$ on
   $\kappa_{\text{within}} > \kappa_{\text{cross}}$.
3. **Faithful VERGE replication.** Source from Singh et al. 2026 or
   reimplementation with Z3 SMT + true MCS construction.
4. **Cross-distribution validation.** Run the DAJV pipeline on
   ProcessBench (gold-free mode) and AbstentionBench (via
   `inspect_evals` harness).

## Stop-do-not-touch invariants

The following are locked and may NOT be re-tuned without a `v3` pre-registration:

- Frozen extractor prompt (`verifyensemble/extractors/prompt.py`).
- Deployment-time adversarial probe set (`verifyensemble/sandbox/adversarial.py`).
- Sandbox timeout (8 seconds per probe).
- DAJV second-order copula form and prior estimator
  (`verifyensemble/aggregate/dajv.py`).
- Theorem 1 statement and DAJV bound formula.

Branch B per v1 remains active: in-distribution calibration only,
with per-benchmark recalibration as the deployment posture.

## Decision tree post-pivot

| Outcome | Action |
|---|---|
| H1' PASS + at least one aggregator exploits cross-modality independence (H7' converted) | Submit to NeurIPS 2026 main track |
| H1' PASS, H7' remains negative, H6' PARTIAL | Submit to AI for Math workshop @ NeurIPS 2026 + ICML 2027 main track with operational-exploitation as future work |
| H1' FAIL (cross-modality not actually independent on richer ensembles) | Treat as a negative result paper; submit to workshop |

## Files touched by the v2 pivot

- `paper/main.tex` (title, abstract, keywords)
- `paper/sections/introduction.tex`
- `paper/sections/empirical.tex` (cross-modality promoted to subsection 3)
- `paper/sections/discussion.tex` (tightened)
- `paper/sections/extended_results.tex` (seed-stability, solver-rotation, A3, LOO, joint-FP scatter moved here)
- `README.md`
- `CHANGELOG.md`
- this file
