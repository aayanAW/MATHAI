# DAJV Novelty Verification Report

**Date:** 2026-05-25
**Author:** A. Alwani
**Trigger:** Pivot from X-SGRV → DAJV (mandatory novelty-verification protocol).

## Headline assessment

**Novelty score: 3-4 / 10.** Below the "5-6 = domain port" floor.

Three concurrent 2026 arXiv papers cover the central claims:

| Concurrent paper | arXiv | Overlap |
|---|---|---|
| Kuai et al. 2026 "How Independent are LLMs?" | [2604.07650](https://arxiv.org/abs/2604.07650) | Derives joint-error bound as function of pairwise Cohen's κ and proposes verifier reweighting. **Near-identical scope to DAJV.** |
| Zhao et al. 2026 "CARE: Confounder-Aware Aggregation" | [2603.00039](https://arxiv.org/abs/2603.00039) | Models LLM judge scores with shared confounders; finite-sample identifiability bound. **Direct theoretical predecessor.** |
| Balasubramanian, Podkopaev, Kasiviswanathan 2026 "Dependence-Aware Label Aggregation via Ising Models" | [2601.22336](https://arxiv.org/abs/2601.22336) | Bayes-optimal aggregation under Ising-model dependence. **Strictly more general than DAJV's second-order copula.** |

All three were verified to exist on arXiv. Submission dates: Jan 29, March 1, April 8 (2026). All three pre-date this DAJV manuscript.

Additional close work:

- Lefort et al. 2024 "Condorcet Jury Theorem for LLMs" ([2409.00094](https://arxiv.org/abs/2409.00094)) already empirically demonstrated independence violation.
- Classical Berend & Sapir (2007), Boland (1989), Ladha (1992) gave Condorcet-with-dependence bounds dating to the 1990s — DAJV Theorem 1 is in their lineage.
- Kim, Garg, Peng, Garg 2025 "Correlated Errors in LLMs" (ICML, [2506.07962](https://arxiv.org/abs/2506.07962)) — already cited; the empirical "independence is violated" finding is not novel.

## Scoop-risk verdict

**Critical (>90% probability core claims are already scooped).**

Active groups with all building blocks already published:

- Frederic Sala's lab @ UW-Madison (CARE).
- Texas A&M + UT (Kuai).
- UC Davis + AWS (Balasubramanian).

## Sharpening recommendations (must reach ≥ 7 to proceed as currently scoped)

The honest path forward is to pivot the framing, not the codebase. The library, theorems, calibration, and empirical pipeline are intact and reusable. What needs to change:

### Option A (recommended): executable + dependency-aware hybrid

Fuse the existing **ExeVer** executable verification signal (see `MEMORY.md` → 84% accuracy on cached results) with DAJV's dependency-aware aggregation. Neither Kuai nor CARE nor Ising-aggregation papers use program execution as a verifier modality. The pitch:

> "When verifiers can *run code*, what is the right dependency-aware aggregation rule? We show that executable signal partially decorrelates jurors (lowering observed κ by X), and that the residual dependency on the *non-executable* abstention layer is what determines selective-prediction risk."

This is a defensible moat. Probability of independent scoop within 12 months: moderate but not yet realized.

### Option B: adversarial dependency

Treat the LLM jurors as potentially compromised — e.g. a subset is jailbroken, prompt-injected, or trained on the same web-scraped contamination batch. Derive robustness bounds against an adversary that selects a subset of correlated jurors. **No existing work covers this.** Niche, but novel.

### Option C: sharpen Theorem 2

If the DAJV calibration's sample-complexity bound is strictly tighter than CARE's in any regime (e.g. binary verifier votes vs continuous LLM judge scores), state the comparison explicitly. Currently DAJV claims a bound but does not show it is tighter than CARE.

### What to drop

- **Drop Theorem 1 as headline.** Demote to "specialization of Ising-aggregation bound to binary verifiers in an appendix."
- **Drop the 15.7× joint-FP claim as a contribution.** Lefort 2024 and Kim 2025 already showed independence-is-broken.
- **Drop the selective-prediction calibrator as a contribution unless** the executable-signal axis (Option A) gives it a defensible angle.

## Recommended next steps

1. **Do not submit DAJV in current form.** Anticipated reviewer behavior: cite Kuai/CARE/Balasubramanian as prior work, mark as incremental.
2. **Commit to Option A pivot** (ExeVer × DAJV hybrid). Re-scope theorems and experiments to put the executable signal in the headline.
3. **Cite all three concurrent works prominently in §2 (Related Work) of the paper.** Reviewers will check for this; omission would be perceived as dishonest.
4. **Keep the library + extractor cache + theory implementations.** They are reusable for Option A.
5. **Update PRE_REGISTRATION_v1.md → v2** before re-running any experiments, with the new framing locked in.

## Files updated this session per this report

- `paper/references.bib` — added correct arXiv IDs and Balasubramanian entry.
- `paper/sections/related_work.tex` — added concurrent-work block.
- `paper/sections/discussion.tex` — added honest scoop acknowledgment.
- This file (`NOVELTY_REPORT_2026-05-25.md`).

## Caveat

The arXiv IDs above were verified by fetching the arXiv abstract pages directly, not from memory. The literature search was checked against primary sources; spot-checks confirmed the three flagship "scoop" papers exist as described.
