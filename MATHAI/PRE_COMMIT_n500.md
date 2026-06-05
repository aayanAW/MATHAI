# Pre-registration: MATH-500 scale-up + PRM baselines + Cell B

**Date:** 2026-04-15
**Author:** authors of the MATHAI X-SGRV manuscript
**Status:** Pre-committed before Tier 2 and Tier 3 experiments launch.

This document exists because the final reviewer pass on `FIX_PLAN_v2.md` noted that pre-committing to a headline number is only a real research practice if the commitment is in-repo and timestamped before the experiments run. This file is that commitment.

## What will be reported

The paper's primary MATH-500 results table (Table 7) will be updated from the current stratified $n{=}175$ subsample to the full $n{=}500$ benchmark, with both Llama-3.3-70B and DeepSeek-V3 extractors and their consensus. Whatever the resulting consensus precision is, we report it — no cherry-picking.

## Exact metrics that will be reported at $n{=}500$

For each (extractor, mechanism) pair — Llama-70B raw, Llama-70B + filter, DeepSeek-V3 raw, DeepSeek-V3 + filter, Llama $\cap$ DeepSeek consensus strict, Llama $\cup$ DeepSeek loose — we commit to reporting:

- Top-tier size (count of problems receiving a top-tier accept)
- Coverage (top-tier size / 500)
- Top-tier precision with Clopper-Pearson 95% CI
- Adversarial FP rate (over non-None adversarial verdicts)
- Per-level (L1–L5) precision breakdown
- McNemar p-value vs. Semantic Entropy at matched MATH-500 coverage
- Raw X-SGRV precision before filter (already reported in Table 8 "No filter" row)

All six rows will appear in the main table. No mechanism will be omitted on the basis of poor performance. If Llama-70B raw at $n{=}500$ has higher precision than consensus, we will report that and adjust the narrative.

## Three narrative branches for the consensus result

We commit in advance to one of the following three narrative framings, determined by the numeric outcome:

### Branch A — consensus precision $\ge 99\%$
**Abstract sentence:** "On MATH-500 ($n{=}500$), cross-extractor consensus achieves near-perfect top-tier precision (X/Y = [NUM]\% [CI]) at [NUM]\% coverage."
**Framing:** consensus mechanism is near-perfect at substantial coverage; robustness story holds; contamination-clean caveat remains.

### Branch B — consensus precision 95\%–99\%
**Abstract sentence:** "On MATH-500 ($n{=}500$), cross-extractor consensus achieves [NUM]\% top-tier precision at [NUM]\% coverage, with the small residual error rate attributable to a mix of verifier errors and known MATH-500 autograder artifacts (Appendix~\ref{app:errors})."
**Framing:** near-perfect at high coverage; residual errors exist but are partly autograder artifacts; still the strongest available operating point on contamination-assisted MATH-500.

### Branch C — consensus precision $< 95\%$
**Abstract sentence:** "On MATH-500 ($n{=}500$), cross-extractor consensus achieves [NUM]\% top-tier precision at [NUM]\% coverage; the deployment-time adversarial filter remains the primary contribution, delivering [filter-NUM]\% precision at [filter-COV]\% coverage with no gold-label access."
**Framing:** consensus mechanism is not the headline; the deployment-time filter becomes the primary method contribution; paper is repositioned around the filter's precision and the contamination story.

**No Branch D** — if the $n{=}500$ consensus number is below $90\%$, we halt, audit extractor outputs for systematic errors (e.g., prompt changes, rate-limit corruption), and do not submit.

## PRM baseline pre-commitment (Tier 3)

Qwen2.5-Math-PRM-7B \citep{qwenprm2025} and Skywork-o1-Open-PRM-Qwen-2.5-7B \citep{skyworkprm2024} will be run on the cached 10-sample outputs from exp35 across MATH-500 $n{=}500$, AIME 2025 $n{=}30$, and CleanMath $n{=}125$. Protocol:

- **Qwen-PRM:** model card's chat template with `<extra_0>` token between steps; extract per-step positive-class probability; report min-step, product-step, and last-step aggregations.
- **Skywork-PRM:** inference repo's recommended average-across-steps aggregation as primary; also report min-step and last-step for symmetry.
- **Sanity check (ProcessBench):** before trusting our numbers, reproduce each PRM's published ProcessBench ErrorACC on a 50-problem random subset; if our reproduction deviates by more than 5 percentage points, we halt and debug the scoring protocol.
- **Matched coverage comparison:** threshold each PRM at X-SGRV's coverage on each benchmark; report precision at matched coverage.
- **All three aggregations reported in the same table** — no cherry-picking.

If PRMs match or exceed X-SGRV on MATH-500 at matched coverage, we report it and reframe X-SGRV as "orthogonal to learned PRMs, deployable without training data." If PRMs collapse on contamination-clean benchmarks (as we hypothesize, because PRMs are trained on MATH-family data), we report that too.

## Cell B empirical pre-commitment (Tier 3)

Cell B of the $2\times 2$ independence $\times$ randomization ablation (solution-grounded randomized ExeVer) will be measured empirically on $n{=}120$ MATH-500 problems. Power: expected false-acceptance rate of $13.8\%$ at $n{=}120$ yields a $\pm 6$pp 95\% CI, distinguishing the null (randomization does not help) from a halving effect.

- **If Cell B FAR is within $\pm 2$pp of Cell A:** randomization confirmed irrelevant, independence confirmed operative, current "independence is the operative factor" framing survives and the 2x2 ablation becomes fully empirical.
- **If Cell B FAR is $\ge 5$pp below Cell A:** randomization helps non-trivially, "independence is operative" claim is downgraded to "independence is dominant but randomization contributes"; the abstract's Fisher exact framing is adjusted.
- **If Cell B FAR is $< 2\%$:** randomization alone is sufficient to eliminate collapse, independence claim is dropped entirely, and the paper is repositioned around randomization as the remedy. (We assign low prior probability to this outcome based on prior literature, but we commit to reporting it honestly.)

## What we will NOT do

- No post-hoc reclassification of false positives as "autograder artifacts" in the main-table numbers. The appendix error analysis is separate and labeled.
- No choice of PRM aggregation based on which aggregation is best for X-SGRV.
- No choice of PRM scoring window based on which gives the most favorable comparison.
- No addition of further probe values beyond the held-out-selected set $\{\pm 1, +7, \times 2, 42\}$.
- No re-tuning of the extractor prompt mid-experiment.
- No re-rolling of the 30-problem held-out dev split if the chosen probe set happens to underperform on test.

## Commit reference

This file will be committed to the repository at `MATHAI/PRE_COMMIT_n500.md` with a timestamped git commit \emph{before} the Tier 2 scale-up runs. The commit hash is the pre-registration hash.
