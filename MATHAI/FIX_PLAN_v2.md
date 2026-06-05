# Fix Plan — MATHAI X-SGRV for NeurIPS 2026 (v2)

**Changes from v1:** address both reviewer passes. Generic reviewer raised missing items (M1–M6, PRM protocol wrong, reordering). Adversarial reviewer flagged: coverage story still thin, CleanMath CI fragile, need pre-commit to n=500 as primary, Cell B structural hole, arbitrary probe set, PRM aggregation landmine. v2 upgrades the plan to address both.

**Headline pre-commits (before any experiments run):**
1. **MATH-500 full n=500** will be the **primary** headline, not n=175. n=175 moves to "stratified confirmation."
2. **Consensus 100% claim is dropped if even 1 false positive appears at n=500.** We report whatever the number actually is — `X/X` or `X/(X+k)`.
3. **CleanMath CI is moved to the abstract.** The "100% precision" claim becomes "100% precision (4/4, CI [0.40, 1.00], n small)" in the abstract itself — the honesty buys trust.
4. **Probe set is held out.** We split a 30-problem dev set, select probes once, freeze them, and report test-set filter effectiveness. The ±1/+7/×2/42 set from v1 is treated as the candidate set, not pre-validated truth.
5. **If PRMs beat X-SGRV at matched coverage, we report it.** Full honesty — no cherry-picked aggregation.

---

## All issues being fixed (renumbered)

### From self-rating (W)
- W1–W10 as in v1.

### From literature sweep (L)
- L1–L6 as in v1.

### From paper re-read (P)
- P1–P10 as in v1.

### From generic reviewer (M — **NEW in v2**)
- **M1. Contamination decomposition.** Split MATH-175 (and later n=500) by Wu et al contamination labels (memorized vs non-memorized). Report X-SGRV and SE separately on each subset. Free.
- **M2. Statistical significance test.** McNemar on X-SGRV vs SE at matched coverage on MATH-175; bootstrap AUROC difference. Free.
- **M3. Reorder: scale to n=500 BEFORE running PRMs.** Don't waste PRM compute on n=175.
- **M4. Trivial-verifier audit.** What fraction of Llama/DeepSeek verifiers are syntactically trivial (e.g., `return answer == <literal>`)? A trivial verifier trivially passes adversarial filter. This is the "is the extractor just solving the problem" sanity check. Free.
- **M5. Seed sensitivity on consensus.** Re-run both extractors at T=0.3 on a random 30 problems; report variance in consensus verdicts. Small API cost.
- **M6. Anonymization + de-anonymization check (via citation / arxiv).** Make sure `exever` and other self-cites don't de-anonymize.

### From adversarial reviewer (A — **NEW in v2**)
- **A1. Coverage story honesty.** Abstract + intro must say the operating point (3.2% CleanMath coverage) without hiding it in §Limitations. Reframe X-SGRV as "high-precision abstention signal," not "selective prediction replacement."
- **A2. CleanMath CI in abstract.** See pre-commit 3.
- **A3. n=500 consensus pre-commit.** See pre-commit 1. Don't run PRM or anything else until we know the n=500 consensus number.
- **A4. PRM protocol correctness.** See T3.1/T3.2 fixes below.
- **A5. PRM matched-coverage comparison.** Threshold PRM at X-SGRV's coverage (e.g., 41.7% for MATH-500 consensus) and compare precision directly. Not just AUROC.
- **A6. Cell B structural hole.** Two options: (a) measure Cell B empirically (solution-grounded + randomized ExeVer on 320 problems, ~$8 API); (b) drop the "independence is the operative mechanism" claim from the abstract and downgrade to "independence is the remedy that works, randomization alone does not." **v2 picks (a)** — measuring is cheap and closes the hole.
- **A7. Held-out probe selection.** See T1.12 below.
- **A8. Novelty reframing.** In intro, explicitly position X-SGRV as: (i) the first selective-prediction method to produce non-zero high-precision coverage on a contamination-clean math benchmark; (ii) deployment-time gold-free adversarial filter is novel on its own; (iii) cross-family consensus at matched scale. Not "just engineering."

---

## Tiered execution (v2, reordered)

### TIER 1a — FREE WRITING + STRUCTURE (do first, locally, ~4 hrs)

**T1.1 Paper restructuring + writing** *(W8, W9, P1, P2, P10, A1)*
- New title: **"Cross-Family Symbolic Verification for Contamination-Robust Selective Prediction on LLM Math"** (11 words, down from 14).
- Abstract rewrite: honest headline. Structure: (1) problem (correlated-error collapse); (2) contamination finding (AIME 2025 collapse); (3) X-SGRV method + adversarial filter + consensus; (4) results with explicit coverage%, CIs, and the CleanMath CI disclosure; (5) limitations in the abstract ("operating point is narrow on contamination-clean benchmarks"); (6) one-sentence contribution. ~300 words.
- Intro rewrite: open with contamination collapse, then X-SGRV as the remedy. Move §7 (contamination) material forward.
- Contributions list: collapse to 3 — (C1) contamination collapse measurement + replication, (C2) X-SGRV method (cross-family extractor + adversarial filter + consensus) with n=500 MATH + AIME + CleanMath, (C3) deployment-time gold-free adversarial filter as an independent contribution.
- Rename all "Result N" paragraphs to named headings.
- **Cost:** 0. **Deliverable:** updated main.tex. **Risk:** cross-ref breakage → compile after each change.

**T1.2 Figures (TikZ + matplotlib)** *(P3)*
- Fig 1 — X-SGRV pipeline diagram.
- Fig 2 — RC curves per benchmark.
- Fig 3 — Precision × coverage scatter.
- Fig 4 — Per-level MATH-500 breakdown.
- Fig 5 — Adversarial FP histogram.
- Fig 6 — Contamination decomposition bar chart (memorized vs non-memorized precision per method).
- **Cost:** 0. **Deliverable:** 6 PDFs in `paper/figures/`.

**T1.5 Representative code listing** *(P4)*
- One real verifier script from a MATH problem, one from an AIME problem.
- LaTeX `listings` package.
- **Cost:** 0.

**T1.6 Reproducibility appendix** *(P8)*
- One-page appendix: seeds, temperatures, prompts, probe values, API versions, sandbox timeouts, model names + dates.
- **Cost:** 0.

**T1.7 Expanded related work + citations** *(L6, P9)*
- Add to `references.bib`: rStar-Math, PRIME/Eurus-2, DeepSeekMath-V2, AceMath-PRM, Self-Taught Evaluator, Self-Refine, Self-Debug, ProcessBench, OlympiadBench, Omni-MATH, MATH-Perturb, MathArena, Semantic Entropy Probes, Qwen2.5-Math-PRM, Skywork-o1-PRM.
- Expand §Related Work to 6 subsections: PRMs, process supervision, selective prediction, self-verification/self-correction, contamination, formal verification.
- **Cost:** 0.

**T1.11 Anonymization pass** *(M6)*
- Grep for author names, emails, repo URLs, uncommon phrasings that could de-anonymize.
- Check that exever cite doesn't give away authorship (it's cited as 3rd party, which is correct if ExeVer was a prior paper).
- Verify `\author{Anonymous}` is in place.
- **Cost:** 0.

**Tier 1a subtotal: 0 dollars, ~4 hrs wall time.**

### TIER 1b — FREE-ISH ANALYSIS + BASELINES (do second, ~3 hrs + small API spend)

**T1.8 NLI-clustering SE baseline** *(L3, W10)*
- Download microsoft/deberta-large-mnli.
- Run phase 2 of exp35 without `--skip-nli` — samples cached.
- Add `se_nli` column to Tables 3/4/new SE-only table.
- Reposition NLI-SE as main-table baseline, not camera-ready.
- **Cost:** 0 dollars, ~45 min CPU.

**T1.9 p(True) / Kadavath baseline** *(W5)*
- Prompt Qwen-7B-Turbo: "Is this answer correct? Answer only True or False." Use `logprobs=True` to get token probabilities.
- 330 calls × ~500 tok = ~$0.10 (reviewer corrected v1's $0.50 estimate).
- Add to Tables 3/4.
- **Cost:** ~$0.10.

**T1.3 Error-case analysis** *(P5)*
- Manual inspection of 5 MATH filter-caught FPs, 1 residual consensus-caught FP, 1 AIME geometry FP, and why CleanMath plurality accuracy = 12%.
- **Cost:** 0.

**T1.4 CleanMath per-competition breakdown** *(P6)*
- HMMT/BRUMO/SMT/APEX separately from exp34 JSON.
- **Cost:** 0.

**T1.10 Cost/compute table** *(W7, P7)*
- Per-verification $ cost + latency for X-SGRV raw/consensus, SE-math, SE-NLI, p(True), SC, PRMs.
- **Cost:** 0.

**M1 Contamination decomposition** *(new, critical)*
- Use Wu et al's contamination labels (from wu2025contam) to split MATH-175 into memorized vs non-memorized.
- Recompute X-SGRV and SE precision/coverage on each subset.
- Report in Table 10 + Figure 6.
- **Cost:** 0 if labels are public. If not, flag as future work.
- **Risk:** wu2025contam may not release per-problem labels publicly. Check before committing.

**M2 McNemar / bootstrap significance test** *(new, critical)*
- McNemar on {X-SGRV correct, SE correct} pairs at matched coverage.
- Bootstrap CI on AUROC difference.
- One sentence in Result 11 + appendix table.
- **Cost:** 0.

**M4 Trivial-verifier audit** *(new, critical)*
- Static analysis of all Llama-70B and DeepSeek-V3 verifier scripts.
- Classify as: (a) trivial literal equality (`return answer == <literal>`), (b) simple SymPy equivalence, (c) enumeration/constraint check, (d) compound logic.
- Report fraction per benchmark.
- Expect: most CleanMath passes are compound, most MATH passes are SymPy equivalence.
- **Cost:** 0.

**Tier 1b subtotal: ~$0.10, ~3 hrs wall time.**

### TIER 1.5 — PROBE SELECTION + SEED SENSITIVITY (~$5, mandatory before Tier 3)

**T1.12 Held-out probe selection** *(A7)*
- Split a held-out 30-problem dev set from MATH-500 (non-overlapping with n=175).
- Candidate probe sets: {±1}, {±1,+7}, {±1,+7,×2}, {±1,+7,×2,42}, {×2,+7}, {random ±k for k∈[1,10]}.
- Run deployment-time filter with each set on dev; pick the set that maximizes precision−miss_rate.
- Freeze. Report test-set filter effectiveness using the frozen set.
- **Cost:** ~$5 (dev set × 6 probe configurations × extractor calls).

**M5 Seed sensitivity on consensus** *(new, critical)*
- Re-run Llama-70B and DeepSeek-V3 extractors at T=0.3 on a random 30 MATH problems.
- Report variance in consensus verdicts across seeds.
- One paragraph in §Discussion.
- **Cost:** ~$2.

**Tier 1.5 subtotal: ~$7, sequential after Tier 1a/b.**

### TIER 2 — SCALE-UP (~$25–40, APPROVAL REQUIRED)

**T2.0 MATH-175 → MATH-500 (PROMOTED — must come before PRMs)** *(W2, M3, A3)*
- Reuse solver samples from exp5_math500_full.json.
- Run Llama-70B extractor on remaining 325 problems (~$10).
- Run DeepSeek-V3 extractor on remaining 325 problems (~$8).
- Re-run deployment-time filter (exp33 logic) on all 500.
- Recompute Tables 7/8 + consensus + all RC curves at n=500.
- **This is now the primary MATH result.** n=175 stays for cross-consistency but is no longer headline.
- **Cost:** ~$20 (reviewer-corrected range $15–25).
- **Pre-commit:** whatever the n=500 consensus number is, that's what we report.

**T2.1 Omni-MATH hard-100** *(L5, W1)*
- 100 hardest post-Qwen-cutoff problems from Omni-MATH.
- Solver: Qwen2.5-7B-Instruct-Turbo (10 samples each) ~ $5.
- X-SGRV with both extractors ~ $12.
- New contamination-clean column.
- **Cost:** ~$17.

**Tier 2 subtotal: ~$37.**

### TIER 3 — PRMs + CELL B (~$10–20 + HPC, APPROVAL REQUIRED)

**T3.1 Qwen2.5-Math-PRM-7B baseline — CORRECTED PROTOCOL** *(L1, W4, A4, A5)*
- Model: `Qwen/Qwen2.5-Math-PRM-7B` (FP16 ~14GB).
- **Scoring protocol (corrected from v1):**
  - Use the model card's chat template: system + user (problem) + assistant (solution broken into steps joined by `<extra_0>` token).
  - Forward pass returns per-`<extra_0>`-position logits; extract the 2-class head's positive-class probability.
  - **Report THREE aggregations** (per reviewer feedback): min step-score, product of step-scores, and last-step score.
  - Sample-level score = chosen aggregation; problem-level score = plurality-sample score.
- Compute AUROC per benchmark.
- **Matched-coverage comparison:** threshold PRM at X-SGRV's coverage (41.7% MATH, 3.3% AIME, 3.2% CleanMath); report precision directly. This is the fair apples-to-apples comparison.
- Infrastructure: HPC via SSH (user has HPC). Fallback: Modal A100 ~$4.
- **Cost:** ~$0 (HPC) or ~$4 (Modal).
- **Deliverable:** `results/exp37_qwen_prm.json` with all three aggregations + matched-coverage table.

**T3.2 Skywork-o1-Open-PRM-Qwen-2.5-7B — CORRECTED PROTOCOL** *(L2, W4, A4, A5)*
- Model: `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`.
- **Scoring protocol (corrected):**
  - Skywork's inference repo `SkyworkAI/skywork-o1-prm-inference` uses **average-across-steps** as the primary aggregation for Best-of-N.
  - Report primary (average) + min + last for symmetry with T3.1.
- Same matched-coverage comparison.
- **Cost:** ~$0 HPC or ~$4 Modal.

**T3.3 PRIME / Eurus-2-7B-PRIME (optional)** *(L4)*
- Implicit PRM via PRIME repo.
- Run only if T3.1 and T3.2 are consistent.
- **Cost:** ~$4.

**T3.4 Cell B empirical — NEW in v2** *(A6)*
- Goal: close the structural hole in the 2×2 (independence × randomization) ablation.
- Protocol: re-run ExeVer (solution-grounded) on a 50-problem MATH subset with a randomized prompt variation — ask Pass 2 to generate 5 random assertion checks instead of one, insert 2-3 random check orderings, randomize variable naming.
- Compare FAR vs Cell A (solution-grounded deterministic).
- **Expected result:** FAR roughly unchanged (randomization alone doesn't break correlation).
- **Cost:** ~$5 API (50 problems × 2 passes).
- **Deliverable:** 2×2 table is now fully empirical, Fisher exact test valid.

**Tier 3 subtotal: ~$13 Modal/API + HPC time.**

---

## Execution order (v2 — clean)

```
Phase 1: Tier 1a (writing/figures/repro/related work)           [free, 4h]
Phase 2: Tier 1b (SE-NLI + p(True) + M1/M2/M4 analyses)         [$0.10, 3h]
  → Recompile. Spot-check numbers.
Phase 3: Pre-flight review of Tier 1.5 + Tier 2 + Tier 3 batches [free, 10min]
Phase 4: Get user approval for Tier 1.5 + Tier 2 + Tier 3
Phase 5: Tier 1.5 (probe selection + seed sensitivity)          [$7, 1h]
Phase 6: Tier 2 (MATH-500 scale + Omni-MATH)                    [$37, 2h wall]
  → Pre-commit to whatever consensus number comes out.
Phase 7: Tier 3 (Qwen-PRM + Skywork-PRM + Cell B)                [$13, 3h wall]
  → Pre-commit to whatever matched-coverage numbers come out.
Phase 8: Final numbers self-check on completed paper            [free, 15min]
Phase 9: Final compile + anonymization verification             [free, 10min]
```

## Risk register (v2)

- R1. Writing regression. Mitigation: backup main.tex → main.tex.v10.bak before any edit; compile after each section; spot-check every number.
- R2. LaTeX compile failures from new packages. Mitigation: add one at a time.
- R3. API cost overrun. Hard budget $60. Cache every call. Pre-flight before launch.
- R4. PRM protocol mismatch. Mitigation: report all three aggregations; cite model card exactly.
- R5. Story erosion at n=500. Mitigation: pre-committed to honest reporting; X-SGRV's orthogonal contributions (deployment filter, consensus, cross-family independence) still stand even if precision is 98% instead of 100%.
- R6. Omni-MATH auto-grader drift. Mitigation: SymPy equivalence only, no LLM judge.
- R7. Wu et al contamination labels may not be public. Mitigation: check; fallback to "future work" if unavailable.
- R8. Cell B measurement may show randomization DOES help (contrary to current claim). Mitigation: report honestly; if so, the paper's independence claim gets downgraded from "operative" to "dominant" and the story still holds.
- R9. HPC SSH access may not be set up in-session. Mitigation: Modal fallback at $4–8 total.

## v3 Addendum — Final Reviewer Fixes (applied before execution)

**Fix 1: Abstract-rewrite timing contradiction.** Split T1.1 into:
- **T1.1a** (Phase 1, Tier 1a): Structure, contributions list, named headings, intro reorder, figures, related work, anonymization. Abstract gets a *skeleton* with `[NUM]` placeholders for anything that will change at n=500.
- **T1.1b** (Phase 7, after Tier 2 and Tier 3): Fill in final n=500 / PRM / Cell B numbers. Final abstract rewrite.

**Fix 2: Cell B statistical power.** Upgrade T3.4 from n=50 to **n=120** MATH problems. Expected API cost: ~$12 (up from $5). Justification: FAR~14% with n=50 has ±10pp CI, inconclusive; n=120 gives ±6pp which distinguishes null from halving. $7 extra is negligible against $60 budget.

**Fix 3: Git-committed pre-registration.** Before Tier 2 launches, commit a dated `PRE_COMMIT_n500.md` to git containing:
- Exact metrics to be reported (Table 7/8 column names and formulas).
- Three narrative branches for the n=500 consensus number:
  - **≥99%:** headline stays "essentially perfect at 41.7% coverage."
  - **95–99%:** headline becomes "near-perfect at high coverage; residual errors caught by filter."
  - **<95%:** headline shifts to "deployment-time adversarial filter as primary contribution; consensus as defense-in-depth."
- Which abstract sentence gets replaced in each branch.

This is the mechanism that makes pre-commit #1–#3 real research practice rather than intention.

**Fix 4: Wu et al label availability — check NOW.** Add a step 0 to Tier 1a: WebFetch `wu2025contam` paper + its supplementary materials. If per-problem labels are public, M1/Figure 6 proceed. If not, M1 is dropped from the plan before execution and Figure 6 is replaced by a Level-conditional precision plot (which we already have data for).

**Fix 5: Two free additions to strengthen PRM comparison.**
- **T1.10b. Filter-with-no-probes ablation row** in Table 8. Run exp33-logic with the probe set empty — i.e., raw X-SGRV with no filter. Already covered by existing JSONs (exp31/32 = raw X-SGRV). Just add the row explicitly for clarity. Zero new compute.
- **T3.1b. ProcessBench sanity check paragraph.** Before reporting Qwen-PRM and Skywork-PRM on our benchmarks, confirm both reproduce published ProcessBench numbers on a small random 50 ProcessBench sample. If they don't, scoring protocol is wrong and we catch it before Tables 9 are built. ~$0 (inference only, local HPC).

**All five fixes are now part of the executable plan. No further revisions needed.**

---

## What I am NOT doing (v2)

- Training any model.
- Running MATH-500 in streaming / online mode — reuse cached.
- Extending vault / memory system in this session.
- Rebuilding exp numbers 1–34 from scratch — strictly additive.
