# Fix Plan — MATHAI X-SGRV for NeurIPS 2026 (v1)

**Goal:** turn the paper from "borderline NeurIPS main track" (6.5/10) to "solid acceptance" (~8/10) by closing the critical reviewer gaps identified in the self-rating and the external literature sweep.

**Constraints:**
- Only Together API + user HPC cluster + local Mac. No paid GPU beyond what's already available.
- Must preserve every existing verified number.
- Must pre-flight any spend >$1 of API calls.

---

## Known issues being fixed

Numbered to tie back to the self-rating and literature sweep.

**Self-rating weaknesses:**
- W1. Sample sizes on hard benchmarks tiny (AIME 1/1, CleanMath 4/4).
- W2. MATH-175 not MATH-500.
- W3. Only one solver (Qwen2.5-7B); no solver rotation.
- W4. No PRM baseline.
- W5. No p(True) / Kadavath baseline.
- W6. Coverage story on hard benchmarks is thin (~3%).
- W7. Extractor cost story is one sentence.
- W8. Title is long/clunky.
- W9. Writing is dense; no narrative arc; 11 numbered results spread across 4 sections.
- W10. NLI variant of SE deferred to camera-ready.

**Literature sweep — critical:**
- L1. Qwen2.5-Math-PRM-7B not run (SOTA open PRM, single biggest gap).
- L2. Skywork-o1-Open-PRM-Qwen-2.5-7B not run (second open PRM, cross-family to Qwen-PRM).
- L3. NLI-clustering SE = canonical variant, deferring looks like cherry-picking.
- L4. PRIME / Eurus-2-7B-PRIME not run (implicit PRM, different training signal).

**Literature sweep — strong (cite or optional run):**
- L5. MathArena / Omni-MATH / MATH-Perturb — additional contamination-clean benchmarks.
- L6. rStar-Math, DeepSeekMath-V2, AceMath-PRM, Self-Taught Evaluator, Self-Refine, Self-Debug, ProcessBench, OlympiadBench, Semantic Entropy Probes — all missing citations.

**Paper review — additional issues I found reading the compiled paper:**
- P1. Abstract is 6 sentences of dense results; no headline.
- P2. Contamination finding (the paper's most important result) is buried in §8.
- P3. Zero figures. No pipeline diagram, no RC curves, no precision×coverage plots.
- P4. No code listing showing what a verifier actually looks like.
- P5. No error-case analysis (just aggregate numbers).
- P6. CleanMath per-competition breakdown not reported.
- P7. No explicit compute/cost table.
- P8. Reproducibility details scattered across sections.
- P9. Related work is 4 short paragraphs.
- P10. Results 1–11 are numbered globally but spread across sections — reviewers will lose the thread.

---

## Tiered execution

Prioritized by (reviewer impact / cost). Each item lists: what, cost, deliverable, risk, dependency.

### TIER 1 — FREE, CPU-ONLY, FULLY LOCAL
Execute immediately without approval. All items reuse existing JSONs or make zero new API calls.

**T1.1 Paper restructuring + writing**
- Shorten title: *"Cross-Family Symbolic Verification for Contamination-Robust LLM Math Reasoning"* (9 words, down from 14).
- Rewrite abstract: lead sentence = the headline result (consensus 100% on MATH, working coverage on contamination-clean), then method, then contamination caveat.
- Collapse "Our contributions" from 5 items to 3 (C1 contamination collapse + measurement; C2 X-SGRV method; C3 deployment-time filter + consensus).
- Move contamination story (currently §8) forward — mention the AIME collapse in the intro.
- Rename "Results 1–11" to named, non-numbered paragraph heads so each section stands alone.
- **Cost:** 0. **Deliverable:** updated main.tex. **Risk:** breaking references; mitigated by not renumbering cross-refs.

**T1.2 Figures (all generated from existing JSONs)**
- **Fig 1 — Pipeline diagram.** X-SGRV flow: problem → Llama / DeepSeek extractor (parallel) → verifier → adversarial probe (filter) → consensus → tier verdict. TikZ.
- **Fig 2 — RC curves.** MATH-500 and CleanMath: X-SGRV raw, X-SGRV+filter, consensus, SE-math, SE-NLI (after T1.8), p(True) (after T1.9), self-consistency. One panel per benchmark.
- **Fig 3 — Precision × coverage scatter.** Single figure, all methods × all benchmarks, log-axes.
- **Fig 4 — Per-level MATH-500 breakdown** (already exists in text; convert to bar chart).
- **Fig 5 — Adversarial FP histogram.** Distribution of which adversarial probe (gold±1, gold+7, gold×2, fixed) triggers rejection.
- **Cost:** 0. **Deliverable:** 5 PDFs in `paper/figures/`. **Risk:** latex compile breakage; mitigated by compile-after-each-figure.

**T1.3 Error-case analysis appendix**
- Manual inspection of the 5 MATH false positives caught by the filter.
- The 1 residual MATH FP caught only by consensus.
- The AIME geometry FP (aime2025_29).
- CleanMath: why plurality accuracy is only 12% (is it the model, the competition, or the cutoff?).
- **Cost:** 0. **Deliverable:** `paper/appendix_errors.tex` (1 page). **Risk:** low.

**T1.4 CleanMath per-competition breakdown**
- Split exp34_cleanmath_llama70b.json by competition (HMMT, BRUMO, SMT, APEX).
- Solver accuracy, coverage, tier precision, Adv-FP for each.
- **Cost:** 0. **Deliverable:** new subtable or text block in §9. **Risk:** low.

**T1.5 Representative code listing**
- Extract a real verifier script from exp31 for a MATH-500 problem (sanitize if verbose).
- Add as `\begin{lstlisting}` in §X-SGRV method.
- **Cost:** 0. **Deliverable:** one listing block. **Risk:** listings package conflict (handle first).

**T1.6 Reproducibility appendix**
- All seeds, temperatures, API version strings, prompts, adversarial probe values, sandbox timeouts, solver/extractor names + dates.
- **Cost:** 0. **Deliverable:** `paper/appendix_repro.tex` (2 pages). **Risk:** low.

**T1.7 Expanded related work + citations**
- Add to `references.bib`: rStar-Math, PRIME/Eurus-2, DeepSeekMath-V2, AceMath-PRM, Self-Taught Evaluator, Self-Refine, Self-Debug, ProcessBench, OlympiadBench, Omni-MATH, MATH-Perturb, MathArena, Semantic Entropy Probes.
- Expand §Related Work from 4 short paragraphs to 6: (1) PRMs, (2) process supervision, (3) selective prediction, (4) self-verification/self-correction, (5) contamination and contamination-clean eval, (6) formal verification.
- **Cost:** 0. **Deliverable:** updated references.bib + §Related Work. **Risk:** fake DOIs; mitigated by using canonical arXiv IDs only.

**T1.8 NLI-clustering SE baseline**
- Download microsoft/deberta-large-mnli (700MB) via transformers.
- Run phase 2 of `run_exp35_semantic_entropy.py` without `--skip-nli` — phase 1 samples already cached.
- Add `se_nli` column to Result 11 and Tables 3/4.
- Reposition SE-NLI as a first-class main-table baseline, not "camera-ready."
- **Cost:** 0 dollars, ~45 min CPU wall time. **Deliverable:** updated exp35 JSON with `se_nli` rows + paper updates. **Risk:** DeBERTa download fails → fall back to alternative HF-hosted checkpoint. **Dependency:** transformers + torch in Python env.

**T1.9 p(True) / Kadavath baseline**
- Prompt: "Question: {q}\nProposed answer: {a}\nIs this answer correct? Answer with only 'True' or 'False'."
- Call Qwen2.5-7B-Instruct-Turbo once per (problem × plurality candidate) = 330 calls.
- Use `logprobs=True` to get the logit of "True" vs "False" token as continuous score.
- Compute AUROC per benchmark, threshold-tier precision.
- Add to Tables 3/4 and Result 4.
- **Cost:** ~$0.50 (330 calls, ~500 tokens each). **Deliverable:** `experiments/run_exp36_ptrue.py` + `results/exp36_ptrue.json` + paper updates. **Risk:** Together API logprobs format edge cases; mitigated by fallback to sampling 10× and taking Yes-fraction.

**T1.10 Cost/compute expansion**
- New subsection or table: per-verification latency + $ cost for X-SGRV raw, X-SGRV consensus, SE-math, SE-NLI, p(True), SC (4 samples), SC (10 samples).
- Source: Together public pricing.
- **Cost:** 0. **Deliverable:** `tab:cost`. **Risk:** low.

**T1.11 Anonymization pass (NeurIPS double-blind)**
- Grep for author names, emails, institutions, repo URLs.
- Redact to "Anonymous" / "[REDACTED]".
- **Cost:** 0. **Deliverable:** verified anonymous main.tex. **Risk:** low but important.

### TIER 2 — MODERATE API SPEND ($20–60, APPROVE BEFORE EXECUTING)

**T2.1 Scale MATH-175 → MATH-500 (all extractors)** — *addresses W2*
- exp5 already has solver outputs for all 500 problems; we only need to run extractors on the missing 325.
- Llama-70B: 325 problems × 1 extractor call × 5 adversarial verifier runs (~2k tokens/call) ≈ $10.
- DeepSeek-V3: same ≈ $8.
- Recompute Tables 7, 8, consensus, RC curves on full scale.
- **Cost:** ~$18. **Deliverable:** updated exp31/32/33/consensus JSONs with n=500. **Risk:** Together API rate limits / hangs (we have the timeout fix from the prior session); mitigated by resumable cache.

**T2.2 Omni-MATH hard subset (50 problems)** — *addresses L5, W1*
- Grab 50 hardest post-cutoff problems from Omni-MATH.
- Solve with Qwen2.5-7B-Instruct-Turbo (10 samples each) ≈ $3.
- Run X-SGRV with both extractors ≈ $6.
- Report as an additional contamination-clean column in Tables 7/8.
- **Cost:** ~$10. **Deliverable:** `results/omnimath_hard50.json` + new results paragraph. **Risk:** Omni-MATH has some auto-grader quirks; mitigated by using SymPy equivalence only.

**T2.3 Solver rotation (optional in tier 2)** — *addresses W3*
- Current solver: Qwen2.5-7B-Instruct-Turbo only.
- Add Llama-3-8B-Lite as second solver (cheap, different family).
- Run on MATH-175 + AIME + CleanMath ≈ $15.
- Compare X-SGRV tier precision across solvers.
- **Cost:** ~$15. **Deliverable:** new Table 9 + one results paragraph. **Risk:** if tier precision drops dramatically, the paper story changes — but that's also the information we need.

**Tier 2 total:** ~$43 if all executed. Pre-flight review required before launching.

### TIER 3 — HPC / LOCAL GPU REQUIRED (CRITICAL — REVIEWER WILL DEMAND)

**T3.1 Qwen2.5-Math-PRM-7B baseline** — *addresses L1, W4*
- Download `Qwen/Qwen2.5-Math-PRM-7B` (7B, ~14GB FP16).
- Inference environment options (ranked):
  1. HPC cluster (have access) — SSH, run transformers inference on cached samples.
  2. Modal (user has skill) — rent a single A100 for ~$2/hr × 2hrs = $4.
  3. Mac local with MLX (7B Q4 is ~4GB, slow but works).
- Scoring protocol: split each of 330 × 10 = 3,300 samples into steps by `\n\n`; score each step via the `<extra_0>` token logit; take min step score per sample as the sample confidence; take max across the 10 samples per problem as the problem-level PRM score; threshold to define a tier.
- Add a 3-row PRM comparison table (Table 9 or merged into Table 3/4).
- **Cost:** ~$0 (HPC) or ~$4 (Modal). **Deliverable:** `results/exp37_qwen_prm.json` + paper updates. **Risk:** HPC connectivity; mitigated by Modal fallback.

**T3.2 Skywork-o1-Open-PRM-Qwen-2.5-7B baseline** — *addresses L2, W4*
- Same pipeline as T3.1 with `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`.
- Inference is `Skywork/skywork-o1-prm-inference` repo style (their scoring protocol differs slightly from Qwen's).
- **Cost:** ~$0 (HPC) or ~$4 (Modal). **Deliverable:** `results/exp38_skywork_prm.json` + paper updates.

**T3.3 PRIME / Eurus-2-7B-PRIME (optional)** — *addresses L4*
- Implicit PRM. Runs via the PRIME repo.
- If T3.1 and T3.2 tell a coherent story, T3.3 is additive but not essential.
- **Cost:** ~$4 Modal. **Deliverable:** `results/exp39_prime_prm.json`.

**T3.4 Solver rotation with Llama-70B as solver (optional)** — *addresses W3*
- Use Together API (same as solver rotation in T2.3 but with Llama-70B).
- Cost: ~$25. More expensive because Llama-70B is pricier per token.
- Deliverable: Table showing X-SGRV tier precision when solver family changes.

**Tier 3 total:** ~$8–40 + HPC time. Pre-flight review mandatory.

---

## Execution order

1. T1.1–T1.11 (free). Sequential. ~3–4 hrs of tool calls.
2. Recompile paper after each T1 item; commit locally per item.
3. Pre-flight review of Tier 2 batch → user approval → execute T2.1, T2.2, T2.3.
4. Pre-flight review of Tier 3 batch → user approval → execute T3.1, T3.2 (critical), T3.3/3.4 optional.
5. Final numbers self-check (Opus agent) on completed paper.
6. Final compile + anonymization verification.

## Risk register

- **R1. Writing quality regression.** Rewriting the abstract / restructuring results risks introducing errors. Mitigation: keep old main.tex backed up; spot-check all numbers after rewrites; run Opus self-check before compile.
- **R2. LaTeX compile failures.** New packages (listings, TikZ) can fight with tectonic's bundle. Mitigation: add one at a time; compile after each.
- **R3. API cost overrun.** Tier 2 could blow past $60 if rate limits cause retries. Mitigation: cache every call; set hard per-call budget; pre-flight agent verifies logic before launch.
- **R4. PRM scoring protocol mismatch.** PRM evaluators vary by paper (min-step, product, last-step). Mitigation: report all three; note the choice; use what the PRM card recommends.
- **R5. Story erosion.** If PRMs beat X-SGRV, the paper's headline shifts. Mitigation: honest reporting; X-SGRV's orthogonal contribution (structural independence, deployment-time filter) still stands.

## What I am NOT doing

- Training any new model (no PRM fine-tuning, no custom verifier training).
- Running Cell B of the 2×2 (solution-grounded + randomized) — still marked analytic per original limitations.
- Adding new theoretical claims — this is an empirical paper.
- Extending the Vault or memory system in this session — out of scope.
