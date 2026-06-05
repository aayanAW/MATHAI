# MATHAI Architecture Document v10.0 (X-SGRV scale-up)

**Last updated**: 2026-04-14
**Paper title**: "Cross-Family LLM-Extracted Symbolic Verification for Contamination-Robust Selective Prediction on LLM Math Reasoning"
**Core claim**: Same-model self-verification suffers from correlated-error collapse; a cross-family LLM extractor (Llama-3.3-70B and/or DeepSeek-V3) that emits a SymPy verifier script from the problem statement provides a reliable high-precision selective prediction signal, and requiring *consensus* between two independent cross-family extractors drives residual false positives to zero at the cost of ~25% of coverage.
**Target venues**: NeurIPS 2026 main track (primary), ICML AI4Math Workshop (backup), Nature Machine Intelligence (stretch)
**Compute budget**: ~$900 total, ~$35 spent (v9 + v10 scale-up)
**Status**: v8 mechanism + selective prediction complete; v9 X-SGRV pivot complete; v10 scale-up complete; v10 paper compiled (160 KB). Semantic Entropy baseline still running.

## v10 KEY RESULTS (measured)

**Contamination finding** — MATH-500 is measured contaminated for Qwen2.5-Math-7B (54.6% verbatim memorization per Wu et al. 2025, arXiv:2507.10532). On AIME 2025 (contamination-clean), **every top-tier signal tested collapses**: template SGRV 0/30, ExeVer 0/30, SC 4/4-agreement 0/30, verbalized confidence degrades to near-baseline. Baseline accuracy drops 76.6% → 10.0%. See Section 8 (sec:contam) of the paper.

**X-SGRV primary results** (Llama-3.3-70B cross-family extractor, scale-up to full exp25 sample):

| Benchmark | n | Non-abs | Working | Top-tier precision | 95% CI | Adv-FP |
|---|---|---|---|---|---|---|
| MATH-500 (exp31) | 175 | 94% | 51% | **90/96 = 0.938** | [0.869, 0.977] | 1.7% |
| AIME 2025 (exp30) | 30 | 67% | 20% | 1/2 = 0.500 | [0.013, 0.987] | 5.0% |
| CleanMath combo (exp34) | 125 | 77% | 16% | **4/4 = 1.000** | [0.398, 1.000] | 2.0% |

CleanMath = HMMT Feb 2025 (30) + BRUMO 2025 (30) + SMT 2025 (53) + APEX 2025 (12), all post-Qwen cutoff, not gated on HF. Replaces the originally-planned LiveMathBench (gated, required manual terms-accept on HF).

**Deployment-time adversarial filter** (no gold access — tests verifier against candidate±1, candidate+7, candidate*2):

| Benchmark | Pre-filter precision | Post-filter precision | Filter effect |
|---|---|---|---|
| MATH-500 n=175 | 90/96 = 0.938 | 86/91 = 0.945 | Removes 5/6 FPs, drops 3 working verifiers, 5 top-tier cases |
| AIME 2025 n=30 | 1/2 = 0.500 | **1/1 = 1.000** | Catches aime2025_29 geometry FP |

One residual MATH FP survives the filter (verifier insensitive to order-1 perturbations). Consensus catches it (see below).

**Cross-extractor consensus** (Llama-3.3-70B ∩ DeepSeek-V3, both cross-family to Qwen solver):

| Benchmark | Llama solo | DeepSeek solo | Consensus (strict) | Consensus (loose) |
|---|---|---|---|---|
| MATH-500 n=175 | 90/96 = 0.938 | 82/88 = 0.932 | **73/73 = 1.000** at 41.7% cov | 75/79 = 0.949 at 45.1% cov |
| AIME 2025 n=30 | 1/2 = 0.500 | 1/1 = 1.000 | 1/1 = 1.000 at 3.3% cov | 1/1 = 1.000 |

Strict consensus = both extractors produce working verifiers AND both accept the candidate. Loose consensus = both accept but working-verifier constraint relaxed. Disagreements between extractors on MATH-175: 8/175.

**Prior v8 results (still valid)**:
- 2×2 ablation (independence × randomization): Fisher p < 10⁻⁸ for independence as operative factor
- Cell C measured empirically (exp28): deterministic SGRV matches randomized within 4 all-pass verdicts
- Cross-model replication (4 models × 3 families): SGRV top-tier 100% on all four
- Selective prediction MATH-500 n=175: SGRV top tier 33.7% cov, 100% acc [93.9, 100], E[RC-AUC] 0.103 ± 0.009 (best of SGRV/SC/Verb)

**Honest caveats**:
- MATH-500 numbers are contamination-assisted for Qwen; use AIME 2025 + CleanMath combo as realistic operating points.
- CleanMath top-tier precision CI is wide (4/4 → [0.40, 1.00]) because solver baseline is only 8%.
- 7B-vs-70B same-family ablation confounds scale with family; DeepSeek-V3 vs Llama-70B consensus is the clean cross-family test.
- Semantic entropy baseline (Kuhn 2023, Farquhar 2024) still running — numbers will be filed when exp35 completes.

---

## 1. Project Framing (v7 vs v6)

### What changed from v6 to v7

v6 was "spec-grounded verification for math." v7 is "a cross-domain empirical characterization of a universal failure mode in LLM self-verification, with SGRV as a domain-agnostic remedy."

The reframe is not cosmetic. It changes:
- **Scope**: math-only → 4 reasoning domains (math, code, physics, logic)
- **Primary contribution**: a method → a measurement of a general phenomenon + method
- **Mechanism evidence**: none → 2×2 ablation isolating independence
- **Scaling evidence**: two models → 5 Qwen sizes (1.5B to 72B) + 3 other families
- **Artifact**: code only → released benchmark (SPECBENCH-MINI)
- **Baselines**: ExeVer only → 4 published PRMs head-to-head

### Core narrative (4 claims)

1. **Correlated-verification collapse is universal**: Same-model self-verification has a structural false-positive floor that scales with problem difficulty, across 4 reasoning domains and 8 model families.

2. **The independence principle is falsifiable**: A 2×2 ablation (independence × randomization) isolates the causal mechanism. Independence alone produces most of the FPVR reduction; randomization adds coverage but not precision.

3. **SGRV is a domain-agnostic implementation**: The same pipeline (classify → extract from spec → generate test → execute) works on math (SymPy), code (Python exec), physics (SymPy units), and propositional logic (z3 SMT), with FPVR <5% across all four.

4. **Scaling law**: Same-model FPVR scales inversely with model capability (weaker models produce more catchable errors); SGRV FPVR is approximately flat across model scales. The effect size grows for weaker models.

---

## 2. System Architecture

```
                        MATHAI v7: Cross-Domain SGRV
                        ============================

   +------------------+  +------------------+  +------------------+  +------------------+
   |  MATH problems   |  |  Code problems   |  | Physics problems |  |  Logic problems  |
   |  (MATH-500,      |  |  (HumanEval+,    |  | (GPQA-physics    |  |  (ProofWriter,   |
   |   AIME-24/25,    |  |   MBPP+)         |  |  quantitative)   |  |   FOLIO)         |
   |   OlympiadBench) |  |                  |  |                  |  |                  |
   +--------+---------+  +--------+---------+  +--------+---------+  +--------+---------+
            |                     |                     |                     |
            +---------------------+---------------------+---------------------+
                                          |
                                          v
                             +------------+------------+
                             |   Solver Model          |
                             |   (Qwen2.5 1.5/7/14/32/ |
                             |    72B, Llama-3.3-70B,  |
                             |    Mistral-Large,       |
                             |    DeepSeek-V3)         |
                             +------------+------------+
                                          |
                                          v
                              NL Solution (per domain format)
                                          |
                               +----------+----------+
                               |                     |
                               v                     v
                     +---------+-------+  +---------+-------+
                     | ExeVer          |  | SGRV            |
                     | (same-model     |  | (spec-grounded  |
                     |  baseline)      |  |  testing)       |
                     +---------+-------+  +---------+-------+
                               |                     |
                               |      +-----+--------+--------+------+
                               |      |     |                 |      |
                               |      v     v                 v      v
                               |  SymPy  Python exec       SymPy   z3 SMT
                               |  (math) (code)            units   (logic)
                               |                           (physics)
                               |                     |
                               +----------+----------+
                                          |
                                          v
                             +------------+------------+
                             |   Evaluation Pipeline   |
                             |                         |
                             | Track 1: Verifier quality
                             |  - FPVR [95% CI]        |
                             |  - Risk-coverage curves |
                             |  - Matched-coverage     |
                             |  - Per-domain results   |
                             |                         |
                             | Track 2: Mechanism      |
                             |  - 2x2 ablation         |
                             |  - Scaling laws         |
                             |                         |
                             | Track 3: Downstream PRM |
                             |  - Best-of-N selection  |
                             |  - Head-to-head vs      |
                             |    Qwen-Math-PRM,       |
                             |    GenPRM,              |
                             |    Math-Shepherd-PRM    |
                             |                         |
                             | Track 4: External       |
                             |  - ProcessBench         |
                             |  - PRMBench             |
                             |                         |
                             | Track 5: Artifact       |
                             |  - SPECBENCH-MINI       |
                             |    (400 traces, 4       |
                             |    domains, released    |
                             |    on HuggingFace)      |
                             +-------------------------+
```

---

## 3. Directory Structure

```
MATHAI/
+-- src/
|   +-- exever/                       # Baseline same-model verification
|   +-- pbt/                          # SGRV framework
|   |   +-- claim_classifier.py       # Math claim types (v5)
|   |   +-- code_classifier.py        # [v7 NEW] Code claim types
|   |   +-- physics_classifier.py     # [v7 NEW] Physics claim types
|   |   +-- logic_classifier.py       # [v7 NEW] Logic claim types (z3)
|   |   +-- test_templates.py         # Math templates (v5)
|   |   +-- code_templates.py         # [v7 NEW] Code execution templates
|   |   +-- physics_templates.py      # [v7 NEW] Unit-aware physics templates
|   |   +-- logic_templates.py        # [v7 NEW] z3 SMT templates
|   |   +-- pipeline.py               # Orchestrator (domain-agnostic)
|   |   +-- prm_data.py               # Math-Shepherd format converter
|   |   +-- evaluate.py               # Best-of-N with PRM scoring
|   |
|   +-- inference/
|   |   +-- model_wrapper.py
|   |   +-- together_client.py        # [v7 NEW] Async Together API wrapper
|   |
|   +-- eval/
|   |   +-- answer_check.py
|   |   +-- metrics.py                # FPVR, CIs, risk-coverage
|   |   +-- processbench_eval.py      # External benchmark eval
|   |   +-- prmbench_eval.py          # [v7 NEW]
|   |
|   +-- data/
|       +-- load_math.py              # MATH-500
|       +-- load_aime.py              # [v7 NEW] AIME-24, AIME-25
|       +-- load_olympiadbench.py     # [v7 NEW]
|       +-- load_humaneval_plus.py    # [v7 NEW]
|       +-- load_gpqa_physics.py      # [v7 NEW]
|       +-- load_proofwriter.py       # [v7 NEW]
|
+-- training/
|   +-- train_prm_modal.py            # Qwen2.5-Math-7B LoRA PRM training
|
+-- experiments/
|   +-- run_exp{1..18}_*.py           # v5 + ProcessBench
|   +-- run_exp19_scaling_qwen.py     # [v7] Qwen 1.5/7/14/32/72B
|   +-- run_exp20_cross_family.py     # [v7] Llama, Mistral, DeepSeek
|   +-- run_exp21_code_domain.py      # [v7] HumanEval+, MBPP+
|   +-- run_exp22_physics_domain.py   # [v7] GPQA-physics
|   +-- run_exp23_logic_domain.py     # [v7] ProofWriter, FOLIO
|   +-- run_exp24_mechanism_2x2.py    # [v7] 2x2 ablation
|   +-- run_exp25_bon_with_prms.py    # [v7] Best-of-8 head-to-head
|   +-- run_exp26_specbench_release.py# [v7] Generate SPECBENCH-MINI
|
+-- analysis/
|   +-- risk_coverage.py              # v5 complete
|   +-- processbench_stratified.py    # v5 complete
|   +-- scaling_laws.py               # [v7 NEW] FPVR vs model capability
|   +-- mechanism_ablation.py         # [v7 NEW] 2x2 analysis
|   +-- cross_domain_comparison.py    # [v7 NEW]
|
+-- figures/
|   +-- fig{1..26}_*.{pdf,png}        # All paper figures
|
+-- results/
|   +-- exp{15..26}_*.json
|   +-- specbench_mini.json           # [v7 NEW] Released artifact
|   +-- scaling_analysis.json
|   +-- mechanism_ablation.json
|
+-- paper/
|   +-- main.tex                      # Paper source (v7 rewrite)
|   +-- main.pdf
|   +-- references.bib
|
+-- ARCHITECTURE.md                   # This file (v7.0)
+-- PROPOSAL.md                       # Technical proposal (v7.0)
```

---

## 4. Method: Domain-Agnostic SGRV

### 4.1 The universal pipeline

The SGRV pipeline is domain-agnostic. Only four components change between domains:

| Component | Math | Code | Physics | Logic |
|---|---|---|---|---|
| Claim classifier | regex over LaTeX | regex over function signatures + docstrings | regex over quantity claims | regex over logical propositions |
| Template library | SymPy expressions | Python exec + assertions | SymPy.physics.units | z3-solver formulas |
| Extraction source | problem text | docstring + test cases | problem statement with units | premise statements |
| Executor | Python subprocess + SymPy | Python subprocess | Python subprocess + SymPy | z3 Solver().check() |

The core invariants (independence taxonomy, extraction validation, sandboxed execution, FPVR metric) are shared across all domains. This makes SGRV a genuine framework, not a math-specific hack.

### 4.2 Independence taxonomy (unchanged from v5)

For every generated test:
- **Fully independent**: All operands from the problem/docstring/spec
- **Partially independent**: Template from spec, some operands from prior model steps
- **Dependent**: All operands from model state → FORBIDDEN (falls to UNTESTABLE)

### 4.3 Claim types per domain

**Math** (v5 complete):
```
ROOT_CLAIM, FACTORING, ALGEBRAIC_EQUIV, DIVISIBILITY, NUMERICAL_EVAL,
ENUMERATION, COORDINATE, MODULAR, FINAL_ANSWER
```

**Code** (v7 NEW):
```
RETURN_VALUE: "f(inputs) returns expected" -> exec and compare
TYPE_CLAIM:   "output is type T"            -> isinstance check
INVARIANT:    "loop terminates" / "list sorted" -> runtime check
ASSERTION:    "assert P holds at line L"    -> insert assertion, run tests
COMPLEXITY:   "runs in O(f(n))"             -> empirical timing curve fit
```

**Physics** (v7 NEW):
```
QUANTITATIVE_EVAL: "the energy is 4.2 J"    -> SymPy + units check
UNIT_CONSISTENCY: "result in meters"        -> unit propagation check
DIMENSIONAL_ANALYSIS: "has dimension [L][T^-1]" -> SymPy dimensional check
PROPORTIONALITY: "F is proportional to q_1*q_2/r^2" -> regression on synthetic data
CONSERVATION: "energy is conserved"         -> sum-check across steps
```

**Logic** (v7 NEW):
```
LOGICAL_IMPLICATION: "A, B therefore C"     -> z3 check (A ∧ B) → C tautology
TAUTOLOGY_CLAIM:     "P is always true"    -> z3 check P is tautology
UNSAT_CLAIM:         "P is unsatisfiable"  -> z3 check ¬∃assignment(P)
QUANTIFIER_CLAIM:    "∀x.P(x)"             -> z3 FOL check
CASE_EXHAUSTIVE:     "cases cover all"     -> z3 disjunction check
```

### 4.4 Critical design decision: specification extraction

For v7 to work cross-domain, **the specification must be machine-readable**. For each domain, we define a canonical spec format:

| Domain | Spec source | Format |
|---|---|---|
| Math | Problem text | LaTeX equations + NL constraints |
| Code | Docstring + test cases | Function signature + example I/O |
| Physics | Problem text | Given quantities with units + query |
| Logic | Premises + query | Propositional/FOL formulas |

Extraction templates parse the spec into structured form BEFORE the model generates a solution. Then when the model produces a claim, the template checks it against this pre-parsed spec. This enforces independence: the spec is parsed independently of the solution.

---

## 5. Experiments (v7)

### 5.1 E1-E18: v5 experiments (already complete)

All v5 results stand. Risk-coverage curves and per-generator stratification are already computed.

### 5.2 E19: Scaling laws within Qwen family

**Goal**: Show same-model FPVR scales inversely with model capability; SGRV FPVR is flat.

**Protocol**:
- Models: Qwen2.5 1.5B, 7B, 14B, 32B, 72B (all available on Together)
- Problems: MATH-500 (same 500 problems for all models)
- For each model × each problem: generate solution, run ExeVer, run SGRV
- Plot: FPVR vs model accuracy (6 points per method)

**Expected outcome**: Monotonic ExeVer FPVR curve (decreases as models get better), flat SGRV FPVR. The gap is largest for weaker models where correlated errors are most numerous.

**Cost**: $120 Together API (5 models × 500 problems × ~1500 tokens each)

### 5.3 E20: Cross-family generalization (scaling breadth)

**Goal**: Show the universal-collapse claim is not Qwen-specific.

**Protocol**:
- Models: Llama-3.3-70B, Mistral-Large-2411, DeepSeek-V3
- Problems: MATH-500 (all 500)
- Metric: FPVR per family with 95% CIs

**Expected outcome**: FPVR <5% across all 3 families (combined n ≈ 900 PASS cases, CI upper bound ~0.4%). Addresses v6's n=50 weakness.

**Cost**: $200 Together API (3 models × 500 problems)

### 5.4 E21: Code domain

**Goal**: Cross-domain generalization to code verification.

**Protocol**:
- Models: Qwen2.5-Coder-7B, DeepSeek-V3
- Benchmarks: HumanEval+ (164), MBPP+ (378)
- SGRV runs with code templates (E4.3 claim types)
- Verification: run generated code against claimed test outputs via subprocess
- Compare to running the actual test suite (oracle)

**Expected outcome**: FPVR <2% (code specs are cleaner than math), coverage ~40% (higher than math because docstrings give cleaner extraction).

**Cost**: $50 Together API

### 5.5 E22: Physics domain

**Goal**: Cross-domain generalization to physics computation.

**Protocol**:
- Dataset: GPQA-physics (quantitative subset, ~200 problems)
- Models: Qwen2.5-72B, Llama-3.3-70B
- SGRV with unit-aware SymPy templates (E4.3)
- Verification: numerical check with unit consistency

**Expected outcome**: FPVR <3%, coverage ~30% (limited by problems that give all quantities numerically).

**Cost**: $40 Together API

### 5.6 E23: Logic domain

**Goal**: Cross-domain generalization to propositional/FOL reasoning.

**Protocol**:
- Datasets: ProofWriter (propositional), FOLIO (first-order logic)
- Sample n=400 problems stratified by depth
- Models: Qwen2.5-72B, Llama-3.3-70B
- SGRV with z3-based logic templates (E4.3)
- Verification: z3 tautology/satisfiability check on claimed implications

**Expected outcome**: FPVR <1% (logic is fully formally checkable), coverage ~70% (much higher than math because logical claims are cleaner).

**Cost**: $40 Together API + z3-solver pip install (free)

### 5.7 E24: 2×2 mechanism ablation (THE critical experiment)

**Goal**: Isolate whether SGRV's improvement comes from independence or randomization.

**Protocol**: Run all four cells on MATH-500 with Qwen2.5-Math-7B:

| Cell | Method | Description |
|---|---|---|
| A | ExeVer | Solution-grounded, deterministic (`expand(A) == expand(B)`) |
| B | SGRV-no-indep | Solution-grounded, randomized (same code but at 200 random points) |
| C | SGRV-no-random | Spec-grounded, deterministic (extract from problem, check at gold values) |
| D | SGRV (full) | Spec-grounded, randomized (extract from problem, check at random points) |

**Analysis**: 2-way ANOVA on FPVR with independence and randomization as factors. Report main effects and interaction. If independence explains most of the variance, the claim "independence is the mechanism" is supported.

**Cost**: $0 (reuses existing exp5 data and existing MATH-500 solutions)

### 5.8 E25: Head-to-head with published PRMs on best-of-8

**Goal**: Fix the current "no strong baselines" weakness.

**Protocol**:
- Generate 8 candidates per problem on MATH-500 with Qwen2.5-Math-7B (temperature 0.7)
- Score each candidate with:
  - Random@1 (lower bound)
  - Majority@8
  - Qwen2.5-Math-PRM-7B (public weights on HF)
  - GenPRM (public weights, AAAI 2026)
  - Math-Shepherd PRM (public weights, ACL 2024)
  - SGRV filter + Majority@remaining (our method)
  - SGRV-trained PRM (from v5 training runs)
  - Best-of-8 oracle (upper bound)

**Protocol**: Matched-compute comparison. Report best-of-8 accuracy with 95% CIs.

**Cost**: $220 ($120 candidate generation + $100 PRM inference on Modal A100)

### 5.9 E26: SPECBENCH-MINI benchmark release

**Goal**: Release a public artifact.

**Protocol**:
- 400 reasoning traces: 100 each from math, code, physics, logic
- Each trace labeled:
  - `problem`: the problem statement
  - `solution`: the model's chain-of-thought
  - `gold_label`: correct/incorrect (from dataset)
  - `sgrv_testable`: which steps SGRV can test
  - `sgrv_verdict`: PASS/FAIL/UNTESTABLE per testable step
  - `independence_category`: fully/partially/untestable
  - `spot_check`: automated sanity check (not a ground-truth annotation, but a transparency layer)
- Upload to HuggingFace Hub as `mathai/specbench-mini`
- Include reproducibility script

**Cost**: $0 (reuses existing solutions from E1-E23)

---

## 6. Metrics and Statistical Protocol

### 6.1 Primary metrics

| Metric | Definition | Required? |
|---|---|---|
| **FPVR** | P(answer wrong \| all tested steps PASS) | YES |
| **FPVR 95% CI** | Clopper-Pearson exact interval | YES, on every FPVR |
| **Coverage (PC)** | Fraction of problems receiving a verdict | YES |
| **Step coverage (SC)** | Fraction of individual steps tested | YES |
| **Independence rate** | Fraction of tests that are fully independent | YES |
| **Risk-coverage curve** | FPVR as a function of coverage threshold | YES (primary figure) |
| **Matched-coverage FPVR** | SGRV vs ExeVer at same coverage | YES |
| **Error-detection P/R/F1** | For ProcessBench/PRMBench | YES |
| **Acceptance precision** | Fraction of PASS verdicts that are correct | YES (distinct from error-detection precision) |
| **Best-of-N accuracy** | For downstream PRM comparison | YES |

### 6.2 Confidence intervals

Every proportion in the paper is accompanied by a Clopper-Pearson 95% CI. Every mean is accompanied by a percentile bootstrap 95% CI (B=10,000 resamples, fixed seed 42).

### 6.3 Hypothesis testing

For the 2×2 mechanism ablation, we use two-way ANOVA with independence and randomization as factors. We report F-statistics, p-values, and effect sizes (eta-squared) for main effects and interaction. Alpha = 0.05.

### 6.4 Multiple comparisons

When reporting per-domain results, per-model results, or per-claim-type results, we apply Holm-Bonferroni correction to p-values.

---

## 7. Risks and Contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| FPVR on hard benchmarks (AIME, GPQA) is higher than 5% | Medium | Report honestly; the risk-coverage curve still dominates |
| Logic domain has low coverage (<30%) | Low | z3 handles propositional logic natively; should be high |
| Physics domain units extraction fails on >50% of problems | Medium | Fall back to numerical-only check without unit verification |
| PRM baselines don't load from HF | Low | Alternate: Math-Shepherd PRM is the most stable; use it alone |
| 2×2 ablation interaction effect is large (independence ⊥ randomization don't separate) | Medium | Still publishable — report the interaction as the finding |
| Modal runs silently fail (from v6 experience) | Medium | Use `modal run --detach` for all long jobs |
| SPECBENCH-MINI has annotation errors | Low | Mark it as "auto-labeled with spot-check" not "gold-standard" |

---

## 8. Compute Budget ($900 hard cap, fully automated)

| Experiment | Cost | Platform |
|---|---|---|
| E19: Qwen scaling (1.5/7/14/32/72B × 500) | $120 | Together API |
| E20: Cross-family (Llama, Mistral, DeepSeek × 500) | $200 | Together API |
| E21: Code (2 models × 542 problems) | $50 | Together API |
| E22: Physics (2 models × 200) | $40 | Together API |
| E23: Logic (2 models × 400) | $40 | Together API |
| E24: Mechanism ablation (reuses data) | $0 | Local CPU |
| E25: Best-of-N (3 models × 500 × 8 + 3 PRMs) | $220 | Together + Modal |
| E26: SPECBENCH-MINI (reuses data) | $0 | Local CPU |
| E10: Full ProcessBench + PRMBench | $60 | Local CPU + Modal |
| Buffer for debugging / re-runs | $170 | |
| **Total** | **$900** | |

All experiments are scripted end-to-end against existing Modal + Together credentials. No manual annotation, no external expert work, no deployment study.

---

## 9. Expected Outcomes (v7 projections)

### Strongest expected result

**Same-model FPVR scales inversely with model capability; SGRV FPVR is flat across scale**

This is the single most important finding. If confirmed across Qwen 1.5B→72B, it establishes correlated-verification collapse as a fundamental property of self-verification and SGRV as a scale-independent solution.

### Quantitative predictions

| Domain | Model | Expected FPVR (ExeVer) | Expected FPVR (SGRV) |
|---|---|---|---|
| Math | Qwen2.5-1.5B | 35-45% | 2-5% |
| Math | Qwen2.5-Math-7B | 13.8% (measured) | 0-2% (measured) |
| Math | Qwen2.5-72B | 8-12% | 0-2% |
| Code | Qwen-Coder-7B | 15-25% | 0-2% |
| Physics | Qwen2.5-72B | 10-15% | 0-3% |
| Logic | Qwen2.5-72B | 5-10% | 0-1% |

### Mechanism ablation prediction

| Cell | FPVR prediction |
|---|---|
| A: solution-grounded + deterministic (ExeVer) | ~14% |
| B: solution-grounded + randomized | ~12% |
| C: spec-grounded + deterministic | ~2% |
| D: spec-grounded + randomized (SGRV) | ~0% |

**Interpretation if this holds**: Independence alone gets you ~86% of the way. Randomization adds the final 2pp. The "independence principle" claim is justified; randomization is a secondary coverage improvement.

### Best-of-N prediction

| Method | Best-of-8 Accuracy (Qwen2.5-Math-7B, MATH-500) |
|---|---|
| Random@1 | ~81% |
| Majority@8 | ~87% |
| Qwen2.5-Math-PRM-7B | ~89% |
| GenPRM | ~88% |
| Math-Shepherd PRM | ~86% |
| **SGRV filter + Majority** | **~88-90%** |
| Best-of-8 oracle | ~94% |

If SGRV filter matches or exceeds Qwen2.5-Math-PRM-7B (trained on much more data), that's a strong NeurIPS-level result. If it underperforms, the fallback framing is "SGRV as complementary filter that's additive to trained PRMs."

---

## 10. What v7 Does NOT Do (honest scope limits)

- **No human deployment trial**: Would require IRB + users + months
- **No new theoretical framework**: This is an empirical paper only
- **No novel training method**: PRM training uses standard SFT + LoRA
- **No cross-modality**: Text-only, no vision/speech/multimodal
- **No proofs**: No theorems, no formal bounds beyond citing Schwartz-Zippel
- **No large-scale (>100B) solvers**: Budget doesn't allow frontier models
- **No RL training**: No GRPO, no DPO, no process reward RL

These limitations cap v7 at Nature Machine Intelligence / NeurIPS main track tier. Nature main is out of reach without deployment + theory.

---

## 11. Venue Strategy

| Venue | Probability (after v7) | Why |
|---|---|---|
| ICML AI4Math Workshop | 95% | v5 already passes; v7 is overkill |
| NeurIPS 2026 main track | 55-65% | Strong empirical case with 4 domains, scaling, ablation, baselines |
| NeurIPS Datasets & Benchmarks | 60% | SPECBENCH-MINI is a valid artifact contribution |
| ICML 2026 main track | 50% | Same bar as NeurIPS |
| EMNLP 2026 | 55% | Process supervision is active there |
| **Nature Machine Intelligence** | **10-20%** | Cross-domain empirical story + released artifact might clear the bar, but no deployment |
| Nature Communications | 10% | Broader scope, but still needs the impact story |
| Nature main | <1% | Would need deployment + theory, not in budget |

**Recommended strategy**: Submit to NeurIPS 2026 main track as the primary target. Workshop version is a guaranteed fallback. Nature Machine Intelligence is a stretch submission if NeurIPS reviews are positive.

---

## 12. Reproducibility

- All code in public repo
- All data from public HuggingFace datasets
- SPECBENCH-MINI released on HF Hub
- Modal scripts with pinned dependencies
- Fixed random seeds (seed=42) throughout
- Bootstrap B=10,000 with fixed seed
- Per-experiment requirements.txt
- Expected reproduction cost for full pipeline: ~$400 on a single A100 (lower than the $900 development cost because re-runs aren't needed)
