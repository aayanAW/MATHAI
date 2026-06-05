# Project Architecture — Dependency-Aware Jury Verification for LLM Math Reasoning

**Internal codename:** DAJV
**Library release name:** `verifyensemble`
**Public-facing paper title (working):** *Calibrated LLM-Jury Verification: A Dependency-Aware Aggregation Framework for Selective Prediction on Math Reasoning*
**Document version:** 0.1 (2026-05-25)
**Target venue (primary):** ICML 2027 main track
**Target venue (fallback):** NeurIPS 2026 main track; ICLR 2027 main track
**Workshop submissions allowed:** AI for Math @ ICML 2026 (with explicit framing as preliminary work); MATH-AI @ NeurIPS 2026
**Owners:** four PhD-level researchers, workstreams A–D
**Lead artifact:** existing X-SGRV codebase + extractor cache + result JSONs (do not throw out; treat as Stage 0 evidence)

---

## 1. Why we are pivoting

The Area-Chair audit (`HANDOFF_XSGRV_paper_2026-05-14.md`, audit appended to that session) identified the following load-bearing problems with the original X-SGRV submission strategy:

1. **VERGE (Singh et al. 2026)** does multi-model consensus + formal verification on the same problem class, published ahead of us. The naïve X-SGRV pipeline (cross-family LLM → SymPy → unanimous consensus) is a strictly weaker variant.
2. **The independence assumption is empirically broken.** Kuai 2026, CARE (Zhao 2026), Denisov-Blanch 2026, and Kim 2025 all establish that LLM verifiers exhibit correlated errors that scale with model capability. Naïve consensus under correlated errors does not bound joint FP by the product of marginals.
3. **Statistical power is too thin.** Contamination-clean comparisons (n=4 on CleanMath) cannot support "ties Skywork-PRM, doubles Qwen-PRM" claims.
4. **No theoretical contribution.** Top-tier reviewers expect a new method, a new theoretical result, or a paradigm-shifting empirical finding. The original X-SGRV is a careful engineering combination.

The pivot retargets the contribution to the residual gap that the recent literature opens:

> **The field documents that LLM verifiers are correlated. Nobody has provided a *calibrated* consensus method that quantifies the residual dependency and converts it into a deployment-ready selective-prediction signal with valid coverage guarantees.**

DAJV closes that gap.

---

## 2. Research question and hypotheses

### Primary research question

*Given a set of $k$ cross-family LLM verifiers with measured pairwise dependency $\rho_{ij}$ on math-reasoning problems, can we derive a calibrated joint acceptance probability for a candidate answer that (a) provides valid Clopper–Pearson coverage at deployment time, (b) Pareto-dominates naïve consensus on the risk–coverage curve, and (c) generalizes from a calibration set to held-out, contamination-clean benchmarks?*

### Hypotheses (pre-registered before any new data is collected)

- **H1 (theory):** There exists a concentration inequality bounding joint false-positive rate as a function of measured pairwise dependency that is strictly tighter than the trivial product-of-marginals bound and strictly looser than the union bound; the gap shrinks with $\rho \to 0$.
- **H2 (empirical, calibration):** Across $k \geq 5$ frontier LLM verifiers, the empirical joint false-positive rate on a held-out set deviates from the independence prediction by $\geq 3\times$ on at least 50% of pairs.
- **H3 (empirical, deployment):** A dependency-aware aggregation rule reduces expected calibration error (ECE) by $\geq 30\%$ versus naïve consensus on a held-out test set drawn from the same problem distribution as the calibration set.
- **H4 (empirical, generalization):** Pairwise dependency $\rho_{ij}$ estimated on a calibration set (MATH-500) transfers to held-out benchmarks (AIME 2025, CleanMath, OlympiadBench) within $\pm 0.10$ of the calibration value on $\geq 70\%$ of pairs.
- **H5 (empirical, Pareto):** At any matched coverage $\in [5\%, 90\%]$ on MATH-500, the dependency-aware consensus achieves precision $\geq$ Qwen-PRM, Skywork-PRM, GenPRM, and naïve consensus.
- **H6 (mechanism, optional):** Measured dependency $\rho_{ij}$ correlates with shared training-data overlap, shared base-architecture family, and shared instruction-tuning recipe.

### Falsification branches (committed before launch)

| Branch | Condition | Action |
|---|---|---|
| A | All five primary hypotheses hold | Submit to ICML 2027 main |
| B | H1 + H3 hold, H4 fails | Reframe around in-distribution calibration only; submit to workshop and reduce scope |
| C | H1 fails (bound not provable / trivial) | Drop theory; pivot to mechanism (audit Pivot C below) |
| D | H2 fails (verifiers actually independent on math) | This is itself a result — flip the paper to "Cross-family LLM verifiers on math are empirically independent: a measurement study" |
| E | H5 fails (PRMs Pareto-dominate) | Abandon main-track submission, release library + benchmark, write workshop paper on the negative result |

---

## 3. Contribution stack

| Layer | Contribution | Output artifact |
|---|---|---|
| Theory | Concentration inequality on joint FP under measured pairwise dependency. Sample-complexity bound for estimating the dependency matrix. | Theorems 1–3 in the paper; proofs in appendix |
| Method | Dependency-aware aggregation rule. Calibration procedure that converts the dependency matrix + per-verifier marginals into a posterior over correctness with valid confidence interval. | `verifyensemble.aggregate(...)` API |
| Empirical | Map of pairwise dependency across $\geq 12$ frontier LLM verifiers on $\geq 6$ math benchmarks. First open dataset of (extractor, problem, verifier-code, gold-answer, candidate, accept/reject) tuples for ensemble-verification research. | Released dataset (~250k rows) |
| Calibration | Calibrated risk–coverage curves on contamination-clean benchmarks with bootstrap confidence bands. ECE numbers for the aggregation rule. | Figures 2–5 in the paper |
| System | Python library (`verifyensemble`) for plug-in calibrated jury verification. Sandbox-hardened verifier execution. Caching, async batching, dependency-matrix re-estimation. | PyPI release; GitHub repo |
| Benchmark | Evaluation harness covering AbstentionBench, VerifyBench, ProcessBench, AIME-style stress tests. | Reproducibility code |

---

## 4. Technical architecture

### 4.1 System overview

```
                ┌──────────────────────────────────────────────────────┐
                │                  Calibration set                     │
                │  (problem, gold answer, candidate distribution)      │
                └──────────────┬───────────────────────────────────────┘
                               │
                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                      Dependency-mapping stage                       │
   │                                                                     │
   │   For each pair (E_i, E_j) of cross-family extractors:              │
   │       run E_i, E_j on calibration problems                          │
   │       collect (verify_i, verify_j) acceptance pattern               │
   │       compute kappa_ij, joint_FP_ij, CIG_ij                         │
   │   Output: dependency matrix D ∈ R^{k×k}                             │
   └──────────────┬──────────────────────────────────────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                      Deployment stage                               │
   │                                                                     │
   │   Input: problem x, candidate answer â                              │
   │   1) Run all k extractors on x                                      │
   │   2) Sandbox-execute each verify_i against â                        │
   │   3) Apply deployment-time adversarial probe filter                 │
   │      (gold-free; legacy from X-SGRV)                                │
   │   4) Aggregate accept/reject votes using                            │
   │      dependency-aware rule with matrix D                            │
   │   5) Return: P(correct), Clopper–Pearson CI, recommendation         │
   │      ∈ {COMMIT, ESCALATE, ABSTAIN}                                  │
   └─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Module breakdown

```
verifyensemble/
├── extractors/
│   ├── api_wrappers.py        # Together AI, OpenAI, Anthropic, etc.
│   ├── prompt.py              # frozen extractor prompt (legacy X-SGRV)
│   └── parser.py              # extract verify() function from response
├── sandbox/
│   ├── subprocess_sandbox.py  # SIGALRM + 10s timeout, restricted imports
│   ├── adversarial_filter.py  # gold-free probe set {±1, +7, ×2, fallback}
│   └── classify.py            # working / wrong-spec / trivial-or-broken
├── dependency/
│   ├── kappa.py               # Cohen's κ on verifier-pair output patterns
│   ├── joint_fp.py            # empirical joint-FP estimator
│   ├── cig.py                 # Cumulative Information Gain (Kuai 2026)
│   ├── matrix_builder.py      # builds dependency matrix D
│   └── bootstrap.py           # CIs on dependency estimates
├── aggregate/
│   ├── naive_consensus.py     # baseline: unanimous + product-of-marginals
│   ├── care.py                # CARE-style baseline (Zhao 2026)
│   ├── dajv.py                # primary contribution: dependency-aware rule
│   └── posterior.py           # converts vote pattern + D → P(correct)
├── theory/
│   ├── bound.py               # symbolic + numeric tightness of theorems
│   └── sample_complexity.py   # estimator for required calibration n
├── evaluation/
│   ├── risk_coverage.py       # curves with bootstrap bands
│   ├── ece.py                 # expected calibration error
│   ├── reliability.py         # reliability diagrams
│   └── pareto.py              # Pareto-dominance test across baselines
└── cli.py                     # `verifyensemble verify --problem ... --answer ...`
```

### 4.3 Core algorithm pseudocode

```python
def deployment_predict(problem, candidate, extractors, D, marginals,
                       coverage_target=0.90, abstain_threshold=0.95):
    """Return (P_correct, CI, recommendation)."""
    # Stage 1: extract and sandbox-execute
    votes = []
    for E in extractors:
        script = E.extract(problem)
        if script == "UNVERIFIABLE":
            votes.append(("abstain", None))
            continue
        verifier = sandbox.compile(script)
        if not verifier.working():
            votes.append(("broken", None))
            continue
        # Deployment-time adversarial probe
        if any(verifier(p) for p in probe_set(candidate)):
            votes.append(("broken", None))
            continue
        votes.append(("accept" if verifier(candidate) else "reject", verifier))
    # Stage 2: dependency-aware aggregation
    accepts = [i for i, (v, _) in enumerate(votes) if v == "accept"]
    rejects = [i for i, (v, _) in enumerate(votes) if v == "reject"]
    if not accepts and not rejects:
        return None, None, "ABSTAIN_NO_VERIFIERS"
    P_correct = dajv.posterior(accepts, rejects, D, marginals)
    lower, upper = clopper_pearson_interval(P_correct, n_effective(D, accepts))
    if lower >= abstain_threshold:
        return P_correct, (lower, upper), "COMMIT"
    if upper < 0.5:
        return P_correct, (lower, upper), "ABSTAIN_LIKELY_WRONG"
    return P_correct, (lower, upper), "ESCALATE"
```

### 4.4 Aggregation rule (DAJV) — formal

Given $k$ extractors with acceptance pattern $v \in \{0,1\}^k$, per-verifier marginal FP rate $\pi_i$, and dependency matrix $D$ with entries $\rho_{ij} \in [0,1]$:

$$P(\hat a \text{ correct} \mid v) = \frac{P(v \mid \hat a \text{ correct}) \cdot P(\hat a \text{ correct})}{P(v)}$$

The likelihood under the dependency-aware copula is:

$$P(v \mid \hat a \text{ correct}) = \prod_i (1 - \pi_i)^{v_i} \pi_i^{1-v_i} \cdot \exp\!\left(\sum_{i<j} \rho_{ij} \cdot \phi(v_i, v_j)\right)$$

where $\phi$ is a 2-way interaction term derived from the empirical second-order log-likelihood of the calibration set. Estimation: maximum pseudo-likelihood on calibration data; $O(k^2)$ parameters.

**Theorem 1 (informal):** Joint FP $\leq \prod_i \pi_i \cdot \exp(\sum_{i<j} g(\rho_{ij}))$ where $g$ is convex and $g(0) = 0$.

**Theorem 2 (informal):** $O(\log(k^2) / \epsilon^2)$ calibration problems suffice to estimate $D$ to within $\epsilon$ in entrywise $\ell_\infty$ norm.

Full statements + proofs in §A of the paper.

---

## 5. Experimental plan

### 5.1 Datasets (all public)

| Role | Dataset | n | Provenance | Use |
|---|---|---|---|---|
| Calibration (in-distribution) | MATH-500 | 500 | Lightman et al. 2024 | Estimate D, fit aggregation rule |
| In-distribution test | GSM8K test | 1,319 | Cobbe et al. 2021 | Held-out, easy regime |
| Out-of-distribution (contamination-clean) | AIME 2024 | 30 | MathArena | Held-out, hard, post-cutoff |
| Out-of-distribution (contamination-clean) | AIME 2025 | 30 | MathArena | Held-out, hard, post-cutoff |
| Out-of-distribution (contamination-clean) | HMMT-Feb 2025 | 30 | MathArena | Held-out |
| Out-of-distribution (contamination-clean) | BRUMO 2025 | 30 | MathArena | Held-out |
| Out-of-distribution (contamination-clean) | SMT 2025 | 53 | MathArena | Held-out |
| Out-of-distribution (contamination-clean) | APEX 2025 | 12 | MathArena | Held-out |
| Difficulty stress | Omni-MATH (top-100 hardest) | 100 | Gao et al. 2024 | Ceiling test |
| Difficulty stress | OlympiadBench | ~8,000 | Olympiadbench 2024 | Stratified subsample 500 |
| Scope stress (proof-mode) | PutnamBench (random 100) | 100 | Tsoukalas et al. 2024 | Refusal-rate / scope test |
| Selective-prediction benchmark | AbstentionBench (math subsets) | varies | Kirichenko et al. 2025 | Calibration evaluation |
| Verifier benchmark | VerifyBench | 4,000 | Li et al. 2025 | Verifier quality evaluation |
| Process-step benchmark | ProcessBench | 6,310 | Zheng et al. 2024 | Step-level verification |

Pre-register all splits before estimating D.

### 5.2 Extractors and solvers

**Cross-family extractors (k = 12):**

Diverse by lab, base architecture, instruction-tuning, and code-training emphasis. Selected to maximize expected dependency variance.

| ID | Model | Lab | Why included |
|---|---|---|---|
| E01 | Llama-3.3-70B-Instruct-Turbo | Meta | Legacy X-SGRV |
| E02 | Llama-4-Maverick (if released) | Meta | Same family, different generation — within-family dependency |
| E03 | DeepSeek-V3 | DeepSeek | Legacy X-SGRV |
| E04 | DeepSeek-R1 | DeepSeek | Reasoning-tuned variant — within-family |
| E05 | gpt-oss-120b | OpenAI | Open-weights frontier |
| E06 | GPT-5-mini | OpenAI | Closed-weights, similar provider |
| E07 | Claude Sonnet 4.6 | Anthropic | Legacy X-SGRV |
| E08 | Claude Opus 4.7 | Anthropic | Within-family |
| E09 | Qwen3-Coder-480B | Alibaba | Code-emphasis |
| E10 | Qwen2.5-Math-72B | Alibaba | Math-emphasis, within-Alibaba but different specialization |
| E11 | Mistral-Large-2 | Mistral | Adds Mistral lab |
| E12 | Gemini-2.5-Pro | Google | Adds Google lab |

**Solvers (4):**

| ID | Model | Use |
|---|---|---|
| S01 | Qwen2.5-7B-Instruct-Turbo | Legacy X-SGRV baseline solver |
| S02 | Qwen2.5-Math-7B-Instruct | Contamination-flagged solver (FAR scaling) |
| S03 | DeepSeek-V3.1 | Strong solver (solver-rotation control) |
| S04 | Llama-3.3-70B-Instruct-Turbo | Strong open-weights solver |

### 5.3 Baselines (concrete configurations)

| Baseline | Configuration | Hyperparameters |
|---|---|---|
| Naïve unanimous consensus | All k extractors must accept | k ∈ {2, 5, 12} |
| Majority consensus | ≥ k/2 accept | Same k |
| Self-consistency | k-sample agreement on solver | k ∈ {4, 10, 64} |
| Semantic entropy (SymPy clustering) | Kuhn 2023 method | DeBERTa-large-mnli |
| Semantic entropy (NLI clustering) | Farquhar 2024 method | Same |
| p(True) | Kadavath 2022 | Default prompt |
| Qwen2.5-Math-PRM-7B | Qwenprm2025 | min/mean/product/last aggregations |
| Skywork-o1-Open-PRM-7B | Skyworkprm2024 | Same aggregations |
| GenPRM | Zhao et al. 2025 | Default |
| DTV (autoformalization) | Zhou 2024 | Isabelle, default |
| Math-Rev / Code-Rev | Liang 2024 | Best-reported config |
| MATH-VF | Kuo Zhou 2025 | CAS + SMT |
| VERGE | Singh 2026 | GPT-OSS-120B base, default |
| CARE | Zhao 2026 | Confounder-aware aggregation |
| Naïve X-SGRV (legacy) | This project, prior version | Frozen prompts and probes |

VERGE and MATH-VF require head-to-head implementation effort — Workstream C, Q2.

### 5.4 Metrics (definitions, all reported with bootstrap 95% CI)

| Metric | Definition | Use |
|---|---|---|
| Precision@coverage | P(answer correct | committed) at fixed coverage | Primary headline |
| Coverage | P(committed) / P(committed ∨ abstain) | Standard |
| Risk–coverage AUC | Area under risk–coverage curve, coverage ∈ [0, 1] | Pareto comparison |
| ECE | Expected calibration error, 15 equal-frequency bins | Calibration quality |
| Brier score | Mean squared error of P(correct) prediction | Calibration quality |
| Reliability diagram slope | Slope of reliability diagram regression | Calibration quality |
| Pairwise Cohen's κ | Inter-verifier agreement on calibration set | Dependency map |
| CIG | Cumulative Information Gain (Kuai 2026) | Dependency map |
| Empirical joint-FP / independence-FP ratio | Joint-FP observed / product-of-marginals predicted | Independence-violation magnitude |
| McNemar mid-p | Matched-binary comparison vs each baseline | Power-aware significance |

### 5.5 Ablations

A1. **Per-component contribution.** Naïve consensus → +adversarial filter → +dependency weighting → +calibration. Report ECE and risk–coverage AUC at each step.

A2. **Dependency matrix sparsity.** Mask all $\rho_{ij}$ below threshold $\tau$. Sweep $\tau \in \{0, 0.05, 0.1, 0.2\}$. Result: how dense does D need to be?

A3. **Calibration-set size.** Estimate D on $n_{cal} \in \{50, 100, 200, 500, 1000\}$ calibration problems. Plot ECE on held-out vs $n_{cal}$. Verifies the theoretical sample complexity bound.

A4. **Cross-benchmark dependency stability.** Estimate D on MATH-500, transfer to AIME 2025. Report per-pair $|\rho_{MATH} - \rho_{AIME}|$ distribution. Confirms H4.

A5. **Within-family vs cross-family pairs.** Stratify dependency by lab. Confirms H6.

A6. **Verifier-count scaling.** Sweep $k \in \{2, 3, 5, 8, 12\}$. Plot ECE and coverage at fixed precision. Determines optimal jury size.

A7. **Solver-rotation invariance.** Use S03 (DeepSeek-V3.1) and S04 (Llama-70B) as alternate solvers, holding extractor cache fixed. Verifies the selective-prediction property is solver-independent.

A8. **Same-model FAR scaling reproduction.** Re-run the L1→L5 FAR experiment on three open solvers. Confirms the legacy 13.8% finding generalizes.

### 5.6 Stress tests

S1. **Adversarial extractor injection.** Submit a known-broken verifier to the ensemble; confirm it is filtered.
S2. **Single-lab attack.** Reduce ensemble to 3 extractors all from one lab; show dependency-aware aggregation correctly degrades calibration with a warning.
S3. **Contamination-shifted problem set.** Use rephrased GSM8K problems (Yang et al. 2023); show calibration holds.
S4. **Cross-language stress.** Translate MATH-500 to Spanish (per Salido 2026); show extractor refusal rate increases proportionally and recommendation correctly shifts to ESCALATE.
S5. **Compute-cost frontier.** Plot precision vs cumulative API cost; compare against VERGE, MATH-VF, PRMs.

### 5.7 Generalization tests

G1. **Benchmark transfer.** Calibrate on MATH-500, evaluate on OlympiadBench. Pre-registered $\Delta$ECE threshold.
G2. **Lab-set transfer.** Calibrate with 6 extractors, deploy with the other 6. Pre-registered $\Delta$ECE threshold.
G3. **Generation-time transfer.** Calibrate in Q1 of the project; evaluate the same D 9 months later with re-served extractors (lab updates / re-tunes). Tests temporal stability of dependency.

---

## 6. Workstreams

### Workstream A — Theory, literature, pre-registration (R1)

**Mandate:** prove the two theorems; maintain the lit-tracking corpus; lead pre-registration commits before each empirical phase.

| Task | Deliverable | Due |
|---|---|---|
| Theorem 1 (concentration bound) | Statement + proof + numeric tightness check | Month 3 |
| Theorem 2 (sample complexity) | Statement + proof | Month 4 |
| Pre-registration #1: H1 + H2 | Committed to GitHub before any new data | Month 1 |
| Pre-registration #2: deployment evaluation | Committed before evaluation runs | Month 8 |
| Related-work section | Comprehensive, addresses VERGE/CARE/Kuai/Denisov-Blanch/Kim explicitly | Month 12 |
| Theory chapter of paper | 10-page section (in supplement if needed) | Month 15 |

**Failure pivots:**
- Bound not provable → fall back to empirical regression as primary; theorem becomes a "conjecture, verified empirically" result
- Sample complexity bound trivial → use empirical learning curves only

**Decision gate (month 3):** Is Theorem 1 non-trivial (bound strictly between product-of-marginals and union bound, by more than 1pp on synthetic data)?

### Workstream B — Data, dependency mapping, benchmarks (R2)

**Mandate:** build the dependency-mapping pipeline; curate all benchmarks; produce the open dataset release.

| Task | Deliverable | Due |
|---|---|---|
| Adapt legacy X-SGRV sandbox & extractor harness | Tested, documented | Month 1 |
| Extend extractor set from 6 → 12 | API wrappers + smoke tests | Month 2 |
| Dependency-mapping pipeline | Computes κ, joint-FP, CIG with bootstrap CIs | Month 3 |
| MATH-500 calibration run | Full 12 × 12 D with CIs | Month 4 |
| OOD benchmark runs (AIME 2024/25, CleanMath, OlympiadBench) | Cached extractor outputs | Month 6 |
| AbstentionBench + ProcessBench + VerifyBench evaluation | Numbers checked into repo | Month 8 |
| Open dataset release | ~250k row CSV/Parquet on HuggingFace | Month 14 |

**Failure pivots:**
- Dependency matrix unstable (bootstrap κ CI > 0.10) → enlarge calibration set; if still unstable, switch from κ to CIG as primary dependency measure
- Benchmark API quotas exceeded → caching layer; budget contingency

**Decision gate (month 4):** Bootstrap CI on each $\rho_{ij}$ < 0.05?

### Workstream C — Modeling, system, library (R3)

**Mandate:** implement the calibrated aggregation rule; ship `verifyensemble` as a polished open-source library; sandbox-security audit.

| Task | Deliverable | Due |
|---|---|---|
| Aggregation rule implementation | `verifyensemble.aggregate.dajv` | Month 5 |
| Calibration procedure | Fit + cross-validate | Month 6 |
| Library v0.1 (internal) | Working API, tests passing | Month 7 |
| Sandbox security audit | Pen-test report; fix anything found | Month 8 |
| Library v0.9 (beta release on PyPI) | Public preview | Month 12 |
| Library v1.0 (paired with submission) | Release-quality | Month 16 |

**Failure pivots:**
- Aggregation rule fits poorly on calibration (ECE > 0.10 on MATH-500) → switch from Bayesian copula to a fitted MLP correction on top of naïve consensus; report this as the "empirical aggregation" baseline
- Sandbox vulnerability found → halt public release until patched

**Decision gate (month 7):** Library passes integration tests on all benchmarks with calibration ECE < 0.05?

### Workstream D — Evaluation, baselines, writing, submission (R4)

**Mandate:** run head-to-head against every required baseline; produce paper-quality figures with bootstrap bands; write the paper.

| Task | Deliverable | Due |
|---|---|---|
| Re-implement VERGE | Working pipeline matching paper's published numbers within 2pp | Month 6 |
| Re-implement MATH-VF | Same | Month 7 |
| Re-implement Math-Rev/Code-Rev | Same | Month 8 |
| Run all baselines on all benchmarks | Cached outputs | Month 10 |
| Power analysis on all comparison claims | Documented per-claim minimum-detectable effect | Month 11 |
| Figures 1–8 with bootstrap bands | Vector PDFs | Month 14 |
| Paper draft v0.5 | Self-review, all sections | Month 15 |
| Internal review by 3 external collaborators | Reviews collected, addressed | Month 16 |
| Submission package | Anonymized; main + supplement; pre-registration | Month 17 |

**Failure pivots:**
- VERGE Pareto-dominates → reframe around calibration (which VERGE does not provide); narrow contribution to calibrated jury verification
- Power analysis kills any headline → drop that claim from the abstract

**Decision gate (month 15):** Paper draft passes red-team review (no LLM-written prose detected; all numbers power-analyzed; all citations spot-checked against primary sources)?

---

## 7. Timeline (18 months, 2026-06 → 2027-12)

| Month | A — Theory | B — Data | C — System | D — Eval/Write |
|---|---|---|---|---|
| 1 | Pre-reg #1 | Extend sandbox | — | — |
| 2 | Lit corpus | k = 12 extractors live | — | — |
| 3 | Theorem 1 | Dependency pipeline | — | — |
| 4 | Theorem 2 | MATH-500 D done | — | — |
| 5 | — | — | DAJV aggregation | — |
| 6 | — | OOD benchmarks done | Calibration procedure | VERGE reimplemented |
| 7 | — | — | Library v0.1 | MATH-VF reimplemented |
| 8 | Pre-reg #2 | AbstentionBench done | Sandbox audit | Math-Rev reimplemented |
| 9 | — | — | — | — |
| 10 | — | — | — | All baseline runs |
| 11 | — | — | — | Power analysis |
| 12 | Lit review | — | Library v0.9 (PyPI) | — |
| 13 | — | — | — | — |
| 14 | — | Open dataset | — | Figures |
| 15 | Theory chapter | — | — | Draft v0.5 |
| 16 | — | — | Library v1.0 | External review |
| 17 | — | — | — | Submit ICML 2027 |
| 18 | — | — | — | Buffer / camera-ready |

**Hard milestones (no-slip):** months 4, 8, 12, 15, 17.

---

## 8. Reuse of existing X-SGRV artifacts

Do not throw out the existing project. Use it as Stage 0 evidence.

| Artifact | Reuse in DAJV |
|---|---|
| `src/xsgrv/extractor.py` (frozen prompt + parser) | Direct adoption; module `verifyensemble/extractors/prompt.py` |
| Sandbox harness, adversarial probe set | Direct adoption; module `verifyensemble/sandbox/` |
| 6-extractor cache on MATH-500, AIME 2025, CleanMath | Direct adoption as initial Stage 0 calibration data |
| Result JSONs (`results/exp*`) | Cited as preliminary motivating data in the paper |
| Same-model FAR experiment (Exp 5) | Reproduced as Ablation A8 across 3 open solvers |
| Pre-registration document `PRE_COMMIT_n500.md` | Cited; new pre-registration documents extend the same template |
| MathArena benchmark snapshots | Direct adoption |
| `references.bib` (audited, 3 fabricated removed) | Direct adoption; extend with VERGE, CARE, Kuai, Denisov-Blanch, etc. |

The existing X-SGRV paper draft itself (`paper/main_workshop.tex`, `paper/main_4page.tex`) is **not the new paper**. It may be retargeted to AI for Math @ ICML 2026 / MATH-AI @ NeurIPS 2026 as a preliminary report with explicit acknowledgment that DAJV is the main contribution (workshop = preliminary, main track = DAJV).

---

## 9. Release artifacts

At ICML 2027 submission, public release of:

1. `verifyensemble` v1.0 on PyPI (BSD-3 license)
2. Open dataset of 250k+ rows: (extractor, problem, candidate, verifier-code, accept/reject, gold) on HuggingFace
3. 12 × 12 dependency matrix dataset across 6+ benchmarks
4. All pre-registration documents in the GitHub repo with git commit hashes
5. Reproducibility script that re-runs all experiments end-to-end
6. Calibration-set + held-out split definitions (frozen with seed)
7. Sandbox security audit report
8. Power-analysis worksheet for every paper claim

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| VERGE Pareto-dominates on every benchmark | Medium | High | Reframe around calibration (VERGE does not provide); D becomes the primary contribution |
| Independence assumption holds empirically (H2 fails) | Low | Medium | Flip paper to "Independence as a measurement" — still publishable |
| Concentration bound is loose / trivial (H1 fails) | Medium | Medium | Fall back to empirical regression; theorem becomes "verified empirically" |
| Calibration does not transfer across benchmarks (H4 fails) | Medium | High | Restrict claim to in-distribution calibration; workshop submission, not main |
| Extractor APIs change mid-project (model deprecations) | High | Medium | Cache aggressively; pin model versions; include version-rotation experiment as a feature |
| Compute / API budget overrun | Medium | Medium | Quarterly budget reviews; soft-cap per workstream |
| Sandbox security vulnerability | Low | Critical | Pen-test in month 8; do not release until cleared |
| Two researchers depart project | Low | High | Pair-program workstreams; document everything in `~/Vault/wiki/projects/DAJV.md` |
| Race risk (another lab publishes Pivot A first) | Medium | Critical | Workstream A leads monthly arXiv scan; if scooped, pivot to mechanism (Pivot C) |
| ICML 2027 deadline slips internally | Medium | Medium | Fallback to NeurIPS 2027 (May–July submission window) |

---

## 11. Pre-registration commitments (committed before any new data collection)

### Pre-registration #1 — Dependency mapping (commit by month 1)

- **Hypotheses:** H1, H2, H6
- **Dataset:** MATH-500 calibration split (random seed 42, n=300 of 500); held-out test (remaining n=200)
- **Extractors:** the locked list of 12 above
- **Procedure:** for each pair (E_i, E_j), estimate κ_ij, joint_FP_ij, CIG_ij on calibration split; bootstrap CIs from 1,000 resamples
- **Decision:** H2 confirmed if ≥ 50% of pairs have joint-FP / independence-FP ≥ 3
- **Branches:** as in §2

### Pre-registration #2 — Deployment evaluation (commit by month 8)

- **Hypotheses:** H3, H4, H5
- **Test sets:** AIME 2024, AIME 2025, HMMT-Feb 2025, BRUMO 2025, SMT 2025, APEX 2025 (frozen list)
- **Comparison set:** every baseline in §5.3
- **Procedure:** for each baseline, compute risk–coverage curve, ECE, Brier; for each pair, McNemar mid-p; pre-specify multiple-comparison correction (Holm)
- **Decision:** H5 confirmed if dependency-aware aggregation Pareto-dominates every baseline on risk–coverage AUC across ≥ 4 of 6 OOD benchmarks
- **Branches:** as in §2

---

## 12. Failure pivot tree

```
[Month 3 gate: Theorem 1?]
├── PASS → continue Workstream A as planned
└── FAIL → Pivot C (mechanism study, see prior audit § Pivot C)

[Month 4 gate: Dependency matrix stable?]
├── PASS → continue
└── FAIL → enlarge calibration n by 3×; if still failing, drop κ for CIG-only

[Month 7 gate: Library passes integration?]
├── PASS → continue
└── FAIL → fitted-MLP empirical aggregation as fallback method

[Month 12 gate: VERGE head-to-head?]
├── DAJV competitive on ≥ 4 OOD benchmarks → continue main-track plan
└── VERGE Pareto-dominates → reframe around calibration only; workshop submission

[Month 15 gate: Draft red-team review?]
├── PASS → submit to ICML 2027
└── FAIL → 2-month rewrite cycle; submit to NeurIPS 2027 May window instead
```

---

## 13. Open questions to resolve in month 1

1. **Should the calibration set include solver-generated wrong candidates, gold-only candidates, or both?** Affects whether the rule calibrates for FP or for full coverage.
2. **Should `verifyensemble` cache the dependency matrix per-benchmark or use a single global D?** Affects H4 generalization claim.
3. **Should we include CARE as a baseline or as a building block of the aggregation rule?** CARE is closer to "method we extend" than "competitor we beat."
4. **Do we need an IRB / ethics review for the dataset release?** Likely no (no human subjects), but confirm with institutional review.
5. **Lab affiliation of the four researchers — does it constrain which extractors we can serve?** API agreements matter for some closed-weights models.

Resolve these before pre-registration #1 is committed.

---

## 14. Cross-references

- Prior project handoff: `HANDOFF_XSGRV_paper_2026-05-14.md`
- Audit that triggered this pivot: same handoff, audit section
- Existing X-SGRV draft (do not throw out): `paper/main_workshop.tex`, `paper/main_4page.tex`
- Existing X-SGRV codebase: `src/xsgrv/`, `src/eval/`, `src/audit/`
- Existing pre-registration template: `PRE_COMMIT_n500.md`
- Vault project page (to be created): `~/Vault/wiki/projects/DAJV.md`

---

## 15. Required reading (must read fully before starting any new work)

### Foundational

- Kuhn, Gal, Farquhar 2023, *Semantic Uncertainty*, ICLR. https://arxiv.org/abs/2302.09664
- Lightman et al. 2024, *Let's Verify Step by Step*, ICLR. https://arxiv.org/abs/2305.20050
- Zhou et al. 2024, *Don't Trust: Verify*, ICLR. https://arxiv.org/abs/2403.18120
- Wu et al. 2025, *Reasoning or Memorization?*, arXiv:2507.10532
- Balunović et al. 2025, *MathArena*, arXiv:2505.23281

### Pivot-A specific

- Kim, Garg, Peng, Garg 2025, *Correlated Errors in Large Language Models*, ICML. arXiv:2506.07962
- Kuai et al. 2026, *How Independent are LLMs?* arXiv (TBD ID — confirm)
- Zhao et al. 2026, *CARE: Confounder-Aware Aggregation for Reliable LLM Evaluation*, arXiv (TBD)
- Denisov-Blanch et al. 2026, *Consensus is Not Verification: Why Crowd Wisdom Strategies Fail for LLM Truthfulness*
- Singh et al. 2026, *VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning*

### Verifier / benchmark

- Kirichenko et al. 2025, *AbstentionBench*, arXiv
- Li et al. 2025, *VerifyBench*, arXiv
- Zheng et al. 2024, *ProcessBench*, arXiv:2412.06559
- Bagheri Nezhad et al. 2025, *SymCode*, arXiv:2510.25975
- Kuo Zhou et al. 2025, *MATH-VF*, arXiv (Step-Wise Formal Verification)
- Liang et al. 2024, *Math-Rev / Code-Rev*, arXiv
- Yu et al. 2024, *ReasonAgain*, arXiv
- Ospanov et al. 2025, *HERMES*, arXiv

### Methodology

- Fagerland, Lydersen, Laake 2013, *The McNemar Test for Binary Matched-pairs Data*, BMC Med Res Methodol 13:91

---

End of architecture document v0.1. Next revision: month 1, after pre-registration #1 is committed.
