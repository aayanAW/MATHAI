# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.15] - 2026-05-25 - ICML 2026 Workshop (Muslims in ML), 4-page body

### Changed
- Target: ICML 2026 main track -> **ICML 2026 Workshop (Muslims in ML)**.
- Body trimmed from 9 pages to **4 pages**:
  - Abstract: 41 lines -> 16 lines.
  - Introduction: 91 lines -> 38 lines.
  - Method: 84 lines -> 35 lines.
  - Theory: 110 lines -> 32 lines (full proofs in appendix).
  - Related work: 85 lines -> 50 lines.
  - Empirical: 478 lines -> 105 lines (head DAJV + BSIA + ProcessBench + H6 only).
  - Discussion: 56 lines -> 30 lines.
- All theorems remain (statement in body, proof in appendix).
- All k=12 numbers preserved (cross-modality table, H6, BSIA suite).
- Anonymous affiliation updated to workshop name.

### Output
- 4-page body, 15 pages PDF (4 body + 1 refs + 10 appendix).
- 103/103 tests still pass.

## [0.1.14] - 2026-05-25 - ICML 2026 target, anti-overclaim + anti-LLM cleanup pass

### Changed
- Paper target: ICML 2027 -> **ICML 2026**.
  - `main.tex` header, style comment, anonymous affiliation updated.
  - `icml2026.sty` is the official 2026 style file (not a proxy).
- Anti-overclaim sweep per stricter prompt rules:
  - All "first variant" claims softened to "the only one of N tested
    that ...".
  - "pioneered", "essentially perfect", "dramatic", "overwhelmingly",
    "plug-and-play", "operational gain ... first variant"
    softened to precise alternatives.
- Anti-LLM phrase sweep: removed filler such as "converged on a
  popular pattern", "directly contradicts", "natural reading".
- BSIA framing reconciled across sections: BSIA-iso 34% ECE
  reduction (matched precision) is the cleanest H7' win; BSIA-fixed
  $+4.4$pp coverage at $0.962$ prec; BSIA-nested-CV $+3.8$pp
  coverage at $0.972$ prec.
- All "X-SGRV" project-name references removed from paper.

## [0.1.13] - 2026-05-25 - k=12 four-lab ensemble, H6 reaches p=0.091

### Changed
- Together AI credit ($10) unlocked two new extractors on the
  serverless tier:
  - E13 Llama-3.3-70B-Turbo (Meta, new lab)
  - E14 Qwen3-235B-A22B-Instruct (Alibaba sibling to E09 Qwen3-Coder)
- Note: Llama-4 family, DeepSeek-R1-Distill, and Qwen2.5-72B-Turbo
  have migrated off Together AI serverless to dedicated endpoints,
  so substitutions were made.
- $k=12$ four-lab ensemble: 5 OpenAI, 4 Anthropic, 2 Alibaba, 1 Meta.
- All k-auto scripts wired to auto-include E13 + E14.

### Findings
- **Cross-modality independence strengthens at $k=12$:** B-full
  cross-LLM cross-modality $\kappa = 0.009$ across $132$ pairs
  (vs $0.017$ at $k=10$ on $90$ pairs). Block-sparsity finding
  REPLICATES + intensifies with cross-lab diversity.
- **H6 reaches $\boldsymbol{p = 0.091}$** at $k=12$ on both
  math175 and B-full -- closest to $\alpha = 0.05$ achieved at any
  $k$ in this study (vs $p = 0.32$ at $k=8$, $p = 0.42$ at $k=10$).
  Direction supports H6. The Meta lab added genuine cross-lab
  diversity ($\kappa_{\text{cross}}$ dropped from $0.662$ to $0.496$).
- **Robustness at $k=12$:** default DAJV catastrophically
  over-commits on B-full ($\text{cov} \approx 1.0$,
  $\text{prec} = 0.42$); BSIA-iso holds at precision $0.993$,
  ECE $0.078$.  Confirms the small-$n_{\text{cal}}$ failure mode.
- Headline operating-point comparison remains at $k=8$ in main
  text; appendix B.10 documents $k \in \{10, 12\}$ as robustness
  ablation.

### Tests
- 103/103 still pass.

## [0.1.12] - 2026-05-25 - k=10 ensemble finalised, robustness appendix added

### Changed
- Anthropic ensemble extended to 4 extractors:
  - E07 claude-sonnet-4-6 (cached)
  - E01A claude-opus-4-7 (cache complete, 330/330)
  - E02A claude-opus-4-6 (cache complete, 330/330)
  - E03A claude-haiku-4-5 (cache complete, 330/330)
- $k = 10$ ensemble total (5 OpenAI, 4 Anthropic, 1 Alibaba).
- Cross-modality measurement at $k=10$:
  - within-modality executable $\kappa = 0.767$ (B-full, 45 pairs)
  - cross-LLM cross-modality $\kappa = 0.017$ (B-full, 90 pairs)
  - Block-sparsity finding REPLICATES at $k=10$ with 50\% more pairs.
- H6 at $k=10$: $n_{\text{within}} = 16$, $n_{\text{cross}} = 29$,
  $p = 0.424$ (up from $p = 0.321$ at $k=8$).  Direction still
  supported but the gap shrinks because the new Anthropic-OpenAI
  cross-lab pairs are themselves highly correlated.

### Findings (new)
- **Robustness at $k=10$ small-$n_{\text{cal}}$ regime.**
  Default DAJV's copula MLE overfits the $k(k-1)/2 = 45$
  pairwise interactions when $n_{\text{cal}} = 122$ is below the
  Theorem~\ref{thm:sample} threshold of $\approx 225$; test
  precision on math175 collapses ($0.984 \to 0.571$).  BSIA-iso's
  post-hoc isotonic recalibration is robust to this overfitting:
  precision $0.952$, ECE $0.087$, coverage $0.551$.  New appendix
  section ``Robustness at $k=10$'' (B.10) documents this and
  recommends BSIA-iso for deployments scaling beyond $n_{\text{cal}}
  / k$ ratios supported by the Hoeffding bound.

## [0.1.11] - 2026-05-25 - ProcessBench expanded to n=150, Anthropic suite extended

### Changed
- ProcessBench-Math validation re-run at $n=150$ (was $30$) on
  3 OpenAI extractors. New result: DAJV TP=5 vs naive unanimous
  TP=1 ($5\times$ lift), precision $1.000$ matched, ECE $0.114$
  vs naive $0.267$ ($57\%$ reduction, up from $45\%$ at $n=30$).
  Confirms cross-distribution Pareto dominance at larger sample.
- Anthropic ensemble extended: Opus 4.6 (E02A) and Haiku 4.5
  (E03A) extractor caches in progress; combined with Opus 4.7
  (E01A) and Sonnet 4.6 (E07) gives 4 Anthropic extractors and
  $C(4,2) = 6$ within-Anthropic pairs.

### Paper updates
- ProcessBench section ($n=60$ test split, $90$ cal): $5\times$
  coverage lift over naive unanimous, $57\%$ ECE reduction.
- ProcessBench table updated to new numbers.

## [0.1.10] - 2026-05-25 - BSIA suite (fixed/temp/iso/ensemble), Theorem 3, k=10 in-flight

### Added
- BSIA variants in ``verifyensemble/aggregate/dajv_xmod.py``:
  - ``fit_bsia_temperature``: post-hoc temperature scaling (NLL min
    over a grid).
  - ``_isotonic_pav`` + ``_isotonic_predict``: pure-Python PAV
    isotonic regression for monotone recalibration.
  - ``bsia_isotonic_aggregate``: BSIA with isotonic post-hoc.
  - ``bsia_ensemble_aggregate``: Bayesian model averaging across
    BSIA-fixed, BSIA-temp, BSIA-iso variants.
- ``Theorem 3`` (BSIA sample complexity under block-sparsity):
  saves $k(k-2) = 48$ parameters at $k=8$ over the saturated
  Ising; statement in body theory section, full proof in
  ``paper/sections/proofs.tex``.
- Anthropic extractors restored: Opus 4.7 (E01A) cache done;
  Opus 4.6 (E02A) and Haiku 4.5 (E03A) caching in background.
  Wrapper patched for the ``temperature`` deprecation on Opus 4.7+.

### Findings
- BSIA suite at $k=8$ math175 (10-seed mean):
  - default DAJV : cov $0.581$, prec $0.984$, ECE $0.111$
  - BSIA fixed    : cov $0.625$, prec $0.962$, ECE $0.102$ (+4.4pp cov)
  - BSIA temp     : cov $0.443$, prec $0.987$, ECE $0.083$ (+0.3pp prec)
  - BSIA iso      : cov $0.557$, prec $0.984$, ECE $\mathbf{0.073}$ (-34% ECE)
  - BSIA ens.     : cov $0.555$, prec $0.984$, ECE $0.081$
- BSIA isotonic delivers a $34\%$ ECE reduction over default DAJV
  at matched precision -- the cleanest calibration win from
  cross-modality independence, even when the coverage criterion
  itself is not crossed.

### Tests
- 98 -> 103 (added temperature, isotonic, ensemble, PAV monotonicity).

## [0.1.9] - 2026-05-25 - k=8 ensemble (Opus 4.7 cache complete), all k=8 experiments re-run

### Changed
- Opus 4.7 (E01A) extractor cache complete (330/330). All k-auto
  scripts now include it: cross-modality measurement, xmod_dajv,
  bsia_sensitivity, 8extractor_aggregation, h6_significance,
  cross_modality_heatmap, hybrid_modality, modality_conjunctive_dajv,
  final_k_analysis.
- Empirical results at $k=8$:
  - Cross-modality cross-LLM $\kappa$ median $0.017$ (B-full),
    $0.062$ (math175) -- replicates the block-sparsity finding.
  - Within-modality executable $\kappa$ median $0.720$ (B-full).
  - H6 within-vs-cross lab: $n_{\text{within}} = 11$ (was 10),
    $n_{\text{cross}} = 17$ (was 11), median within $0.664$ vs
    cross $0.565$ on B-full, $p = 0.321$. Direction preserved;
    significance still insufficient.
  - BSIA at $k=8$ nested-CV math175: cov $0.619$ ($+3.8$pp vs
    default $0.581$), prec $\mathbf{0.972}$ (above the
    pre-registered $0.97$ floor), ECE $0.086$ (vs default $0.111$).
- Paper updated:
  - Abstract: "7-extractor" -> "8-extractor", BSIA paragraph
    re-tightened.
  - Introduction: ensemble size, contribution bullet for BSIA.
  - Empirical: cross-modality table values, H6 table (n=11/17),
    H6 figure caption.
  - Extended results: BSIA H7' table re-tabulated at $k=8$.
  - 9-page body preserved.

## [0.1.8] - 2026-05-25 - BSIA: first Pareto-positive H7' variant, Opus 4.7 extractor restored

### Added
- `BlockSparseIsingCalibration` / `bsia_aggregate` in
  `verifyensemble/aggregate/dajv_xmod.py` — block-sparse Ising
  aggregator that respects the empirical $\kappa$ block-sparsity:
  dense within-modality cross-LLM (modeled via second-order Pearson
  corrections), per-LLM cross-modality (modeled via $2 \times 2$
  cells), and *pinned-to-zero* cross-LLM cross-modality interactions.
  The natural Bahadur factorization of the H7' problem.
- `scripts/run_bsia_sensitivity.py` — nested cross-validation
  ($3$-fold inner, $10$-seed outer) over
  $(\rho_{\text{shrink}}, \text{smooth}) \in \{0, .25, .5, .75\}
  \times \{.1, .5, 1.0\}$.  Selects best HP per outer split.
- BSIA is now part of the `scripts/run_xmod_dajv.py` evaluation
  harness; reported in Appendix B.10's H7' table alongside the four
  earlier negatives.
- `E01A_claude_opus_4_7` entry in the extractor registry. Anthropic
  Opus 4.7 deprecates the `temperature` parameter; the
  `anthropic_call` wrapper now retries without `temperature` on a
  $400$ `invalid_request_error` indicating deprecation.

### Findings
- **H7' partial pass.** BSIA Pareto-dominates default DAJV at fixed
  hyperparameters ($\rho_{\text{shrink}} = 0.5$, smooth $= 0.5$):
  $+2.0$pp coverage ($0.574 \to 0.594$), precision held at $0.985$,
  ECE lower ($0.091 \to 0.084$). With nested-CV HP selection,
  coverage reaches $+4.5$pp ($0.619$) at $0.969$ precision and
  $0.083$ ECE. Below the pre-registered $\geq 5$pp / $\geq 0.97$
  threshold but the first variant of five tested to strictly
  improve on (coverage, precision, ECE) at fixed HP.

### Tests
- 90 → 98 (+8 new BSIA tests: calibration consistency, structural
  sparsity, commit/abstain branches, None-vote marginalization,
  shrinkage extremes, length-mismatch error).

## [0.1.7] - 2026-05-25 - H7' attempts (negative), VERGE-Z3 faithful replication, ProcessBench validation

### Added
- `verifyensemble/aggregate/dajv_xmod.py` — three cross-modality
  DAJV variants:
  - `XmodCalibration` / `xmod_aggregate`: factorized Bayes over two
    per-modality DAJVs.
  - `xmod_agreement_aggregate`: cross-modality agreement gate.
  - `XmodJointCalibration` / `xmod_joint_aggregate`: per-LLM joint
    $2 \\times 2$ Naive Bayes.
- `verifyensemble/aggregate/verge_z3.py` — faithful VERGE replication
  using Z3 SMT for the formal-verification stage; integrates with the
  existing VERGE-proxy MCS fallback. Requires the `z3-solver` PyPI
  package.
- `scripts/run_xmod_dajv.py` — evaluates all three xmod variants
  against default DAJV on the $7$-extractor cache (10-seed average).
- `scripts/run_verge_z3_compare.py` — DAJV vs VERGE-proxy vs
  VERGE-Z3 head-to-head.
- `scripts/run_processbench_validation.py` (gold-free extension) +
  `scripts/run_processbench_dajv_eval.py` — runs 4 OpenAI extractors
  on a 50-problem ProcessBench math subset and evaluates DAJV in
  gold-free deployment mode.

### Findings
- **H7' attempts: all three NEGATIVE.** None of the three
  cross-modality variants beats default DAJV at the default operating
  threshold. The closest (xmod-joint NB) matches default on
  (coverage, precision) but has worse ECE. Reason: cross-modality
  independence is real but is orthogonal to cross-LLM dependence,
  which the new aggregators violate. Documented as a pre-registered
  negative result. See Appendix B.10 of the paper.
- **VERGE-Z3 faithful replication: DAJV still dominates.** On math175,
  DAJV reaches cov $0.62$ / prec $0.97$; VERGE-Z3 alone reaches
  $0.49 / 0.96$; VERGE-Z3 + MCS reaches $0.55 / 0.97$.
  $+7.6$pp coverage gap for DAJV at matched precision.
- **ProcessBench-Math cross-distribution validation: DAJV
  Pareto-dominates.** 3-extractor (gpt-5-mini, gpt-4o, gpt-4.1) on a
  balanced 30-problem subset ($15$ correct + $15$ wrong final
  answers). Test split ($n_{\text{test}} = 12$): DAJV cov $0.25$ vs
  naive unanimous $0.08$ ($3\times$); precision $1.000$ matched;
  specificity $1.000$ matched; ECE drops from $0.333$ to $0.183$
  ($45\%$ lower). Zero false positives. Gold-free deployment mode.
  See \S6.10 in the paper.

### Tests
- 84 / 84 still passing.

## [0.1.6] - 2026-05-25 - Pivot to cross-modality independence framing (Option A)

### Changed (framing only; code unchanged)
- **New title:** "Two Modalities Are Better Than One: Cross-Modality
  Independence in Executable Spec-Grounded LLM-Jury Verification for
  Math Reasoning."
- **New abstract:** leads with cross-modality independence as headline
  finding ($\kappa \approx 0.02$ cross-modality, $0.70$
  within-modality, $42$ cross-LLM pairs); demotes Theorem 1 to
  "specialization of \citet{balasubramanian2026ising} to binary
  executable votes."
- **Introduction restructured:** contributions reordered so
  cross-modality independence is #1; dependency-aware aggregation is
  #2.
- **Empirical reordered:** cross-modality independence promoted to
  third subsection (right after H2 independence-broken).
  Seed-stability, solver-rotation, A3 ablation, LOO ablation, joint-FP
  scatter, cross-modality box plot, and Theorem 1 numerical
  validation moved to appendix to fit the 9-page ICML body limit.
- **Discussion tightened:** "Why CARE underperforms" subsection
  removed; "Limitations" + "Positioning against concurrent work"
  compressed to two short paragraphs.

### Added
- `PRE_REGISTRATION_v2.md` — locks the new framing with hypotheses
  H1'--H7' and the operational-exploitation experiment for future
  work.
- README headline rewritten around the cross-modality finding.

### Pages
- Body: $9$ pages (ICML main-track conformant).
- Total with appendix: $17$ pages.

## [0.1.5] - 2026-05-25 - Scale-up to 7 extractors (gpt-5 added)

### Added
- E10S gpt-5-2025-08-07 with `reasoning_effort=low`: 330 records.
- 7-extractor H6 significance test: 10 within-OpenAI pairs vs 11 cross-lab.
- 7-extractor cross-modality measurement: 21 within-modality pairs, 42 cross-LLM cross-modality.

### Findings (k=7 update)
- **H6:** within-OpenAI κ median 0.664 [CI95 0.50, 1.00]; cross-lab κ median 0.565 [CI95 0.33, 0.66]; permutation $p$ = 0.243.
- **Cross-modality:** within-modality executable κ median 0.720 on B-full; cross-modality 0.021 (still order-of-magnitude smaller).

### Tests
- Still 84 / 84 passing.

## [0.1.4] - 2026-05-25 - Scale-up to 6 extractors + cross-modality finding + VERGE proxy

### Added
- New OpenAI extractors collected on the 330-problem cache:
  E08S gpt-4o-2024-11-20, E04S gpt-4.1-2025-04-14 (both 330/330).
  E10S gpt-5-2025-08-07 in progress at end of session (reasoning_effort=low).
- `verifyensemble/aggregate/verge_proxy.py` — VERGE-style baseline.
- `scripts/run_hybrid_modality.py` — cross-modality κ analysis.
- `scripts/run_hybrid_aggregation.py` — pure-LLM vs hybrid struct×exec.
- `scripts/run_new_extractors.py` — frozen-prompt runner for new extractors with checkpointing.
- `scripts/run_verge_proxy_compare.py` — DAJV vs VERGE-proxy head-to-head.
- `scripts/run_8extractor_aggregation.py` — multi-extractor aggregation harness with H6 within/cross lab analysis.
- `scripts/run_h6_significance.py` — bootstrap CIs + permutation test for H6.
- `verifyensemble/extractors/api_wrappers.py` — handles gpt-5 / o-series `max_completion_tokens` and `reasoning_effort`; Gemini `thinking_budget=0` for 2.5 series.
- Paper sections §Cross-modality independence and §VERGE-proxy h2h (empirical); §Naive 8-signal hybrid and §VERGE-proxy details (appendix).

### Findings
- **Cross-modality independence (key new result).** On the 6-extractor cache, within-modality (LLM-script vs LLM-script) executable κ median 0.72; cross-modality (LLM-script signal vs script-execution signal) κ median 0.04. Pure LLM-jury aggregators (Kuai 2026, CARE, Balasubramanian Ising) operate within a single modality.
- **Naive 8-signal hybrid does not beat default DAJV** at default operating threshold. Independence is necessary, not sufficient: weaker per-modality marginals dominate.
- **VERGE-proxy head-to-head.** DAJV math175 (cov 0.623 / prec 0.970) slightly Pareto-dominates VERGE-proxy (0.585 / 0.968).
- **H6 with 6 extractors.** Within-lab (n=6 OpenAI pairs) median κ = 0.664; cross-lab (n=9) median κ = 0.565. Bootstrap CIs overlap; permutation test p=0.378 single-sided. Direction supported, still inconclusive. Adding gpt-5 (in progress) → k=7 → 10 within-lab pairs.

### Skipped this session
- **AbstentionBench.** HF dataset uses a script (deprecated in
  `datasets` 4.x). Integration requires the inspect_evals harness.
  Out of scope.
- **ProcessBench.** Schema (per-step error annotations on solver
  reasoning chains) does not provide gold final answers for
  wrong-final rows, so the binary verification task DAJV solves does
  not have a direct mapping. A "gold-free" DAJV variant would be
  required; deferred.

### Hard blockers (user-side)
- Together AI account requires prepaid credit top-up (HTTP 402). Blocks E02 / E04 / E10 (Llama-4-Maverick, DeepSeek-R1, Qwen2.5-72B).
- Gemini 2.5 Flash daily request cap exhausted; either bill-enable key or wait ~24h. Pro tier requires billing.
- Anthropic Opus 4.7 key not provided in this session (user said "later").

### Fixed
- ruff B023 closure-over-loop-variable in `run_hybrid_aggregation.py` (moved `agg` to module scope).
- mypy type errors on OpenAI / Google SDK call surfaces.

### Tests
- 82 / 82 passing (+19: 7 verge_proxy, 4 new_aggregators, 8 parser);
  ruff + mypy clean.
- Module coverage: aggregate/* 89-100%, theory/* 82-100%,
  dependency/* 81-96%, evaluation/* 74-95%.

## [0.1.3] - 2026-05-25 - Citation audit + novelty verification + new ablations

### Added
- `scripts/run_copula_ablation.py` — second-order copula vs
  independence-only Bayesian aggregator (10-seed average).
- `scripts/run_threshold_sensitivity.py` — DAJV commit-threshold
  sweep over $\tau \in [0.50, 0.995]$ on every benchmark subset.
- `paper/sections/extended_results.tex` — Copula-order ablation table
  + threshold sensitivity table.
- `paper/sections/related_work.tex` — explicit concurrent-work
  paragraph citing Kuai, CARE, Balasubramanian Ising, VERGE.
- `paper/sections/discussion.tex` — Positioning against concurrent
  work (honest scope claim, no priority overreach).
- `NOVELTY_REPORT_2026-05-25.md` — mandatory pivot-novelty assessment.
- BibTeX entry `balasubramanian2026ising` (arXiv:2601.22336).

### Findings
- **Copula-order ablation.** Second-order term reduces ECE by ~28%
  on math175 (0.166 → 0.120) and ~16% on Group B full
  (0.099 → 0.083). Coverage and precision at default operating point
  unchanged → the dependency term refines posterior shape rather than
  flipping commit decisions.
- **Threshold sensitivity.** DAJV operating point is stable across
  $\tau \in [0.70, 0.97]$; only $\tau \geq 0.99$ trims coverage
  noticeably, and precision is unchanged.
- **Novelty assessment: 3-4 / 10.** Three concurrent 2026 papers
  (Kuai, CARE, Balasubramanian Ising) cover the core claims. See
  `NOVELTY_REPORT_2026-05-25.md` for sharpening recommendations.

### Fixed
- **Citation audit.** Verification against arXiv flagged 6
  bibtex entries with fabricated first authors (`xin2025sbsc`,
  `stepco2025`, `cosc2024`, `lemma2025`, `cannotspot2025`,
  `codeprm2025`) and 1 duplicate (`mathvf2025` = `kuoZhou2025mathvf`
  with fabricated author). All replaced with verified author lists
  and arXiv IDs. The other 17 audited entries had only missing arXiv
  IDs and were corrected.
- Paper compiles cleanly with no overfull boxes (was: 4 overfull
  warnings in empirical.tex tables); used `\setlength{\tabcolsep}`
  and column-name abbreviations.
- `scripts/run_copula_ablation.py` line 135: switched population
  variance to Bessel-corrected sample variance for n=10 seeds.

### Tests
- 63/63 still passing.
- ruff + mypy clean across 30 source files + 2 new scripts.

## [0.1.2] - 2026-05-25 - PRM head-to-head + H6 + solver rotation

### Added
- `scripts/run_prm_head_to_head.py` — risk-coverage AUC vs Qwen-PRM-7B and
  Skywork-o1-Open-PRM-7B across math175, AIME, CleanMath.
- `scripts/run_prm_matched_coverage.py` — point precision + McNemar mid-p
  comparison at matched coverage.
- `scripts/run_within_lab_dependency.py` — H6 within-lab vs cross-lab
  pairwise κ analysis using the gpt-oss-120B + gpt-5-mini OpenAI pair.
- `scripts/run_solver_rotation.py` — DAJV with DeepSeek-V3 as alternate
  solver (exp44 cache).
- 4 new paper figures: PRM head-to-head AUC, PRM matched-coverage bars,
  within-lab vs cross-lab bars, sample complexity (Theorem 2).
- 3 new empirical subsections: PRM head-to-head (H5 PASS),
  within-lab (H6 partial), solver-rotation control.

### Findings
- **H5 PASS.** DAJV beats Qwen-PRM-best by 35× on AIME and 88× on CleanMath
  in risk-coverage AUC. On in-distribution math175 DAJV ties Qwen-PRM-last.
- **H6 partial.** Within-lab pair κ = 0.79 > cross-lab median 0.65, but
  the strongest cross-lab pair (κ = 1.0) exceeds within-lab. Direction
  supported; statistically inconclusive at n_within = 1.
- **Solver rotation.** With DeepSeek-V3 as solver, DAJV achieves
  26/26 = 1.000 precision at 26% coverage with ECE 0.049, vs naive
  23/23 at 23% with ECE 0.285 → 83% ECE reduction.

### Tests
- Still 61/61 (no new unit tests needed; new scripts are integration-level).
- ruff + mypy clean.

## [0.1.1] - 2026-05-25 - Stability + lint

### Added
- `scripts/run_seed_stability.py` — 10-seed stability experiment on math175.
- `scripts/run_risk_coverage_bands.py` — risk-coverage curves with bands.
- `tests/test_properties.py` — 11 hypothesis-based property tests.
- `pyproject.toml` ruff + mypy config.
- Seed-stability table + figure in the paper.

### Fixed
- Code-review fixes from cavecrew-reviewer pass (executor.py temp-file
  leak, dajv.py/matrix.py dim-2 validation, validation tests).
- mypy clean across all 30 source files.
- ruff clean.

### Tests
- 50 unit + 11 property-based = 61 / 61 pass.

## [0.1.0] - 2026-05-25 - Initial release

### Added
- Core library `verifyensemble` with sandbox, extractors, dependency,
  aggregate, theory, evaluation, utils subpackages.
- Theorem 1 (concentration bound on joint FP under measured pairwise
  dependency) with proof and numerical implementation.
- Theorem 2 (calibration sample-complexity) with proof and numerical
  implementation.
- DAJV second-order copula aggregation rule.
- Naive consensus (unanimous, majority) and CARE-style logistic
  baselines.
- Cohen's $\kappa$, joint-FP rate, and Cumulative Information Gain
  dependency estimators.
- Bootstrap confidence intervals over every dependency matrix entry.
- Risk-coverage, ECE, Brier, McNemar mid-p evaluation harness.
- 46-test unit test suite (all passing).
- End-to-end script set:
  - `scripts/run_dependency_mapping.py`
  - `scripts/run_synthetic_validation.py`
  - `scripts/run_aggregation_comparison.py`
  - `scripts/run_figures.py`
  - `scripts/run_ablations.py`
- Six PDF figures embedded in the paper.
- Empirical findings on cached X-SGRV extractor outputs:
  - Median pairwise Cohen's $\kappa = 0.79$ (math175 wrong subset).
  - Median joint-FP / independence-bound ratio = $15.7\times$.
  - DAJV ECE = $0.111$ vs naive unanimous = $0.208$ (47% reduction).
  - DAJV precision $0.97$ at $62\%$ coverage vs naive $0.96$ at $49\%$.
  - Theorem 1 validated in 45 of 45 Monte Carlo trials.
- 12-page LaTeX paper draft (`paper/main.pdf`, 220 KB) with theory,
  empirical results, ablations, discussion, and full proofs.
- Pre-registration document v1 (H1--H6 hypotheses, branches A--E,
  decision thresholds, locked parameters).
- Branch B activated: H4 (cross-benchmark dependency transfer) failed;
  reframed contribution around in-distribution calibration with
  per-benchmark recalibration.

### Known issues
- No new LLM API calls in this release. The empirical results use
  cached extractor outputs from the X-SGRV project. New API calls
  are required to test H5 (vs PRMs at matched coverage) and H6
  (within-lab vs cross-lab dependency).
- VERGE (Singh 2026) is not yet reimplemented as a baseline.
- AbstentionBench, VerifyBench, and ProcessBench evaluation harnesses
  are wired but not yet run on the DAJV ensemble.

### Pre-registration status
- H1 (Theorem 1 holds): PASS
- H2 (independence broken): PASS (median ratio $15.7\times$)
- H3 (ECE reduction $\geq 30\%$): PASS (47%)
- H4 (cross-benchmark transfer): FAIL (per pre-registered Branch B)
- H5 (Pareto dominance): PASS on math175 vs naive baselines; PRM
  comparison pending API calls.
- H6 (within-lab vs cross-lab dependency): PENDING (no within-lab
  pairs in current cache).
